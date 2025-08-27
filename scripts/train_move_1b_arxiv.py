#!/usr/bin/env python3
"""
MoVE 1B Model Training Script for ArXiv Dataset

Optimized training pipeline for 1B parameter MoVE model using ArXiv datasets
from Common Pile, specifically designed for RTX 4090 (16GB VRAM).

Features:
- Memory-optimized training for RTX 4090
- Mixed precision training (FP16)
- Gradient accumulation for effective large batch sizes
- Advanced learning rate scheduling
- Comprehensive logging and checkpointing
- Integration with ArXiv dataset processor
"""

import os
import sys
import json
import argparse
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import MoVE components
from move_large import MoVELarge, create_move_1b
from scripts.prepare_arxiv_dataset import prepare_arxiv_dataset, estimate_1b_training_requirements

class MoVE1BTrainer:
    """Trainer for 1B MoVE model on ArXiv dataset."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Mixed precision setup
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler()
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        
        # Logging setup
        self.log_interval = config.get('log_interval', 100)
        self.eval_interval = config.get('eval_interval', 1000)
        self.save_interval = config.get('save_interval', 2000)
        
        # Checkpointing
        self.output_dir = config.get('output_dir', 'checkpoints/move_1b_arxiv')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Memory optimization
        self._setup_memory_optimization()
        
        print(f"MoVE 1B Trainer initialized")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay."""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer_type = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('learning_rate', 1e-4)
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine_with_warmup')
        num_training_steps = len(self.train_dataloader) * self.config.get('num_epochs', 3)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        if scheduler_type == 'cosine_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'linear_with_warmup':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 1e-4),
                total_steps=num_training_steps,
                pct_start=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_memory_optimization(self):
        """Setup memory optimization techniques."""
        # Enable gradient checkpointing if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Set memory efficient attention if available
        if hasattr(self.model, 'set_memory_efficient_attention'):
            self.model.set_memory_efficient_attention(True)
            print("Memory efficient attention enabled")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute language modeling loss."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        if self.use_amp:
            with autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Compute loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        self.model.train()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        self.optimizer.zero_grad()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                loss = self.compute_loss(batch)
                
                batch_size = batch['input_ids'].size(0)
                seq_len = batch['input_ids'].size(1)
                tokens = batch_size * seq_len
                
                total_loss += loss.item() * tokens
                total_tokens += tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"New best checkpoint saved: {best_path}")
        
        # Save periodic checkpoint
        if self.global_step % self.save_interval == 0:
            periodic_path = os.path.join(self.output_dir, f'checkpoint_step_{self.global_step}.pt')
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded: epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 3)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Total training steps: {len(self.train_dataloader) * num_epochs}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Optimizer step (with gradient accumulation)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = self.optimizer.param_groups[0]['lr']
                        
                        log_data = {
                            'train_loss': avg_loss,
                            'learning_rate': lr,
                            'epoch': epoch,
                            'global_step': self.global_step
                        }
                        
                        # Log to wandb if available
                        if wandb.run:
                            wandb.log(log_data)
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{lr:.2e}',
                            'step': self.global_step
                        })
                    
                    # Evaluation
                    if self.global_step % self.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        
                        print(f"\nEvaluation at step {self.global_step}:")
                        print(f"Validation loss: {eval_metrics['val_loss']:.4f}")
                        print(f"Validation perplexity: {eval_metrics['val_perplexity']:.2f}")
                        
                        # Log evaluation metrics
                        if wandb.run:
                            wandb.log(eval_metrics)
                        
                        # Save best checkpoint
                        is_best = eval_metrics['val_loss'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = eval_metrics['val_loss']
                        
                        self.save_checkpoint(is_best=is_best)
                        
                        # Memory cleanup
                        torch.cuda.empty_cache()
                    
                    # Periodic checkpoint
                    elif self.global_step % self.save_interval == 0:
                        self.save_checkpoint()
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            print(f"\nEnd of epoch {epoch+1}:")
            print(f"Validation loss: {eval_metrics['val_loss']:.4f}")
            print(f"Validation perplexity: {eval_metrics['val_perplexity']:.2f}")
            
            # Save checkpoint
            is_best = eval_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = eval_metrics['val_loss']
            
            self.save_checkpoint(is_best=is_best)
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")

def create_dataloaders(
    dataset_path: str,
    batch_size: int = 2,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    print(f"Train dataset size: {len(train_dataset):,}")
    print(f"Validation dataset size: {len(val_dataset):,}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader

def setup_wandb(config: Dict[str, Any], project_name: str = "move-1b-arxiv"):
    """Setup Weights & Biases logging."""
    if config.get('use_wandb', False):
        wandb.init(
            project=project_name,
            config=config,
            name=f"move-1b-{config.get('run_name', 'arxiv')}"
        )
        print("Weights & Biases logging enabled")
    else:
        print("Weights & Biases logging disabled")

def main():
    parser = argparse.ArgumentParser(description='Train MoVE 1B model on ArXiv dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, default='data/arxiv_dataset',
                       help='Path to processed ArXiv dataset')
    parser.add_argument('--prepare_dataset', action='store_true',
                       help='Prepare ArXiv dataset before training')
    parser.add_argument('--abstracts_samples', type=int, default=50000,
                       help='Number of abstract samples for dataset preparation')
    parser.add_argument('--papers_samples', type=int, default=10000,
                       help='Number of paper samples for dataset preparation')
    
    # Model arguments
    parser.add_argument('--model_config', type=str, default='1b',
                       choices=['1b'], help='Model configuration')
    parser.add_argument('--tokenizer', type=str, 
                       default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Tokenizer to use')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine_with_warmup',
                       choices=['cosine_with_warmup', 'linear_with_warmup', 'onecycle'],
                       help='Learning rate scheduler')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='checkpoints/move_1b_arxiv',
                       help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (steps)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Evaluation interval (steps)')
    parser.add_argument('--save_interval', type=int, default=2000,
                       help='Checkpoint saving interval (steps)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for logging')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare dataset if requested
    if args.prepare_dataset:
        print("Preparing ArXiv dataset...")
        dataset, metadata = prepare_arxiv_dataset(
            output_dir=args.dataset_path,
            abstracts_samples=args.abstracts_samples,
            papers_samples=args.papers_samples,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length
        )
        
        # Estimate training requirements
        estimate_1b_training_requirements(
            dataset_size=len(dataset['train']),
            tokens_per_example=args.max_length
        )
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset not found at {args.dataset_path}. Use --prepare_dataset to create it.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print(f"Creating MoVE 1B model...")
    model = create_move_1b(
        vocab_size=len(tokenizer),
        max_seq_len=args.max_length
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Training configuration
    config = {
        'model_config': args.model_config,
        'tokenizer': args.tokenizer,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'max_grad_norm': args.max_grad_norm,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'use_amp': args.use_amp,
        'output_dir': args.output_dir,
        'log_interval': args.log_interval,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'use_wandb': args.use_wandb,
        'run_name': args.run_name,
        'seed': args.seed
    }
    
    # Setup logging
    setup_wandb(config)
    
    # Create trainer
    trainer = MoVE1BTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.save_checkpoint()
        raise
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == '__main__':
    main()