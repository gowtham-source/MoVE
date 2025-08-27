#!/usr/bin/env python3
"""
Memory-Optimized MoVE 500M Model Training Script for ArXiv Dataset

Optimized training pipeline for 500M parameter MoVE model using ArXiv datasets
from Common Pile, specifically designed for RTX 4090 (16GB VRAM) with aggressive
memory optimization to prevent CUDA OOM errors.
"""

import os
import sys
import json
import argparse
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Set memory optimization environment variables before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import MoVE components
from move_large import MoVELarge
from scripts.optimize_1b_for_rtx4090 import create_optimized_move_model, get_optimized_training_config
from scripts.prepare_arxiv_dataset import prepare_arxiv_dataset

class MemoryOptimizedMoVE500MTrainer:
    """Memory-optimized trainer for 500M MoVE model on ArXiv dataset."""
    
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
        
        # Memory optimization setup - CRITICAL
        self._setup_aggressive_memory_optimization()
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler()
        
        # Gradient accumulation - increased for smaller batches
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
        
        # Logging setup
        self.log_interval = config.get('log_interval', 50)
        self.eval_interval = config.get('eval_interval', 500)
        self.save_interval = config.get('save_interval', 1000)
        
        # Checkpointing
        self.output_dir = config.get('output_dir', 'checkpoints/move_500m_arxiv')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Memory-Optimized MoVE 500M Trainer initialized")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Print memory info
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def _setup_aggressive_memory_optimization(self):
        """Setup aggressive memory optimization techniques."""
        print("Setting up aggressive memory optimization...")
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            print("✓ Gradient checkpointing enabled (alternative method)")
        
        # Set memory efficient attention if available
        if hasattr(self.model, 'set_memory_efficient_attention'):
            self.model.set_memory_efficient_attention(True)
            print("✓ Memory efficient attention enabled")
        
        # Enable compilation for memory efficiency (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✓ Model compiled for memory efficiency")
            except Exception as e:
                print(f"⚠ Model compilation failed: {e}")
        
        # Set memory fraction to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
            print("✓ GPU memory fraction set to 95%")
        
        # Clear cache and force garbage collection
        self._clear_memory_cache()
        
        # Set environment variables for memory optimization
        torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
        torch.backends.cudnn.deterministic = True
        print("✓ CUDNN optimizations configured")
    
    def _clear_memory_cache(self):
        """Clear memory cache and force garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
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
        learning_rate = self.config.get('learning_rate', 1.2e-4)
        
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
                max_lr=self.config.get('learning_rate', 1.2e-4),
                total_steps=num_training_steps,
                pct_start=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute language modeling loss with memory optimization."""
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
        
        # Forward pass with memory optimization
        if self.use_amp:
            with autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
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
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        
        # Clear intermediate tensors
        del shift_logits, shift_labels
        if 'outputs' in locals():
            del outputs
        if 'logits' in locals():
            del logits
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step with memory optimization."""
        self.model.train()
        
        try:
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            loss_value = loss.item() * self.gradient_accumulation_steps
            
            # Clear batch from memory
            del batch
            
            return loss_value
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM in train_step: {e}")
            self._clear_memory_cache()
            raise e
    
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
        
        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train(self):
        """Main training loop with aggressive memory management."""
        print("Starting memory-optimized training...")
        
        num_epochs = self.config.get('num_epochs', 3)
        total_steps = len(self.train_dataloader) * num_epochs
        
        print(f"Training for {num_epochs} epochs")
        print(f"Total training steps: {total_steps}")
        
        # Clear memory before training
        self._clear_memory_cache()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Progress bar
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                total=len(self.train_dataloader)
            )
            
            for step, batch in enumerate(pbar):
                try:
                    # Training step
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Gradient accumulation
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_step()
                        self.global_step += 1
                        
                        # Clear cache periodically
                        if self.global_step % 10 == 0:
                            self._clear_memory_cache()
                    
                    # Update progress bar
                    avg_loss = epoch_loss / num_batches
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
                    })
                    
                    # Logging
                    if self.global_step % self.log_interval == 0:
                        self._log_metrics({
                            'train_loss': avg_loss,
                            'learning_rate': current_lr,
                            'epoch': epoch,
                            'global_step': self.global_step,
                            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3
                        })
                    
                    # Evaluation
                    if self.global_step % self.eval_interval == 0:
                        val_loss = self.evaluate()
                        self._log_metrics({'val_loss': val_loss})
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint('best_model')
                    
                    # Save checkpoint
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(f'checkpoint_step_{self.global_step}')
                
                except torch.cuda.OutOfMemoryError as e:
                    print(f"\nCUDA OOM at step {step}: {e}")
                    print("Clearing cache and continuing...")
                    self._clear_memory_cache()
                    continue
                
                except Exception as e:
                    print(f"\nError at step {step}: {e}")
                    self._clear_memory_cache()
                    raise e
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
            
            # Clear cache at end of epoch
            self._clear_memory_cache()
        
        print("Training completed!")
        return self.best_val_loss
    
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    loss = self.compute_loss(batch)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clear batch
                    del batch, loss
                    
                except torch.cuda.OutOfMemoryError:
                    print("OOM during evaluation, skipping batch")
                    self._clear_memory_cache()
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'{name}.pt')
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to console and wandb if available."""
        # Console logging
        metric_str = ' | '.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                                for k, v in metrics.items()])
        print(f"Step {self.global_step} | {metric_str}")
        
        # WandB logging
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=self.global_step)


def collate_fn(batch, tokenizer, max_length=512):
    """Memory-efficient collate function with reduced sequence length."""
    # Reduce max_length for memory efficiency
    input_ids = []
    attention_masks = []
    
    for item in batch:
        ids = item['input_ids']
        # Truncate to max_length
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        input_ids.append(ids)
        attention_masks.append([1] * len(ids))
    
    # Pad sequences
    max_len = min(max([len(ids) for ids in input_ids]), max_length)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        padding_length = max_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id] * padding_length
        padded_mask = mask + [0] * padding_length
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long)
    }


def create_dataloaders(dataset_path: str, tokenizer, batch_size: int = 1, max_length: int = 512):
    """Create memory-efficient dataloaders with very small batch size."""
    print(f"Loading dataset from {dataset_path}...")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print(f"Train dataset size: {len(train_dataset):,}")
    print(f"Validation dataset size: {len(test_dataset):,}")
    
    # Create dataloaders with very small batch size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
        num_workers=0,  # Disable multiprocessing to save memory
        pin_memory=False,  # Disable pin_memory to save memory
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    print("Dataloaders created successfully!")
    return train_dataloader, val_dataloader


def setup_wandb(config: Dict[str, Any]):
    """Setup Weights & Biases logging."""
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'move-500m-arxiv'),
            name=config.get('wandb_run_name', 'move-500m-training'),
            config=config
        )
        print("Weights & Biases initialized")
    else:
        print("Weights & Biases logging disabled")


def main():
    """Main training function with memory optimization."""
    parser = argparse.ArgumentParser(description='Train MoVE 500M model on ArXiv dataset with memory optimization')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the prepared ArXiv dataset')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='500m',
                       choices=['500m', '700m', '1b'],
                       help='Model size configuration')
    
    # Training arguments - MEMORY OPTIMIZED
    parser.add_argument('--batch_size', type=int, default=1,  # Reduced from 2
                       help='Training batch size (reduced for memory efficiency)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,  # Increased
                       help='Gradient accumulation steps (increased to maintain effective batch size)')
    parser.add_argument('--max_length', type=int, default=512,  # Reduced from 1024
                       help='Maximum sequence length (reduced for memory efficiency)')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine_with_warmup',
                       choices=['cosine_with_warmup', 'linear_with_warmup', 'onecycle'],
                       help='Learning rate scheduler')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='models/move_500m_arxiv',
                       help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=500,
                       help='Checkpoint saving interval')
    
    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='move-500m-arxiv',
                       help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='WandB run name')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Prepare config
    config = vars(args)
    
    # Check dataset exists
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    
    print("Creating optimized MoVE 500M model...")
    model = create_optimized_move_model(args.model_size)
    print("Model creation completed!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Memory estimation
    if torch.cuda.is_available():
        model_memory = total_params * 4 / (1024**3)  # 4 bytes per parameter
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_utilization = (model_memory / gpu_memory) * 100
        print(f"Model fits in RTX 4090: {'✓' if memory_utilization < 80 else '✗'} ({memory_utilization:.1f}% memory utilization)")
    
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        args.dataset_path, 
        tokenizer, 
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Setup WandB
    setup_wandb(config)
    
    # Create trainer
    trainer = MemoryOptimizedMoVE500MTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # Start training
    try:
        best_val_loss = trainer.train()
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if config.get('use_wandb', False):
            wandb.finish()


if __name__ == '__main__':
    main()