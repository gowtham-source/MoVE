#!/usr/bin/env python3
"""
MoVE 300M Model Training Script for ArXiv Dataset

This script trains a smaller 300M parameter MoVE model optimized for RTX 4090.
Designed to fit comfortably within 6GB VRAM with aggressive memory optimization.
"""

import os
import sys
import json
import argparse
import logging
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from move_large import MoVELarge

class MoVE300MConfig:
    """Ultra-optimized 300M MoVE configuration for RTX 4090."""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get 300M model config optimized for 6GB VRAM."""
        return {
            'vocab_size': 32000,
            'hidden_size': 768,        # Smaller hidden size
            'intermediate_size': 2048, # Smaller FFN
            'num_hidden_layers': 12,   # Fewer layers
            'num_attention_heads': 12, # Fewer heads
            'num_key_value_heads': 4,
            'max_position_embeddings': 1024,  # Shorter sequences
            'rms_norm_eps': 1e-5,
            'use_cache': True,
            'tie_word_embeddings': False,
            'rope_theta': 10000.0,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.0,
            
            # MoVE specific - minimal experts
            'num_experts': 2,          # Minimal experts
            'num_experts_per_tok': 1,  # Single expert per token
            'router_aux_loss_coef': 0.01,
            
            # Memory optimization
            'gradient_checkpointing': True,
            'use_flash_attention': True,
            'memory_efficient_attention': True
        }

class ArxivDataset(Dataset):
    """ArXiv dataset for training."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, split: str = 'train'):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data using datasets library
        try:
            dataset = load_from_disk(str(self.data_path))
            self.data = dataset[split]
            print(f"Loaded {split} split with {len(self.data)} examples")
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.data_path}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', item.get('content', ''))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class MoVE300MTrainer:
    """Trainer for MoVE 300M model with aggressive memory optimization."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        device: str = 'cuda',
        output_dir: str = 'models/move_300m',
        log_interval: int = 10,
        save_interval: int = 500,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_wandb = use_wandb
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup memory optimization
        self._setup_memory_optimization()
        
        # Initialize tracking
        self.global_step = 0
        self.epoch = 0
        
        # Setup logging
        self._setup_logging()
        
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="move-300m-training",
                    config={
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'model_params': sum(p.numel() for p in model.parameters()),
                    }
                )
            except ImportError:
                print("Weights & Biases not available, skipping...")
                self.use_wandb = False
    
    def _setup_memory_optimization(self):
        """Setup aggressive memory optimization."""
        print("Setting up memory optimization...")
        
        # Set GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("✓ GPU memory fraction set to 95%")
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Memory cache cleared")
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        if self.use_amp:
            with autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                # Calculate loss manually for causal LM
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            # Calculate loss manually for causal LM
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=True
        )
        
        for step, batch in enumerate(progress_bar):
            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Memory cleanup
                if self.global_step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': self.global_step
                })
                
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/global_step': self.global_step
                        })
                    except:
                        pass
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint-{self.global_step}')
        
        return {'loss': total_loss / num_batches}
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch
        }, checkpoint_dir / 'pytorch_model.bin')
        
        # Save config
        config = MoVE300MConfig.get_config()
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Total training steps: {len(self.train_dataloader) * num_epochs}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Loss: {train_metrics['loss']:.4f}"
            )
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch-{epoch + 1}')
        
        # Save final model
        self.save_checkpoint('final')
        print("Training completed!")

def create_move_300m_model(vocab_size: int = 32000) -> nn.Module:
    """Create MoVE 300M model."""
    config = MoVE300MConfig.get_config()
    config['vocab_size'] = vocab_size
    
    model = MoVELarge(
        vocab_size=config['vocab_size'],
        d_model=config['hidden_size'],
        num_layers=config['num_hidden_layers'],
        moe_experts=config['num_experts'],
        moe_topk=config['num_experts_per_tok'],
        dropout=config['hidden_dropout'],
        tie_weights=config['tie_word_embeddings'],
        use_lora=False
    )
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description='Train MoVE 300M model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--val_dataset_path', type=str, help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='models/move_300m_arxiv', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Save interval')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
        
        # Create model
        print("Creating MoVE 300M model...")
        model, config = create_move_300m_model(vocab_size=len(tokenizer))
        print("Model creation completed!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {total_params:,} parameters")
        
        # Memory check
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU memory: {gpu_memory:.1f} GB")
        
        # Create datasets
        print("Creating dataloaders...")
        train_dataset = ArxivDataset(args.dataset_path, tokenizer, args.max_length, split='train')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if device == 'cuda' else False
        )
        
        val_dataloader = None
        if args.val_dataset_path:
            val_dataset = ArxivDataset(args.val_dataset_path, tokenizer, args.max_length, split='test')
        else:
            # Use test split from same dataset if no separate validation path provided
            try:
                val_dataset = ArxivDataset(args.dataset_path, tokenizer, args.max_length, split='test')
            except:
                val_dataset = None
        
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device == 'cuda' else False
            )
        
        print(f"Train dataset size: {len(train_dataset):,}")
        if val_dataloader:
            print(f"Validation dataset size: {len(val_dataset):,}")
        print("Dataloaders created successfully!")
        
        # Create trainer
        trainer = MoVE300MTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            use_wandb=args.use_wandb
        )
        
        print(f"MoVE 300M Trainer initialized")
        print(f"Device: {device}")
        print(f"Mixed precision: True")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Model parameters: {total_params:,}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU memory allocated: {allocated:.2f} GB")
        
        # Start training
        trainer.train(args.num_epochs)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())