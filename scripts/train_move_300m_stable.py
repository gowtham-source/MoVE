#!/usr/bin/env python3
"""
MoVE 300M Model Training Script - Numerically Stable Version

This script trains a smaller 300M parameter MoVE model with enhanced numerical stability.
Designed to prevent NaN losses and ensure stable training on RTX 4090.
"""

import os
import sys
import json
import argparse
import logging
import gc
import math
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

class StableMoVE300M(nn.Module):
    """Numerically stable 300M MoVE model."""
    
    def __init__(self, vocab_size=32000, d_model=768, num_layers=12, num_heads=12,
                 max_seq_len=1024, moe_experts=2, moe_topk=1, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding with proper scaling
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Scale embeddings
        with torch.no_grad():
            self.token_embed.weight.mul_(0.5)
            self.pos_embed.weight.mul_(0.5)
    
    def _init_weights(self, module):
        """Initialize weights with conservative scaling to prevent NaN."""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization with smaller scale
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Token embeddings
        token_emb = self.token_embed(input_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embed(positions)
        
        # Combine embeddings with scaling
        x = (token_emb + pos_emb) * math.sqrt(self.d_model)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask, tgt_is_causal=True)
        
        # Final layer norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Apply temperature scaling to prevent extreme values
        logits = logits / 1.0  # Temperature of 1.0
        
        return {'logits': logits}

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

class StableTrainer:
    """Numerically stable trainer."""
    
    def __init__(self, model, train_dataloader, learning_rate=1e-5, device='cuda',
                 output_dir='models/move_300m_stable', gradient_accumulation_steps=8):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conservative optimizer settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Mixed precision with conservative settings
        self.scaler = GradScaler(init_scale=2**10, growth_factor=1.1, backoff_factor=0.9)
        
        self.global_step = 0
        
        # Setup memory optimization
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
        gc.collect()
    
    def train_step(self, batch):
        """Single training step with stability checks."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with autocast():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            logits = outputs['logits']
            
            # Check for NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN/Inf detected in logits, skipping batch")
                return 0.0
            
            # Calculate loss with label smoothing
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            
            # Use label smoothing to improve stability
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss detected, skipping batch")
                return 0.0
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        valid_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            
            # Only accumulate if loss is valid (not 0.0 from skipped batch)
            if loss > 0.0:
                total_loss += loss
                valid_batches += 1
            
            num_batches += 1
            
            # Gradient accumulation - only proceed if we have valid gradients
            if (step + 1) % self.gradient_accumulation_steps == 0 and valid_batches > 0:
                try:
                    # Conservative gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    
                    # Check gradient norm
                    if torch.isnan(grad_norm) or grad_norm > 10.0:
                        print(f"Warning: Large gradient norm {grad_norm}, skipping step")
                        self.optimizer.zero_grad()
                        continue
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
                    
                except RuntimeError as e:
                    print(f"Warning: Gradient step failed: {e}, skipping")
                
                self.optimizer.zero_grad()
                
                # Memory cleanup
                if self.global_step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Update progress
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'step': self.global_step,
                    'valid': f'{valid_batches}/{num_batches}'
                })
            
            # Save checkpoint
            if self.global_step % 500 == 0 and self.global_step > 0:
                self.save_checkpoint(f'checkpoint-{self.global_step}')
        
        return {'loss': total_loss / max(valid_batches, 1)}
    
    def save_checkpoint(self, checkpoint_name):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }, checkpoint_dir / 'pytorch_model.bin')
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def train(self, num_epochs):
        """Main training loop."""
        print(f"Starting stable training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1} - Loss: {train_metrics['loss']:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch-{epoch + 1}')
        
        self.save_checkpoint('final')
        print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train stable MoVE 300M model')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='models/move_300m_stable')
    
    args = parser.parse_args()
    
    print("Initializing stable MoVE 300M training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = StableMoVE300M(vocab_size=len(tokenizer))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create dataset
    train_dataset = ArxivDataset(
        args.dataset_path, tokenizer, args.max_length, 'train'
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Create trainer
    trainer = StableTrainer(
        model=model,
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        device=device,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Start training
    trainer.train(args.num_epochs)

if __name__ == '__main__':
    main()