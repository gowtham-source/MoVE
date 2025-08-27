#!/usr/bin/env python3
"""
Minimal Transformer Training Script

A basic transformer model to test training stability and isolate NaN issues.
Uses only PyTorch built-in components with conservative settings.
"""

import os
import sys
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

# Set conservative memory settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class MinimalTransformer(nn.Module):
    """Minimal transformer model for testing."""
    
    def __init__(self, vocab_size=32000, d_model=512, nhead=8, num_layers=6, max_seq_len=1024):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Smaller FFN
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights conservatively
        self.apply(self._init_weights)
        
        # Scale down embeddings
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.1)
            self.position_embedding.weight.mul_(0.1)
    
    def _init_weights(self, module):
        """Conservative weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
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
        
        # Embeddings
        positions = torch.arange(seq_len, device=device)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings with very small scaling
        x = token_emb + pos_emb.unsqueeze(0)
        x = x * 0.1  # Very conservative scaling
        
        # Apply transformer
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        return logits

class SimpleDataset(Dataset):
    """Simple dataset wrapper."""
    
    def __init__(self, data_path, tokenizer, max_length=1024, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        dataset = load_from_disk(data_path)
        self.data = dataset[split]
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', item.get('content', ''))
        
        # Tokenize with conservative settings
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def train_step(model, batch, optimizer, device):
    """Single training step with extensive safety checks."""
    model.train()
    
    # Move to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    
    # Check for NaN/Inf in logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("NaN/Inf detected in logits, skipping batch")
        return 0.0
    
    # Prepare labels (shift for causal LM)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    # Calculate loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Use very conservative loss calculation
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Check loss
    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
        print(f"Invalid loss detected: {loss.item()}, skipping batch")
        return 0.0
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if torch.isnan(torch.tensor(total_norm)) or total_norm > 10.0:
        print(f"Large gradient norm: {total_norm}, skipping step")
        optimizer.zero_grad()
        return 0.0
    
    # Clip gradients very conservatively
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=512)  # Shorter sequences
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-6)  # Very small LR
    parser.add_argument('--output_dir', type=str, default='models/minimal_transformer')
    
    args = parser.parse_args()
    
    print("Starting minimal transformer training...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = MinimalTransformer(
        vocab_size=len(tokenizer),
        d_model=256,  # Very small model
        nhead=4,
        num_layers=4,
        max_seq_len=args.max_length
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create dataset
    dataset = SimpleDataset(args.dataset_path, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Optimizer with very conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            loss = train_step(model, batch, optimizer, device)
            
            if loss > 0.0:
                total_loss += loss
                valid_batches += 1
            
            # Update progress
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'valid': f'{valid_batches}/{step+1}'
                })
            
            # Save checkpoint
            if step > 0 and step % 1000 == 0:
                checkpoint_path = output_dir / f'checkpoint-{step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'loss': avg_loss if valid_batches > 0 else 0.0
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        # Epoch summary
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch + 1} completed - Average loss: {avg_loss:.4f} ({valid_batches} valid batches)")
        else:
            print(f"Epoch {epoch + 1} completed - No valid batches")
    
    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    print("Training completed!")

if __name__ == '__main__':
    main()