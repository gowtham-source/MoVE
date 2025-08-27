#!/usr/bin/env python3
"""
Training script for PosGen module to match RoPE output from TinyLlama.

This script trains the PosGen module to generate positional encodings that
match the RoPE (Rotary Position Embedding) output from TinyLlama.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import math

# Add parent directory to path for imports
sys.path.append('.')
from modules.pos_gen import PosGen, RoPEPosGen, LearnablePosGen

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoPETargetDataset(Dataset):
    """Dataset for generating RoPE targets from TinyLlama."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_seq_len=512, num_samples=1000):
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        
        # Load TinyLlama model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        
        # Extract RoPE parameters
        self.rope_theta = getattr(self.model.config, 'rope_theta', 10000.0)
        self.hidden_size = self.model.config.hidden_size
        self.num_attention_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        logger.info(f"Model config: hidden_size={self.hidden_size}, heads={self.num_attention_heads}, head_dim={self.head_dim}")
        
        # Pre-generate position indices and RoPE targets
        self.position_indices = []
        self.rope_targets = []
        
        self._generate_targets()
        
    def _generate_targets(self):
        """Generate RoPE targets for different sequence lengths."""
        logger.info("Generating RoPE targets...")
        
        with torch.no_grad():
            for i in tqdm(range(self.num_samples), desc="Generating targets"):
                # Random sequence length
                seq_len = torch.randint(10, self.max_seq_len + 1, (1,)).item()
                
                # Create position indices
                pos_ids = torch.arange(seq_len, dtype=torch.long)
                
                # Generate RoPE embeddings
                rope_emb = self._compute_rope_embeddings(pos_ids)
                
                self.position_indices.append(pos_ids)
                self.rope_targets.append(rope_emb)
        
        logger.info(f"Generated {len(self.position_indices)} samples")
    
    def _compute_rope_embeddings(self, position_ids):
        """Compute RoPE embeddings for given position indices."""
        seq_len = position_ids.size(0)
        
        # Compute frequency components
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        
        # Compute position encodings
        freqs = torch.outer(position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Create cos and sin embeddings
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Combine cos and sin (interleaved pattern)
        rope_emb = torch.stack([cos_emb, sin_emb], dim=-1).flatten(-2)
        
        # Expand to full hidden dimension (repeat for all heads)
        rope_emb = rope_emb.repeat(1, self.num_attention_heads)
        
        # Ensure correct dimension
        if rope_emb.size(-1) != self.hidden_size:
            rope_emb = rope_emb[:, :self.hidden_size]
        
        return rope_emb
    
    def __len__(self):
        return len(self.position_indices)
    
    def __getitem__(self, idx):
        return {
            'position_ids': self.position_indices[idx],
            'rope_target': self.rope_targets[idx]
        }

def collate_fn(batch):
    """Collate function to handle variable sequence lengths."""
    # Find max sequence length in batch
    max_len = max(item['position_ids'].size(0) for item in batch)
    
    batch_pos = []
    batch_targets = []
    
    for item in batch:
        pos_ids = item['position_ids']
        rope_target = item['rope_target']
        seq_len = pos_ids.size(0)
        
        # Pad position indices
        if seq_len < max_len:
            pad_size = max_len - seq_len
            pos_ids = torch.cat([pos_ids, torch.zeros(pad_size, dtype=pos_ids.dtype)])
            
            # Pad targets with zeros
            rope_target = torch.cat([rope_target, torch.zeros(pad_size, rope_target.size(-1))])
        
        batch_pos.append(pos_ids)
        batch_targets.append(rope_target)
    
    return {
        'position_ids': torch.stack(batch_pos),
        'rope_target': torch.stack(batch_targets).float()
    }

def train_pos_gen(
    model_type='mlp',
    hidden_dim=2048,
    max_seq_len=512,
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=10,
    num_samples=1000,
    output_dir='models',
    log_dir='logs',
    device=None
):
    """Train PosGen module to match RoPE output."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training on device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = RoPETargetDataset(max_seq_len=max_seq_len, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create model
    logger.info(f"Creating {model_type} PosGen model...")
    if model_type == 'rope':
        model = RoPEPosGen(d=hidden_dim, max_len=max_seq_len)
    elif model_type == 'learnable':
        model = LearnablePosGen(d=hidden_dim, max_len=max_seq_len)
    else:  # 'mlp'
        model = PosGen(d=hidden_dim, max_len=max_seq_len)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    training_log = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            position_ids = batch['position_ids'].to(device)
            rope_target = batch['rope_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Generate positional encodings
            pos_encodings = model(position_ids)
            
            # Compute loss
            loss = criterion(pos_encodings, rope_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Log epoch results
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss
        })
    
    # Save model
    model_path = os.path.join(output_dir, f'pos_gen_{model_type}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'hidden_dim': hidden_dim,
        'max_seq_len': max_seq_len,
        'final_loss': avg_loss
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training log
    log_path = os.path.join(log_dir, f'pos_gen_{model_type}.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log saved to {log_path}")
    
    logger.info(f"Training completed! Final loss: {avg_loss:.6f}")
    
    return model, training_log

def main():
    parser = argparse.ArgumentParser(description='Train PosGen module')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'rope', 'learnable'],
                        help='Type of PosGen model to train')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Train model
    train_pos_gen(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )

if __name__ == '__main__':
    main()