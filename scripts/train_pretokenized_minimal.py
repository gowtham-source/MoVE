#!/usr/bin/env python3
"""
Minimal Transformer Training Script for Pre-tokenized Dataset

This script works with datasets that already contain input_ids and attention_mask.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from datasets import load_from_disk
from tqdm import tqdm
import gc
import math

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.9)

class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized data with input_ids and attention_mask."""
    
    def __init__(self, dataset, max_length=512):
        self.dataset = dataset
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get pre-tokenized data
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        # Create labels for causal LM (shift input_ids)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class SimpleTransformer(nn.Module):
    """Ultra-simple transformer for debugging."""
    
    def __init__(self, vocab_size=50257, d_model=256, nhead=8, num_layers=2, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings with conservative initialization
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Simple transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights conservatively
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Check for invalid input
        if torch.isnan(input_ids.float()).any() or torch.isinf(input_ids.float()).any():
            print(f"Warning: Invalid input_ids detected")
            return {'loss': torch.tensor(float('nan'))}
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Scale embeddings
        x = (token_emb + pos_emb) * math.sqrt(self.d_model) * 0.1  # Extra scaling factor
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True = attend, False = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Apply temperature scaling to prevent extreme values
        logits = logits / 2.0
        
        # Check for invalid logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: Invalid logits detected")
            return {'loss': torch.tensor(float('nan'))}
        
        outputs = {'logits': logits}
        
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with label smoothing
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )
            
            outputs['loss'] = loss
        
        return outputs

def train_step(model, batch, optimizer, scaler, device):
    """Single training step with extensive safety checks."""
    model.train()
    
    # Move to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Safety checks
    if torch.isnan(input_ids.float()).any():
        print("NaN in input_ids, skipping batch")
        return None
    
    if torch.isnan(attention_mask.float()).any():
        print("NaN in attention_mask, skipping batch")
        return None
    
    if torch.isnan(labels.float()).any():
        print("NaN in labels, skipping batch")
        return None
    
    # Forward pass with autocast
    with autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
    
    # Check loss validity
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Invalid loss detected: {loss.item()}, skipping batch")
        return None
    
    if loss.item() > 100.0:  # Extremely high loss
        print(f"Extremely high loss detected: {loss.item()}, skipping batch")
        return None
    
    # Backward pass
    scaler.scale(loss).backward()
    
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to pre-tokenized dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_from_disk(args.dataset_path)
        train_dataset = PreTokenizedDataset(dataset['train'], max_length=args.max_length)
        print(f"Loaded {len(train_dataset)} training samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("Creating model...")
    model = SimpleTransformer(
        vocab_size=50257,  # GPT-2 vocab size
        d_model=256,
        nhead=8,
        num_layers=2,
        max_seq_len=args.max_length
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            loss = train_step(model, batch, optimizer, scaler, device)
            
            if loss is not None:
                total_loss += loss
                valid_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'valid': valid_batches})
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Logging
                if step % args.log_interval == 0 and valid_batches > 0:
                    print(f"Step {step}, Avg Loss: {avg_loss:.4f}, Valid Batches: {valid_batches}")
            
            # Memory cleanup
            if step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"Epoch {epoch + 1} completed. Valid batches: {valid_batches}/{len(train_loader)}")
        if valid_batches > 0:
            print(f"Average loss: {total_loss / valid_batches:.4f}")
        else:
            print("No valid batches processed!")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, model_path)
    
    print(f"Training completed! Model saved to {model_path}")

if __name__ == '__main__':
    main()