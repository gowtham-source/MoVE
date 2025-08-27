#!/usr/bin/env python3
"""
Training Script for Token Embedding Module

Trains the TokenEmbed module using MSE loss against extracted Layer-0 vectors.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import json
from pathlib import Path

# Add modules to path
import sys
sys.path.append('modules')
from token_embed import TokenEmbed, TokenEmbedWithLoRA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDataset(Dataset):
    """Dataset for loading extracted vectors and corresponding token IDs."""
    
    def __init__(self, vectors_dir="data/vecs", max_files=None):
        self.vectors_dir = Path(vectors_dir)
        self.vector_files = sorted(list(self.vectors_dir.glob("l0_step*.pt")))
        
        if max_files:
            self.vector_files = self.vector_files[:max_files]
            
        logger.info(f"Found {len(self.vector_files)} vector files")
        
        # Load metadata
        metadata_path = self.vectors_dir / "metadata.pt"
        if metadata_path.exists():
            self.metadata = torch.load(metadata_path)
            logger.info(f"Loaded metadata: {self.metadata}")
        else:
            logger.warning("No metadata found")
            self.metadata = {}
    
    def __len__(self):
        return len(self.vector_files)
    
    def __getitem__(self, idx):
        # Load vector file
        data = torch.load(self.vector_files[idx])
        
        input_ids = data['input_ids']  # [batch_size, seq_len]
        hidden_states = data['hidden_states']  # [batch_size, seq_len, hidden_dim]
        
        return {
            'input_ids': input_ids,
            'target_vectors': hidden_states,
            'attention_mask': data.get('attention_mask', None)
        }

def collate_fn(batch):
    """Custom collate function to handle batched data."""
    # Flatten batch dimension since each item already contains a batch
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    target_vectors = torch.cat([item['target_vectors'] for item in batch], dim=0).float()  # Convert to float32
    
    attention_masks = [item['attention_mask'] for item in batch if item['attention_mask'] is not None]
    if attention_masks:
        attention_mask = torch.cat(attention_masks, dim=0)
    else:
        attention_mask = None
    
    return {
        'input_ids': input_ids,
        'target_vectors': target_vectors,
        'attention_mask': attention_mask
    }

def train_token_embed(use_lora=True, epochs=10, batch_size=16, lr=5e-4, device=None):
    """Train the token embedding module.
    
    Args:
        use_lora (bool): Whether to use LoRA adaptation
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        device (str): Device to train on
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")
    
    # Create dataset and dataloader
    dataset = VectorDataset("data/vecs")
    if len(dataset) == 0:
        logger.error("No vector files found. Run extract_vectors.py first.")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Each file contains a batch already
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Create model
    if use_lora:
        model = TokenEmbedWithLoRA(vocab=32000, d=2048, lora_rank=16)
        logger.info("Using LoRA adaptation")
    else:
        model = TokenEmbed(vocab=32000, d=2048)
        logger.info("Using standard embedding")
    
    model = model.to(device)
    
    # Setup optimizer and loss
    if use_lora:
        # Only optimize LoRA parameters
        optimizer = optim.AdamW([
            {'params': [model.lora_A, model.lora_B]}
        ], lr=lr, weight_decay=0.01)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    total_loss = 0.0
    step = 0
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_vectors = batch['target_vectors'].to(device)
            
            # Forward pass
            embeddings = model(input_ids)
            
            # Compute loss
            loss = criterion(embeddings, target_vectors)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            epoch_loss += loss.item()
            step += 1
            epoch_steps += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'avg_loss': f"{epoch_loss/epoch_steps:.6f}"
            })
            
            # Log every 10 steps
            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / epoch_steps
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/token_embed.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': 32000,
            'embed_dim': 2048,
            'use_lora': use_lora,
            'lora_rank': 16 if use_lora else None
        },
        'training_config': {
            'epochs': epochs,
            'lr': lr,
            'final_loss': total_loss / step
        }
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Training completed. Final average loss: {total_loss/step:.6f}")

def evaluate_token_embed(model_path="models/token_embed.pt", device=None):
    """Evaluate the trained token embedding model."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    if config['use_lora']:
        model = TokenEmbedWithLoRA(
            vocab=config['vocab_size'],
            d=config['embed_dim'],
            lora_rank=config['lora_rank']
        )
    else:
        model = TokenEmbed(
            vocab=config['vocab_size'],
            d=config['embed_dim']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    dataset = VectorDataset("data/vecs", max_files=10)  # Use first 10 files for eval
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    
    logger.info("Evaluating model...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            target_vectors = batch['target_vectors'].to(device)
            
            embeddings = model(input_ids)
            loss = criterion(embeddings, target_vectors)
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    logger.info(f"Evaluation completed. Average MSE loss: {avg_loss:.6f}")
    
    return avg_loss

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Token Embedding Module")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA adaptation")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained model")
    parser.add_argument("--model_path", type=str, default="models/token_embed.pt", help="Model path for evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_token_embed(args.model_path)
    else:
        train_token_embed(
            use_lora=not args.no_lora,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )