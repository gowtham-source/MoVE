#!/usr/bin/env python3
"""
Training script for FFNMoE module.

This script trains the FFNMoE module to reconstruct layer-0 hidden states
from TinyLlama, using both router loss and reconstruction loss.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from modules.ffn_moe import FFNMoE, AdaptiveFFNMoE, compute_reconstruction_loss

class Layer0Dataset(Dataset):
    """Dataset for loading layer-0 hidden states from TinyLlama."""
    
    def __init__(self, data_dir, num_samples=None):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        # Load vector files
        vec_files = sorted([f for f in self.data_dir.glob("*.pt") if f.name.startswith("l0_step")])
        
        loaded_samples = 0
        for vec_file in vec_files:
            if num_samples and loaded_samples >= num_samples:
                break
                
            try:
                data = torch.load(vec_file, map_location='cpu')
                if isinstance(data, dict) and 'hidden_states' in data and 'input_ids' in data:
                    # Extract samples from this batch
                    hidden_states = data['hidden_states']  # [batch_size, seq_len, hidden_dim]
                    input_ids = data['input_ids']  # [batch_size, seq_len]
                    attention_mask = data.get('attention_mask', None)
                    
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    
                    for i in range(batch_size):
                        if num_samples and loaded_samples >= num_samples:
                            break
                            
                        sample_hidden = hidden_states[i]  # [seq_len, hidden_dim]
                        sample_ids = input_ids[i]  # [seq_len]
                        
                        # Use attention mask if available
                        if attention_mask is not None:
                            sample_mask = attention_mask[i]  # [seq_len]
                            # Only keep valid tokens
                            valid_indices = sample_mask.bool()
                            sample_hidden = sample_hidden[valid_indices]
                            sample_ids = sample_ids[valid_indices]
                        
                        if sample_hidden.size(0) > 0:  # Ensure non-empty
                            self.samples.append({
                                'hidden_states': sample_hidden,
                                'input_ids': sample_ids
                            })
                            loaded_samples += 1
                            
            except Exception as e:
                print(f"Warning: Failed to load {vec_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples from {len(vec_files)} files")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'hidden_states': sample['hidden_states'],
            'input_ids': sample['input_ids']
        }

def collate_fn(batch):
    """Collate function for variable sequence lengths."""
    # Find max sequence length in batch
    max_len = max(item['hidden_states'].size(0) for item in batch)
    hidden_dim = batch[0]['hidden_states'].size(1)
    
    # Pad sequences
    padded_hidden = torch.zeros(len(batch), max_len, hidden_dim)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item['hidden_states'].size(0)
        padded_hidden[i, :seq_len] = item['hidden_states']
        padded_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = True
    
    return {
        'hidden_states': padded_hidden,
        'input_ids': padded_ids,
        'attention_mask': attention_mask
    }

def train_ffn_moe(model_type='basic', batch_size=4, num_epochs=5, num_samples=500, 
                  num_experts=8, topk=2, expert_dim=None, lr=1e-4, device='auto'):
    """Train FFNMoE module.
    
    Args:
        model_type (str): Type of model ('basic' or 'adaptive')
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        num_samples (int): Number of samples to use
        num_experts (int): Number of experts in MoE
        topk (int): Number of top experts to use
        expert_dim (int): Expert hidden dimension
        lr (float): Learning rate
        device (str): Device to use ('auto', 'cuda', 'cpu')
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    data_dir = project_root / "data" / "vecs"
    dataset = Layer0Dataset(data_dir, num_samples=num_samples)
    
    if len(dataset) == 0:
        raise ValueError("No samples loaded from dataset")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    hidden_dim = 2048  # TinyLlama hidden dimension
    
    if model_type == 'basic':
        model = FFNMoE(
            d=hidden_dim,
            experts=num_experts,
            topk=topk,
            expert_dim=expert_dim
        )
    elif model_type == 'adaptive':
        model = AdaptiveFFNMoE(
            d=hidden_dim,
            experts=num_experts,
            topk=topk,
            expert_dim=expert_dim,
            adaptive_topk=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training loop
    model.train()
    training_log = []
    
    print(f"Training {model_type} FFNMoE for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_router_losses = []
        epoch_recon_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            hidden_states = batch['hidden_states'].to(device)  # [batch_size, seq_len, hidden_dim]
            attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_len]
            
            optimizer.zero_grad()
            
            # Forward pass
            output, router_loss = model(hidden_states, return_router_loss=True)
            
            # Compute reconstruction loss (only on valid tokens)
            recon_loss = compute_reconstruction_loss(hidden_states, output, reduction='none')
            
            # Apply attention mask
            masked_recon_loss = recon_loss * attention_mask.unsqueeze(-1)
            recon_loss = masked_recon_loss.sum() / attention_mask.sum()
            
            # Total loss
            total_loss = recon_loss + router_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Log losses
            epoch_losses.append(total_loss.item())
            epoch_router_losses.append(router_loss.item())
            epoch_recon_losses.append(recon_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.6f}",
                'recon': f"{recon_loss.item():.6f}",
                'router': f"{router_loss.item():.6f}"
            })
        
        # Log epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_router_loss = sum(epoch_router_losses) / len(epoch_router_losses)
        avg_recon_loss = sum(epoch_recon_losses) / len(epoch_recon_losses)
        
        training_log.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_router_loss': avg_router_loss,
            'avg_recon_loss': avg_recon_loss
        })
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, Recon={avg_recon_loss:.6f}, Router={avg_router_loss:.6f}")
    
    # Save model and log
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"ffn_moe_{model_type}.pt"
    log_path = logs_dir / f"ffn_moe_{model_type}.json"
    
    torch.save(model.state_dict(), model_path)
    
    with open(log_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'num_experts': num_experts,
            'topk': topk,
            'expert_dim': expert_dim,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'num_samples': len(dataset),
            'final_avg_loss': training_log[-1]['avg_loss'],
            'training_log': training_log
        }, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Training log saved to: {log_path}")
    print(f"Final average loss: {training_log[-1]['avg_loss']:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FFNMoE module")
    parser.add_argument("--model_type", type=str, default="basic", choices=["basic", "adaptive"],
                        help="Type of FFNMoE model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--topk", type=int, default=2, help="Number of top experts")
    parser.add_argument("--expert_dim", type=int, default=None, help="Expert hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    train_ffn_moe(
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        num_experts=args.num_experts,
        topk=args.topk,
        expert_dim=args.expert_dim,
        lr=args.lr,
        device=args.device
    )