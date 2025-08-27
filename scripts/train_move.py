#!/usr/bin/env python3
"""
Mini-Training Loop for MoVE Model

This script implements a lightweight training loop for the integrated MoVE model
with LoRA adaptation, perplexity logging, and efficient memory usage.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import datasets
import os
import sys
import csv
import time
import math
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from move.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location("move_module", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "move.py"))
move_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(move_module)
MoVE = move_module.MoVE
create_move_model = move_module.create_move_model

class TextDataset(Dataset):
    """Text dataset for training using real tokenized data."""
    
    def __init__(self, data_path, max_length=1024, split='train'):
        self.max_length = max_length
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_data.py and tokenise.py first.")
        
        print(f"Loading dataset from {data_path}...")
        self.dataset = datasets.load_from_disk(data_path)
        
        # Create train/validation split if needed
        if hasattr(self.dataset, 'train_test_split'):
            splits = self.dataset.train_test_split(test_size=0.1, seed=42)
            if split == 'train':
                self.dataset = splits['train']
            else:
                self.dataset = splits['test']
        
        print(f"Using {len(self.dataset)} examples for {split}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Ensure proper length
        input_ids = item['input_ids'][:self.max_length]
        attention_mask = item.get('attention_mask', [1] * len(input_ids))[:self.max_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([0] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def compute_perplexity(model, dataloader, device, max_batches=50):
    """Compute perplexity on a dataset.
    
    Args:
        model: MoVE model
        dataloader: DataLoader for evaluation
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        float: Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift for language modeling
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = attention_mask[:, 1:].contiguous()
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs[:, :, :]
            
            # Compute loss
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            losses = losses.view(targets.shape)
            
            # Apply mask
            masked_losses = losses * mask.float()
            total_loss += masked_losses.sum().item()
            total_tokens += mask.sum().item()
    
    model.train()
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def setup_lora_optimizer(model, lr=5e-5, weight_decay=0.01):
    """Setup optimizer for LoRA parameters.
    
    Args:
        model: MoVE model
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    # Collect LoRA parameters
    lora_params = []
    regular_params = []
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            lora_params.append(param)
        else:
            regular_params.append(param)
    
    print(f"LoRA parameters: {len(lora_params)}")
    print(f"Regular parameters: {len(regular_params)}")
    
    # If no LoRA params found, train all parameters
    if not lora_params:
        print("No LoRA parameters found, training all parameters...")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Different learning rates for LoRA and regular params
    param_groups = [
        {'params': lora_params, 'lr': lr, 'weight_decay': 0.0},  # No weight decay for LoRA
        {'params': regular_params, 'lr': lr * 0.1, 'weight_decay': weight_decay}  # Lower LR for regular
    ]
    
    return optim.AdamW(param_groups)

def train_move_model(
    model_config='small',
    data_path='../data/owt_1pct_tok',
    output_dir='../models',
    log_dir='../logs',
    batch_size=4,
    seq_length=1024,
    learning_rate=5e-5,
    num_epochs=1,
    log_interval=100,
    save_interval=500,
    eval_interval=200,
    max_steps=None,
    resume_from=None
):
    """Train MoVE model with mini-training loop.
    
    Args:
        model_config: Model configuration name
        data_path: Path to tokenized dataset
        output_dir: Directory to save models
        log_dir: Directory to save logs
        batch_size: Training batch size
        seq_length: Sequence length
        learning_rate: Learning rate
        num_epochs: Number of epochs
        log_interval: Steps between logging
        save_interval: Steps between saving
        eval_interval: Steps between evaluation
        max_steps: Maximum training steps
        resume_from: Path to checkpoint to resume from
    """
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating MoVE model with config: {model_config}")
    model = create_move_model(model_config)
    model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Setup dataset
    print("Loading training dataset...")
    train_dataset = TextDataset(data_path, max_length=seq_length, split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Setup evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataset = TextDataset(data_path, max_length=seq_length, split='validation')
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = setup_lora_optimizer(model, lr=learning_rate)
    
    # Setup loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Setup logging
    log_file = os.path.join(log_dir, 'move.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'epoch', 'loss', 'perplexity', 'lr', 'time'])
    
    # Training loop
    model.train()
    global_step = 0
    start_time = time.time()
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_steps and global_step >= max_steps:
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Prepare inputs and targets for language modeling
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            # Forward pass
            try:
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs)
                    router_loss = outputs.get('router_loss', 0.0)
                else:
                    logits = outputs
                    router_loss = 0.0
                
                # Compute language modeling loss
                lm_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # Add router loss if available
                total_loss = lm_loss
                if isinstance(router_loss, torch.Tensor) and router_loss.numel() > 0:
                    total_loss = lm_loss + 0.01 * router_loss
                    
            except Exception as e:
                print(f"Error in forward pass at step {global_step}: {e}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'step': global_step
            })
            
            # Logging
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                perplexity = math.exp(avg_loss)
                current_lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                
                print(f"Step {global_step}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}, LR={current_lr:.2e}")
                
                # Log to CSV
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch, avg_loss, perplexity, current_lr, elapsed_time])
            
            # Evaluation
            if global_step % eval_interval == 0:
                print("Running evaluation...")
                eval_ppl = compute_perplexity(model, eval_dataloader, device)
                print(f"Evaluation perplexity: {eval_ppl:.2f}")
                
                # Log evaluation
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f'{global_step}_eval', epoch, 0, eval_ppl, current_lr, time.time() - start_time])
            
            # Save checkpoint
            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'move_checkpoint_step_{global_step}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'config': model.get_config()
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        if max_steps and global_step >= max_steps:
            break
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'move.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'training_info': {
            'final_step': global_step,
            'final_epoch': epoch,
            'total_time': time.time() - start_time
        }
    }, final_model_path)
    
    print(f"Training completed! Final model saved: {final_model_path}")
    print(f"Training log saved: {log_file}")
    
    return model, final_model_path

def main():
    parser = argparse.ArgumentParser(description='Train MoVE model')
    parser.add_argument('--config', type=str, default='small', choices=['small', 'medium', 'large'], help='Model configuration')
    parser.add_argument('--data_path', type=str, default='data/owt_1pct_tok', help='Path to tokenized dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum training steps')
    parser.add_argument('--log_interval', type=int, default=25, help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Save interval')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--log_file', type=str, default='logs/move.csv', help='Log file path')
    
    args = parser.parse_args()
    
    # Train the model
    train_move_model(
        model_config=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        log_dir='logs',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    import argparse
    main()