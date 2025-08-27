"""Large-Scale Training Script for MoVE Models

Optimized for RTX 4090 with 16GB VRAM:
- Mixed precision training (FP16)
- Gradient accumulation
- Memory optimization
- Advanced learning rate scheduling
- Comprehensive logging and checkpointing
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import datasets
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from move_large import create_move_large_model, estimate_memory_usage

class LargeDataset:
    """Optimized dataset for large-scale training."""
    
    def __init__(self, data_path, max_length=2048, split='train'):
        self.max_length = max_length
        
        print(f"Loading dataset from {data_path}...")
        self.dataset = datasets.load_from_disk(data_path)
        
        if hasattr(self.dataset, 'train_test_split'):
            splits = self.dataset.train_test_split(test_size=0.05, seed=42)
            if split == 'validation':
                self.dataset = splits['test']
            else:
                self.dataset = splits['train']
        
        print(f"Using {len(self.dataset)} examples for {split}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get input_ids and ensure proper length
        if isinstance(item, dict) and 'input_ids' in item:
            input_ids = item['input_ids']
        else:
            input_ids = item
        
        # Truncate or pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            # Pad with zeros (assuming 0 is pad token)
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long)  # For language modeling
        }

def setup_logging(log_dir, run_name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{run_name}.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_optimizer(model, learning_rate=1e-4, weight_decay=0.01, beta1=0.9, beta2=0.95):
    """Create AdamW optimizer with proper weight decay."""
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to biases, layer norms, and embeddings
            if any(nd in name for nd in ['bias', 'norm', 'embed']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8
    )
    
    return optimizer

def create_scheduler(optimizer, num_training_steps, warmup_steps=None):
    """Create learning rate scheduler with warmup and cosine decay."""
    if warmup_steps is None:
        warmup_steps = min(2000, num_training_steps // 10)
    
    # Linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=1e-6
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler

def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, loss, save_path, config):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['step'], checkpoint['epoch'], checkpoint['loss']

def train_large_model(
    model_config='move_1b',
    data_path='data/owt_1pct_tok',
    output_dir='models_large',
    log_dir='logs_large',
    run_name=None,
    batch_size=1,
    gradient_accumulation_steps=32,
    seq_length=2048,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_epochs=1,
    max_steps=None,
    warmup_steps=None,
    save_interval=1000,
    eval_interval=500,
    log_interval=10,
    resume_from=None,
    use_wandb=False,
    wandb_project='move-large'
):
    """Train large MoVE model with memory optimization."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if run_name is None:
        run_name = f"{model_config}_{int(time.time())}"
    
    # Setup logging
    logger = setup_logging(log_dir, run_name)
    logger.info(f"Starting training run: {run_name}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                'model_config': model_config,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'seq_length': seq_length,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            }
        )
    
    # Create model
    logger.info(f"Creating {model_config} model...")
    model = create_move_large_model(model_config)
    model = model.to(device)
    
    # Estimate memory usage
    estimate_memory_usage(model, batch_size, seq_length, 'fp16')
    
    # Create datasets
    train_dataset = LargeDataset(data_path, max_length=seq_length, split='train')
    val_dataset = LargeDataset(data_path, max_length=seq_length, split='validation')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    if max_steps is None:
        max_steps = steps_per_epoch * num_epochs
    
    logger.info(f"Training configuration:")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Max steps: {max_steps}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, learning_rate, weight_decay)
    scheduler = create_scheduler(optimizer, max_steps, warmup_steps)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_step = 0
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_step, start_epoch, _ = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler
        )
    
    # Training loop
    model.train()
    global_step = start_step
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(pbar):
            if global_step >= max_steps:
                break
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    ppl = torch.exp(loss * gradient_accumulation_steps).item()
                    
                    logger.info(
                        f"Step {global_step}: Loss={loss.item() * gradient_accumulation_steps:.4f}, "
                        f"PPL={ppl:.2f}, LR={current_lr:.2e}"
                    )
                    
                    if use_wandb:
                        wandb.log({
                            'train/loss': loss.item() * gradient_accumulation_steps,
                            'train/perplexity': ppl,
                            'train/learning_rate': current_lr,
                            'step': global_step
                        })
                
                # Evaluation
                if global_step % eval_interval == 0:
                    eval_loss = evaluate_model(model, val_loader, device)
                    eval_ppl = torch.exp(torch.tensor(eval_loss)).item()
                    
                    logger.info(f"Evaluation - Loss: {eval_loss:.4f}, PPL: {eval_ppl:.2f}")
                    
                    if use_wandb:
                        wandb.log({
                            'eval/loss': eval_loss,
                            'eval/perplexity': eval_ppl,
                            'step': global_step
                        })
                    
                    model.train()
                
                # Save checkpoint
                if global_step % save_interval == 0:
                    checkpoint_path = os.path.join(output_dir, f"{run_name}_step_{global_step}.pt")
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, epoch, loss.item(),
                        checkpoint_path, model.get_config()
                    )
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'step': global_step
            })
        
        if global_step >= max_steps:
            break
    
    # Save final model
    final_path = os.path.join(output_dir, f"{run_name}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'training_info': {
            'final_step': global_step,
            'final_epoch': epoch,
            'final_loss': epoch_loss / len(train_loader)
        }
    }, final_path)
    
    logger.info(f"Training completed! Final model saved: {final_path}")
    
    if use_wandb:
        wandb.finish()

def evaluate_model(model, dataloader, device, max_batches=100):
    """Evaluate model and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='Train large MoVE model')
    
    # Model and data
    parser.add_argument('--model_config', type=str, default='move_1b',
                       choices=['move_1b', 'move_3b', 'move_7b'],
                       help='Model configuration')
    parser.add_argument('--data_path', type=str, default='data/owt_1pct_tok',
                       help='Path to tokenized dataset')
    parser.add_argument('--output_dir', type=str, default='models_large',
                       help='Output directory for models')
    parser.add_argument('--log_dir', type=str, default='logs_large',
                       help='Log directory')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for logging')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                       help='Gradient accumulation steps')
    parser.add_argument('--seq_length', type=int, default=2048,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=None,
                       help='Warmup steps')
    
    # Logging and checkpointing
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Weights & Biases
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='move-large',
                       help='Weights & Biases project name')
    
    args = parser.parse_args()
    
    # Train model
    train_large_model(**vars(args))

if __name__ == '__main__':
    main()