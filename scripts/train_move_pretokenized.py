#!/usr/bin/env python3
"""
MoVE Training Script for Pre-tokenized Dataset

This script trains the MoVE model with datasets that contain input_ids and attention_mask.
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
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import MoVE model
from move_large import MoVELarge

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.85)

class MoVE300MConfig:
    """Configuration for 300M parameter MoVE model."""
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 768
        self.num_layers = 12
        self.num_heads = 8  # k=64 must be divisible by num_heads, 64/8=8 works
        self.max_seq_len = 1024
        self.moe_experts = 8
        self.moe_topk = 2
        self.use_lora = False
        self.dropout = 0.1
        self.use_checkpoint = True
        self.tie_weights = True

class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized data with input_ids and attention_mask."""
    
    def __init__(self, dataset, max_length=1024):
        self.dataset = dataset
        self.max_length = max_length
        print(f"Dataset initialized with {len(dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Get pre-tokenized data
            input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
            
            # Validate data
            if len(input_ids) == 0 or len(attention_mask) == 0:
                # Return a dummy sample for empty data
                input_ids = torch.tensor([0] * 10, dtype=torch.long)
                attention_mask = torch.tensor([1] * 10, dtype=torch.long)
            
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
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a dummy sample
            dummy_length = min(50, self.max_length)
            return {
                'input_ids': torch.tensor([0] * dummy_length, dtype=torch.long),
                'attention_mask': torch.tensor([1] * dummy_length, dtype=torch.long),
                'labels': torch.tensor([0] * dummy_length, dtype=torch.long)
            }

class MoVETrainer:
    """Trainer class for MoVE model with conservative settings."""
    
    def __init__(self, model, train_loader, config, args, device):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.args = args
        self.device = device
        
        # Optimizer with conservative settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(args.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(config), f, indent=2)
    
    def train_step(self, batch):
        """Single training step with extensive safety checks."""
        self.model.train()
        
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Safety checks
        if torch.isnan(input_ids.float()).any():
            return None
        
        if torch.isnan(attention_mask.float()).any():
            return None
        
        if torch.isnan(labels.float()).any():
            return None
        
        # Check for valid tokens
        if attention_mask.sum() == 0:
            return None
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                # Manual loss calculation if model doesn't return loss
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
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
        
        # Check loss validity
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        if loss.item() > 50.0:  # Extremely high loss
            return None
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        return loss.item()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            loss = self.train_step(batch)
            
            if loss is not None:
                total_loss += loss
                valid_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'valid': valid_batches,
                    'step': step
                })
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Logging
                if step % self.args.log_interval == 0 and valid_batches > 0:
                    print(f"Step {step}, Avg Loss: {avg_loss:.4f}, Valid Batches: {valid_batches}")
                
                # Save checkpoint
                if step % self.args.save_interval == 0 and step > 0:
                    self.save_checkpoint(epoch, step, avg_loss)
            
            # Memory cleanup
            if step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    def save_checkpoint(self, epoch, step, loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'config': vars(self.config),
            'args': vars(self.args)
        }
        
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        print("Starting MoVE training...")
        
        for epoch in range(self.args.num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch, -1, avg_loss)
        
        # Save final model
        final_path = os.path.join(self.args.output_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.config),
            'args': vars(self.args)
        }, final_path)
        
        print(f"Training completed! Final model saved to {final_path}")

def create_move_model(config):
    """Create MoVE model with given configuration."""
    model = MoVELarge(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        moe_experts=config.moe_experts,
        moe_topk=config.moe_topk,
        use_lora=config.use_lora,
        dropout=config.dropout,
        use_checkpoint=config.use_checkpoint,
        tie_weights=config.tie_weights
    )
    return model

def main():
    parser = argparse.ArgumentParser(description='Train MoVE model on pre-tokenized dataset')
    parser.add_argument('--dataset_path', required=True, help='Path to pre-tokenized dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Save interval')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    print(f"Created data loader with {len(train_loader)} batches")
    
    # Create model configuration
    config = MoVE300MConfig()
    
    # Create model
    print("Creating MoVE model...")
    try:
        model = create_move_model(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (fp32)")
        
    except Exception as e:
        print(f"Failed to create model: {e}")
        return
    
    # Create trainer
    trainer = MoVETrainer(model, train_loader, config, args, device)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(-1, -1, 0.0)
    except Exception as e:
        print(f"Training failed: {e}")
        trainer.save_checkpoint(-1, -1, 0.0)

if __name__ == '__main__':
    main()