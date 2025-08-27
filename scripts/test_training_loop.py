#!/usr/bin/env python3
"""
Simple test to verify training loop functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import csv
import time
import math
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from move.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location("move_module", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "move.py"))
move_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(move_module)
MoVE = move_module.MoVE
create_move_model = move_module.create_move_model

class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=256):
        self.num_samples = num_samples
        self.seq_length = seq_length
        print(f"Creating dataset with {num_samples} samples, seq_length={seq_length}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 32000, (self.seq_length,), dtype=torch.long),
            'attention_mask': torch.ones(self.seq_length, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def test_training_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SimpleDataset(num_samples=1000, seq_length=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataloader length: {len(dataloader)}")
    
    # Create model
    model = create_move_model('small')
    model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Setup logging
    log_file = 'logs/test_training.csv'
    os.makedirs('logs', exist_ok=True)
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'perplexity', 'time'])
    
    # Training loop
    model.train()
    global_step = 0
    start_time = time.time()
    max_steps = 50
    
    print(f"Starting training for {max_steps} steps...")
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        if global_step >= max_steps:
            break
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Prepare inputs and targets
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        try:
            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
            else:
                logits = outputs
            
            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Log every 10 steps
            if global_step % 10 == 0:
                perplexity = math.exp(loss.item())
                elapsed_time = time.time() - start_time
                
                print(f"Step {global_step}: Loss={loss.item():.4f}, PPL={perplexity:.2f}")
                
                # Log to CSV
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, loss.item(), perplexity, elapsed_time])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step': global_step
            })
            
        except Exception as e:
            print(f"Error at step {global_step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"Training completed! Ran {global_step} steps.")
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    test_training_loop()