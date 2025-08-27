#!/usr/bin/env python3
"""
Debug script to check dataset and dataloader behavior
"""

import torch
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from train_move.py using importlib
import importlib.util
spec = importlib.util.spec_from_file_location("train_move", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_move.py"))
train_move = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_move)
TextDataset = train_move.TextDataset
collate_fn = train_move.collate_fn

def debug_dataset():
    print("=== Dataset Debug ===")
    
    # Create training dataset
    train_dataset = TextDataset('dummy_path', max_length=256, split='train')
    print(f"Training dataset length: {len(train_dataset)}")
    
    # Create evaluation dataset
    eval_dataset = TextDataset('dummy_path', max_length=256, split='validation')
    print(f"Evaluation dataset length: {len(eval_dataset)}")
    
    # Test a few samples
    print("\n=== Sample Data ===")
    for i in range(3):
        sample = train_dataset[i]
        print(f"Sample {i}: input_ids length = {len(sample['input_ids'])}, attention_mask length = {len(sample['attention_mask'])}")
    
    # Create DataLoader
    print("\n=== DataLoader Debug ===")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"DataLoader length: {len(train_dataloader)}")
    print(f"Expected batches: {len(train_dataset) // 2}")
    
    # Test first few batches
    print("\n=== Batch Testing ===")
    batch_count = 0
    for batch_idx, batch in enumerate(train_dataloader):
        batch_count += 1
        print(f"Batch {batch_idx}: input_ids shape = {batch['input_ids'].shape}, attention_mask shape = {batch['attention_mask'].shape}")
        if batch_idx >= 4:  # Only test first 5 batches
            break
    
    print(f"\nTotal batches processed: {batch_count}")
    print(f"Total batches available: {len(train_dataloader)}")

if __name__ == "__main__":
    debug_dataset()