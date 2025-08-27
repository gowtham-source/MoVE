#!/usr/bin/env python3
"""
Debug script to test training script components
"""

import torch
import sys
sys.path.append('.')

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scripts.optimize_1b_for_rtx4090 import create_optimized_move_model

def collate_fn(batch):
    """Custom collate function to handle batch processing"""
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def test_training_components():
    print("Testing training components...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk('data/arxiv_dataset_500m')
    print(f"Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
    
    # Create dataloader
    print("Creating dataloader...")
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"Dataloader created with {len(train_dataloader)} batches")
    
    # Test one batch
    print("Testing batch processing...")
    batch = next(iter(train_dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    
    # Create model
    print("Creating model...")
    model, config = create_optimized_move_model('500m', len(tokenizer))
    print(f"Model created with {model.count_parameters()['total']:,} parameters")
    
    # Test forward pass with batch
    print("Testing forward pass with batch...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(batch['input_ids'], batch['attention_mask'])
        print(f"Output logits shape: {outputs['logits'].shape}")
        print("Forward pass with batch successful!")
    
    print("All components working correctly!")

if __name__ == "__main__":
    test_training_components()