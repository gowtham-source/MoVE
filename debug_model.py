#!/usr/bin/env python3
"""
Debug script to test MoVE model initialization
"""

import torch
import sys
sys.path.append('.')

from scripts.optimize_1b_for_rtx4090 import create_optimized_move_model
from transformers import AutoTokenizer

def test_model_creation():
    print("Testing MoVE model creation...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Create model
    print("Creating model...")
    try:
        model, config = create_optimized_move_model('500m', len(tokenizer))
        print(f"Model created successfully with {model.count_parameters()['total']:,} parameters")
        
        # Test forward pass
        print("Testing forward pass...")
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
        
        print(f"Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"Output logits shape: {outputs['logits'].shape}")
            print("Forward pass successful!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_creation()