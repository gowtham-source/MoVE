#!/usr/bin/env python3
"""
Debug script to test MoVELarge initialization
"""

import torch
import sys
sys.path.append('.')

from move_large import MoVELarge
from scripts.optimize_1b_for_rtx4090 import OptimizedMoVEConfig

def test_move_large_creation():
    print("Testing MoVELarge creation...")
    
    # Get 500m config
    print("Getting 500m config...")
    config = OptimizedMoVEConfig.get_rtx4090_optimized_500m()
    config['vocab_size'] = 32000
    print(f"Config: {config}")
    
    # Create model step by step
    print("Creating MoVELarge model...")
    try:
        model = MoVELarge(
            vocab_size=config['vocab_size'],
            d_model=config['hidden_size'],
            num_layers=config['num_hidden_layers'],
            moe_experts=config['num_experts'],
            moe_topk=config['num_experts_per_tok'],
            dropout=config['hidden_dropout'],
            tie_weights=config['tie_word_embeddings'],
            use_lora=config.get('use_lora', False)
        )
        print("MoVELarge created successfully!")
        
        # Count parameters
        param_count = model.count_parameters()
        print(f"Total parameters: {param_count['total']:,}")
        
        # Test forward pass
        print("Testing forward pass...")
        model.eval()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"Output logits shape: {outputs['logits'].shape}")
            print("Forward pass successful!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_move_large_creation()