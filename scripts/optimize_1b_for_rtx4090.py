#!/usr/bin/env python3
"""
MoVE 1B Model Optimization for RTX 4090

This script provides optimized configurations and model variants
to ensure the 1B MoVE model fits within RTX 4090's 16GB VRAM.

Features:
- Memory-optimized model configurations
- Gradient checkpointing strategies
- Efficient training parameters
- Memory usage estimation and validation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from move_large import MoVELarge

class OptimizedMoVEConfig:
    """Optimized MoVE configurations for RTX 4090."""
    
    @staticmethod
    def get_rtx4090_optimized_1b() -> Dict[str, Any]:
        """Get 1B model config optimized for RTX 4090."""
        return {
            'vocab_size': 32000,
            'hidden_size': 2048,      # Reduced from 2560
            'intermediate_size': 5504, # Reduced from 6912
            'num_hidden_layers': 22,   # Reduced from 24
            'num_attention_heads': 32, # Reduced from 40
            'num_key_value_heads': 4,  # Reduced from 5
            'max_position_embeddings': 2048,
            'rms_norm_eps': 1e-5,
            'use_cache': True,
            'tie_word_embeddings': False,
            'rope_theta': 10000.0,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.0,
            
            # MoVE specific
            'num_experts': 4,          # Reduced from 8
            'num_experts_per_tok': 2,
            'router_aux_loss_coef': 0.01,
            
            # Memory optimization
            'gradient_checkpointing': True,
            'use_flash_attention': True,
            'memory_efficient_attention': True
        }
    
    @staticmethod
    def get_rtx4090_optimized_700m() -> Dict[str, Any]:
        """Get 700M model config optimized for RTX 4090."""
        return {
            'vocab_size': 32000,
            'hidden_size': 1536,       # Further reduced
            'intermediate_size': 4096, # Further reduced
            'num_hidden_layers': 20,   # Reduced
            'num_attention_heads': 24, # Reduced
            'num_key_value_heads': 4,
            'max_position_embeddings': 2048,
            'rms_norm_eps': 1e-5,
            'use_cache': True,
            'tie_word_embeddings': False,
            'rope_theta': 10000.0,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.0,
            
            # MoVE specific
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'router_aux_loss_coef': 0.01,
            
            # Memory optimization
            'gradient_checkpointing': True,
            'use_flash_attention': True,
            'memory_efficient_attention': True
        }
    
    @staticmethod
    def get_rtx4090_optimized_500m() -> Dict[str, Any]:
        """Get 500M model config optimized for RTX 4090."""
        return {
            'vocab_size': 32000,
            'hidden_size': 1024,       # Significantly reduced
            'intermediate_size': 2816, # Significantly reduced
            'num_hidden_layers': 18,   # Reduced
            'num_attention_heads': 16, # Reduced
            'num_key_value_heads': 4,
            'max_position_embeddings': 2048,
            'rms_norm_eps': 1e-5,
            'use_cache': True,
            'tie_word_embeddings': False,
            'rope_theta': 10000.0,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.0,
            
            # MoVE specific
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'router_aux_loss_coef': 0.01,
            
            # Memory optimization
            'gradient_checkpointing': True,
            'use_flash_attention': True,
            'memory_efficient_attention': True
        }

class MemoryEstimator:
    """Estimate memory usage for MoVE models."""
    
    @staticmethod
    def estimate_model_memory(config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory usage for a given model configuration."""
        
        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']
        intermediate_size = config['intermediate_size']
        num_layers = config['num_hidden_layers']
        num_heads = config['num_attention_heads']
        num_experts = config.get('num_experts', 1)
        
        # Parameter counts
        # Embedding layers
        embedding_params = vocab_size * hidden_size
        
        # Attention layers per layer
        attention_params_per_layer = (
            hidden_size * hidden_size * 3 +  # Q, K, V projections
            hidden_size * hidden_size        # Output projection
        )
        
        # FFN layers per layer (with MoE)
        ffn_params_per_layer = (
            hidden_size * intermediate_size * 2 * num_experts +  # Gate and up projections
            intermediate_size * hidden_size * num_experts +      # Down projection
            hidden_size * num_experts                            # Router
        )
        
        # Layer norm parameters
        layernorm_params_per_layer = hidden_size * 2  # Pre and post attention
        
        # Total parameters per layer
        params_per_layer = attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer
        
        # Total model parameters
        total_params = embedding_params + (params_per_layer * num_layers) + hidden_size  # Final layer norm
        
        # Memory calculations (in GB)
        bytes_per_param = 4  # FP32
        model_memory_gb = (total_params * bytes_per_param) / (1024**3)
        
        # Training memory components
        gradients_memory_gb = model_memory_gb  # Same as model
        optimizer_memory_gb = model_memory_gb * 2  # AdamW states (momentum + variance)
        
        # Activation memory (rough estimate)
        batch_size = 1  # Conservative estimate
        seq_len = config.get('max_position_embeddings', 2048)
        activation_memory_gb = (
            batch_size * seq_len * hidden_size * num_layers * 4 * bytes_per_param
        ) / (1024**3)
        
        # Total training memory
        total_training_memory_gb = (
            model_memory_gb + 
            gradients_memory_gb + 
            optimizer_memory_gb + 
            activation_memory_gb
        )
        
        # With gradient checkpointing (reduces activation memory)
        if config.get('gradient_checkpointing', False):
            activation_memory_gb *= 0.3  # Rough reduction factor
            total_training_memory_gb = (
                model_memory_gb + 
                gradients_memory_gb + 
                optimizer_memory_gb + 
                activation_memory_gb
            )
        
        return {
            'total_params': total_params,
            'model_memory_gb': model_memory_gb,
            'gradients_memory_gb': gradients_memory_gb,
            'optimizer_memory_gb': optimizer_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'total_training_memory_gb': total_training_memory_gb,
            'gradient_checkpointing': config.get('gradient_checkpointing', False)
        }
    
    @staticmethod
    def check_rtx4090_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if configuration is compatible with RTX 4090."""
        memory_info = MemoryEstimator.estimate_model_memory(config)
        
        rtx4090_memory_gb = 16.0
        available_memory_gb = rtx4090_memory_gb * 0.9  # Leave 10% headroom
        
        fits = memory_info['total_training_memory_gb'] <= available_memory_gb
        
        # Recommended batch size
        if fits:
            remaining_memory = available_memory_gb - memory_info['total_training_memory_gb']
            # Estimate batch size based on remaining memory
            seq_len = config.get('max_position_embeddings', 2048)
            hidden_size = config['hidden_size']
            memory_per_sample = (seq_len * hidden_size * 4) / (1024**3)  # Rough estimate
            max_batch_size = max(1, int(remaining_memory / memory_per_sample))
        else:
            max_batch_size = 1
        
        return {
            **memory_info,
            'rtx4090_memory_gb': rtx4090_memory_gb,
            'available_memory_gb': available_memory_gb,
            'fits_in_rtx4090': fits,
            'memory_utilization': memory_info['total_training_memory_gb'] / available_memory_gb,
            'recommended_batch_size': max_batch_size
        }

def create_optimized_move_model(config_name: str = '700m', vocab_size: int = 32000) -> nn.Module:
    """Create optimized MoVE model for RTX 4090."""
    
    config_map = {
        '1b': OptimizedMoVEConfig.get_rtx4090_optimized_1b,
        '700m': OptimizedMoVEConfig.get_rtx4090_optimized_700m,
        '500m': OptimizedMoVEConfig.get_rtx4090_optimized_500m
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(config_map.keys())}")
    
    config = config_map[config_name]()
    config['vocab_size'] = vocab_size
    
    # Create model with corrected parameter mapping
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
    
    # Note: Gradient checkpointing is handled within MoVELargeLayer
    
    return model, config

def get_optimized_training_config(model_size: str = '700m') -> Dict[str, Any]:
    """Get optimized training configuration for RTX 4090."""
    
    base_config = {
        'use_amp': True,  # Mixed precision
        'gradient_checkpointing': True,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_warmup',
        
        # Memory optimization
        'dataloader_num_workers': 2,
        'dataloader_pin_memory': True,
        'torch_compile': False,  # Can cause memory issues
        
        # Logging
        'log_interval': 50,
        'eval_interval': 500,
        'save_interval': 1000
    }
    
    # Model-specific configurations
    if model_size == '1b':
        base_config.update({
            'batch_size': 1,
            'gradient_accumulation_steps': 16,
            'learning_rate': 8e-5,
            'num_epochs': 2
        })
    elif model_size == '700m':
        base_config.update({
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1e-4,
            'num_epochs': 3
        })
    elif model_size == '500m':
        base_config.update({
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 1.2e-4,
            'num_epochs': 3
        })
    
    return base_config

def analyze_all_configurations():
    """Analyze all model configurations for RTX 4090 compatibility."""
    
    configs = {
        '1B Optimized': OptimizedMoVEConfig.get_rtx4090_optimized_1b(),
        '700M Optimized': OptimizedMoVEConfig.get_rtx4090_optimized_700m(),
        '500M Optimized': OptimizedMoVEConfig.get_rtx4090_optimized_500m()
    }
    
    print("=== MoVE Model Analysis for RTX 4090 ===")
    print()
    
    for name, config in configs.items():
        print(f"--- {name} ---")
        analysis = MemoryEstimator.check_rtx4090_compatibility(config)
        
        print(f"Parameters: {analysis['total_params']:,}")
        print(f"Model memory: {analysis['model_memory_gb']:.2f} GB")
        print(f"Training memory: {analysis['total_training_memory_gb']:.2f} GB")
        print(f"Memory utilization: {analysis['memory_utilization']:.1%}")
        print(f"Fits in RTX 4090: {'✓' if analysis['fits_in_rtx4090'] else '✗'}")
        print(f"Recommended batch size: {analysis['recommended_batch_size']}")
        print(f"Gradient checkpointing: {'✓' if analysis['gradient_checkpointing'] else '✗'}")
        print()

def save_optimized_configs(output_dir: str = 'configs'):
    """Save optimized configurations to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    configs = {
        'move_1b_rtx4090': {
            'model': OptimizedMoVEConfig.get_rtx4090_optimized_1b(),
            'training': get_optimized_training_config('1b')
        },
        'move_700m_rtx4090': {
            'model': OptimizedMoVEConfig.get_rtx4090_optimized_700m(),
            'training': get_optimized_training_config('700m')
        },
        'move_500m_rtx4090': {
            'model': OptimizedMoVEConfig.get_rtx4090_optimized_500m(),
            'training': get_optimized_training_config('500m')
        }
    }
    
    for name, config in configs.items():
        config_path = os.path.join(output_dir, f'{name}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Optimize MoVE 1B for RTX 4090')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze all configurations')
    parser.add_argument('--save_configs', action='store_true',
                       help='Save optimized configurations')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Directory to save configurations')
    parser.add_argument('--test_model', type=str, choices=['1b', '700m', '500m'],
                       help='Test creating a specific model')
    parser.add_argument('--vocab_size', type=int, default=32000,
                       help='Vocabulary size')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_all_configurations()
    
    if args.save_configs:
        save_optimized_configs(args.config_dir)
    
    if args.test_model:
        print(f"Testing {args.test_model} model creation...")
        try:
            model, config = create_optimized_move_model(args.test_model, args.vocab_size)
            analysis = MemoryEstimator.check_rtx4090_compatibility(config)
            
            print(f"Model created successfully!")
            print(f"Parameters: {analysis['total_params']:,}")
            print(f"Estimated training memory: {analysis['total_training_memory_gb']:.2f} GB")
            print(f"RTX 4090 compatible: {'✓' if analysis['fits_in_rtx4090'] else '✗'}")
            
        except Exception as e:
            print(f"Error creating model: {e}")

if __name__ == '__main__':
    main()