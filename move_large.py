"""Enhanced MoVE Architecture for Large-Scale Training

Optimized for RTX 4090 (16GB VRAM) with memory-efficient configurations.
Supports 1B-7B parameter models with gradient checkpointing and mixed precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Dict, Any

# Import existing modules
from modules.token_embed import TokenEmbed
from modules.pos_gen import PosGen
from modules.attn_approx import AttnApprox
from modules.ffn_moe import FFNMoE, AdaptiveFFNMoE

class MoVELargeLayer(nn.Module):
    """Enhanced MoVE layer with memory optimization."""
    
    def __init__(self, d_model, num_heads, moe_experts=8, moe_topk=2, 
                 use_lora=True, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        
        # Attention with memory optimization
        self.attn = AttnApprox(
            d=d_model,
            k=64,  # Low-rank dimension
            num_heads=num_heads
        )
        
        # FFN with MoE
        self.ffn = AdaptiveFFNMoE(
            d=d_model,
            experts=moe_experts,
            topk=moe_topk,
            expert_dim=None,  # Uses default 4*d
            adaptive_topk=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _forward_impl(self, x, attention_mask=None):
        """Forward implementation for checkpointing."""
        # Pre-norm attention
        normed_x = self.norm1(x)
        attn_out = self.attn(normed_x, attention_mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)
        
        return x
    
    def forward(self, x, attention_mask=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, attention_mask, use_reentrant=False)
        else:
            return self._forward_impl(x, attention_mask)

class MoVELarge(nn.Module):
    """Large-scale MoVE model with memory optimization."""
    
    def __init__(self, vocab_size=32000, d_model=2048, num_layers=24, num_heads=16,
                 max_seq_len=4096, moe_experts=8, moe_topk=2, use_lora=True,
                 dropout=0.1, use_checkpoint=True, tie_weights=True):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_lora = use_lora
        self.moe_experts = moe_experts
        self.moe_topk = moe_topk
        self.use_checkpoint = use_checkpoint
        self.tie_weights = tie_weights
        
        # Token embedding
        self.token_embed = TokenEmbed(
            vocab=vocab_size,
            d=d_model
        )
        
        # Positional encoding
        self.pos_gen = PosGen(
            d=d_model,
            max_len=max_seq_len
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MoVELargeLayer(
                d_model=d_model,
                num_heads=num_heads,
                moe_experts=moe_experts,
                moe_topk=moe_topk,
                use_lora=use_lora,
                dropout=dropout,
                use_checkpoint=use_checkpoint
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights if specified
        if tie_weights:
            self.lm_head.weight = self.token_embed.embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_config(self):
        """Get model configuration."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'use_lora': self.use_lora,
            'moe_experts': self.moe_experts,
            'moe_topk': self.moe_topk,
            'use_checkpoint': self.use_checkpoint,
            'tie_weights': self.tie_weights
        }
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert attention mask to causal mask for transformer layers
        # Create causal mask: [seq_len, seq_len]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        # Convert to additive mask (0 for allowed, -inf for masked)
        causal_mask = (1.0 - causal_mask) * -1e9
        # Expand for batch and heads: [batch_size, 1, seq_len, seq_len]
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # Token embeddings
        x = self.token_embed(input_ids)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_gen(positions)
        x = x + pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

# Model configurations optimized for RTX 4090
MODEL_CONFIGS = {
    'move_1b': {
        'vocab_size': 32000,
        'd_model': 2048,
        'num_layers': 24,
        'num_heads': 16,
        'max_seq_len': 4096,
        'moe_experts': 8,
        'moe_topk': 2,
        'use_checkpoint': True,
        'tie_weights': True
    },
    'move_3b': {
        'vocab_size': 32000,
        'd_model': 2560,
        'num_layers': 32,
        'num_heads': 20,
        'max_seq_len': 4096,
        'moe_experts': 8,
        'moe_topk': 2,
        'use_checkpoint': True,
        'tie_weights': True
    },
    'move_7b': {
        'vocab_size': 32000,
        'd_model': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'max_seq_len': 4096,
        'moe_experts': 8,
        'moe_topk': 2,
        'use_checkpoint': True,
        'tie_weights': True
    }
}

def create_move_large_model(config_name='move_1b', **kwargs):
    """Create a large MoVE model with specified configuration."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    model = MoVELarge(**config)
    
    # Print model info
    param_count = model.count_parameters()
    print(f"Created {config_name} model:")
    print(f"  Total parameters: {param_count['total']:,}")
    print(f"  Trainable parameters: {param_count['trainable']:,}")
    print(f"  Model size: ~{param_count['total'] * 4 / 1e9:.2f}GB (FP32)")
    print(f"  Model size: ~{param_count['total'] * 2 / 1e9:.2f}GB (FP16)")
    
    return model

def estimate_memory_usage(model, batch_size=1, seq_length=2048, precision='fp16'):
    """Estimate memory usage for training."""
    param_count = model.count_parameters()['total']
    
    # Parameter memory
    if precision == 'fp16':
        param_memory = param_count * 2  # 2 bytes per parameter
        grad_memory = param_count * 2   # gradients
        optimizer_memory = param_count * 8  # Adam states (fp32)
    else:
        param_memory = param_count * 4  # 4 bytes per parameter
        grad_memory = param_count * 4   # gradients
        optimizer_memory = param_count * 8  # Adam states
    
    # Activation memory (rough estimate)
    d_model = model.d_model
    num_layers = model.num_layers
    activation_memory = batch_size * seq_length * d_model * num_layers * 4  # rough estimate
    
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
    
    print(f"\nMemory Estimation (batch_size={batch_size}, seq_length={seq_length}, {precision}):")
    print(f"  Parameters: {param_memory / 1e9:.2f}GB")
    print(f"  Gradients: {grad_memory / 1e9:.2f}GB")
    print(f"  Optimizer: {optimizer_memory / 1e9:.2f}GB")
    print(f"  Activations: {activation_memory / 1e9:.2f}GB")
    print(f"  Total: {total_memory / 1e9:.2f}GB")
    
    return total_memory

if __name__ == "__main__":
    # Test model creation
    print("Testing MoVE Large model configurations...\n")
    
    for config_name in MODEL_CONFIGS.keys():
        print(f"\n{'='*50}")
        model = create_move_large_model(config_name)
        estimate_memory_usage(model, batch_size=1, seq_length=2048, precision='fp16')
        print(f"{'='*50}")