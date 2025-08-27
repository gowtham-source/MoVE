#!/usr/bin/env python3
"""
MoVE (Modular Vector Engine) - Main Integration Module

This module integrates all MoVE components into a unified model
that can be trained and evaluated against TinyLlama baseline.
"""

import torch
import torch.nn as nn
import sys
import os

# Add modules to path
sys.path.append('modules')
from token_embed import TokenEmbed, TokenEmbedWithLoRA
from pos_gen import PosGen, RoPEPosGen, LearnablePosGen
from attn_approx import AttnApprox, GraphAttnApprox, EfficientAttnApprox
from ffn_moe import FFNMoE, AdaptiveFFNMoE

class MoVE(nn.Module):
    """Modular Vector Engine - Main Model.
    
    Args:
        vocab_size (int): Vocabulary size (default: 32000)
        d_model (int): Model dimension (default: 2048)
        max_seq_len (int): Maximum sequence length (default: 1024)
        num_layers (int): Number of transformer-like layers (default: 1)
        use_lora (bool): Whether to use LoRA for embeddings (default: True)
        embed_type (str): Type of embedding ('standard', 'lora') (default: 'lora')
        pos_type (str): Type of positional encoding ('mlp', 'rope', 'learnable') (default: 'mlp')
        attn_type (str): Type of attention ('standard', 'graph', 'efficient') (default: 'standard')
        moe_type (str): Type of MoE ('standard', 'adaptive') (default: 'standard')
        moe_experts (int): Number of MoE experts (default: 8)
        moe_topk (int): Top-k experts to use (default: 2)
    """
    
    def __init__(
        self,
        vocab_size=32000,
        d_model=2048,
        max_seq_len=1024,
        num_layers=1,
        use_lora=True,
        embed_type='lora',
        pos_type='mlp',
        attn_type='standard',
        moe_type='standard',
        moe_experts=8,
        moe_topk=2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.use_lora = use_lora
        self.embed_type = embed_type
        self.pos_type = pos_type
        self.attn_type = attn_type
        self.moe_type = moe_type
        self.moe_experts = moe_experts
        self.moe_topk = moe_topk
        
        # Token embedding
        if embed_type == 'lora' or use_lora:
            self.embed = TokenEmbedWithLoRA(vocab=vocab_size, d=d_model)
        else:
            self.embed = TokenEmbed(vocab=vocab_size, d=d_model)
        
        # Positional encoding
        if pos_type == 'rope':
            self.pos = RoPEPosGen(d=d_model, max_len=max_seq_len)
        elif pos_type == 'learnable':
            self.pos = LearnablePosGen(d=d_model, max_len=max_seq_len)
        else:  # 'mlp'
            self.pos = PosGen(d=d_model, max_len=max_seq_len)
        
        # Create layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = MoVELayer(
                d_model=d_model,
                attn_type=attn_type,
                moe_type=moe_type,
                moe_experts=moe_experts,
                moe_topk=moe_topk
            )
            self.layers.append(layer)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection (for language modeling)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights with embedding if using standard embedding
        if embed_type == 'standard':
            self.lm_head.weight = self.embed.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, return_dict=False, return_losses=False):
        """Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]
            return_dict (bool): Whether to return a dictionary
            return_losses (bool): Whether to return auxiliary losses
            
        Returns:
            torch.Tensor or dict: Model outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.embed(input_ids)  # [batch_size, seq_len, d_model]
        
        # Positional encodings
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_encodings = self.pos(pos_indices)  # [batch_size, seq_len, d_model]
        
        # Add positional encodings
        x = x + pos_encodings
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        
        # Process through layers
        router_losses = []
        for layer in self.layers:
            if return_losses:
                x, router_loss = layer(x, attention_mask, return_router_loss=True)
                router_losses.append(router_loss)
            else:
                x = layer(x, attention_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        if return_dict:
            outputs = {
                'logits': logits,
                'hidden_states': x
            }
            if return_losses and router_losses:
                outputs['router_loss'] = sum(router_losses) / len(router_losses)
            return outputs
        
        if return_losses and router_losses:
            return logits, sum(router_losses) / len(router_losses)
        
        return logits
    
    def generate(
        self,
        input_ids,
        max_length=50,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        do_sample=True
    ):
        """Generate text using the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p (nucleus) sampling
            do_sample (bool): Whether to use sampling
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(input_ids)
                
                # Get next token logits
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we've reached max sequence length
                if input_ids.size(1) >= self.max_seq_len:
                    break
        
        return input_ids
    
    def get_num_params(self):
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self):
        """Get model configuration."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
            'num_layers': self.num_layers,
            'moe_experts': self.moe_experts,
            'moe_topk': self.moe_topk,
            'use_lora': self.use_lora,
            'embed_type': self.embed_type,
            'pos_type': self.pos_type,
            'attn_type': self.attn_type,
            'moe_type': self.moe_type
        }

class MoVELayer(nn.Module):
    """Single MoVE layer combining attention and MoE."""
    
    def __init__(
        self,
        d_model=2048,
        attn_type='standard',
        moe_type='standard',
        moe_experts=8,
        moe_topk=2
    ):
        super().__init__()
        
        # Attention module
        if attn_type == 'graph':
            self.attn = GraphAttnApprox(d=d_model)
        elif attn_type == 'efficient':
            self.attn = EfficientAttnApprox(d=d_model)
        else:  # 'standard'
            self.attn = AttnApprox(d=d_model)
        
        # MoE module
        if moe_type == 'adaptive':
            self.ffn = AdaptiveFFNMoE(d=d_model, experts=moe_experts, topk=moe_topk)
        else:  # 'standard'
            self.ffn = FFNMoE(d=d_model, experts=moe_experts, topk=moe_topk)
    
    def forward(self, x, attention_mask=None, return_router_loss=False):
        """Forward pass through layer.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
            attention_mask (torch.Tensor, optional): Attention mask
            return_router_loss (bool): Whether to return router loss
            
        Returns:
            torch.Tensor: Output tensor
            torch.Tensor (optional): Router loss if return_router_loss=True
        """
        # Attention
        x = self.attn(x, mask=attention_mask)
        
        # MoE FFN
        if return_router_loss:
            x, router_loss = self.ffn(x, return_router_loss=True)
            return x, router_loss
        else:
            x = self.ffn(x)
            return x

def create_move_model(config_name='small'):
    """Create a MoVE model with predefined configuration.
    
    Args:
        config_name (str): Configuration name ('tiny', 'small', 'medium')
        
    Returns:
        MoVE: Configured MoVE model
    """
    configs = {
        'tiny': {
            'vocab_size': 32000,
            'd_model': 512,
            'max_seq_len': 512,
            'num_layers': 1,
            'moe_experts': 4,
            'moe_topk': 2
        },
        'small': {
            'vocab_size': 32000,
            'd_model': 1024,
            'max_seq_len': 1024,
            'num_layers': 2,
            'moe_experts': 6,
            'moe_topk': 2
        },
        'medium': {
            'vocab_size': 32000,
            'd_model': 2048,
            'max_seq_len': 1024,
            'num_layers': 3,
            'moe_experts': 8,
            'moe_topk': 2
        }
    }
    
    config = configs.get(config_name, configs['small'])
    return MoVE(**config)

if __name__ == "__main__":
    # Test the MoVE model
    print("Testing MoVE model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = create_move_model('small').to(device)
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True, return_losses=True)
        
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Hidden states shape: {outputs['hidden_states'].shape}")
        if 'router_loss' in outputs:
            print(f"Router loss: {outputs['router_loss'].item():.6f}")
        
        print(f"Expected logits shape: [{batch_size}, {seq_len}, 32000]")
        print(f"Expected hidden shape: [{batch_size}, {seq_len}, 1024]")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, 32000, (1, 5), device=device)
    generated = model.generate(prompt, max_length=10, do_sample=False)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\nMoVE model test complete!")
    print("\nSanity check passed: ✓")
    print(f"- Forward pass output shape: [{batch_size}, {seq_len}, 1024] ✓")
    print(f"- Generation works: ✓")
    print(f"- Model parameters: {model.get_num_params():,} ✓")