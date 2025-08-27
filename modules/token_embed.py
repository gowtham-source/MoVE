#!/usr/bin/env python3
"""
Simplified Token Embedding Module for MoVE Training

This module provides a basic token embedding implementation
that can be trained against extracted Layer-0 vectors.
"""

import torch
import torch.nn as nn

class TokenEmbed(nn.Module):
    """Simple token embedding module.
    
    Args:
        vocab (int): Vocabulary size (default: 32000 for TinyLlama)
        d (int): Embedding dimension (default: 2048 for TinyLlama)
    """
    
    def __init__(self, vocab=32000, d=2048):
        super().__init__()
        self.vocab_size = vocab
        self.embed_dim = d
        self.embed = nn.Embedding(vocab, d)
        
        # Initialize with normal distribution
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Token embeddings [batch_size, seq_len, embed_dim]
        """
        return self.embed(x)
    
    def get_embedding_weights(self):
        """Get embedding weights for analysis."""
        return self.embed.weight.data
    
    def set_embedding_weights(self, weights):
        """Set embedding weights from pretrained model."""
        with torch.no_grad():
            self.embed.weight.copy_(weights)

class TokenEmbedWithLoRA(nn.Module):
    """Token embedding with LoRA adaptation.
    
    This version uses Low-Rank Adaptation for efficient fine-tuning.
    """
    
    def __init__(self, vocab=32000, d=2048, lora_rank=16, lora_alpha=32):
        super().__init__()
        self.vocab_size = vocab
        self.embed_dim = d
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Base embedding (frozen)
        self.embed = nn.Embedding(vocab, d)
        self.embed.weight.requires_grad = False
        
        # LoRA adaptation
        self.lora_A = nn.Parameter(torch.randn(vocab, lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, d))
        self.scaling = lora_alpha / lora_rank
        
    def forward(self, x):
        """Forward pass with LoRA adaptation.
        
        Args:
            x (torch.Tensor): Input token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Adapted token embeddings [batch_size, seq_len, embed_dim]
        """
        # Base embeddings
        base_embed = self.embed(x)
        
        # LoRA adaptation
        lora_embed = torch.matmul(self.lora_A[x], self.lora_B) * self.scaling
        
        return base_embed + lora_embed
    
    def load_base_weights(self, weights):
        """Load base embedding weights from pretrained model."""
        with torch.no_grad():
            self.embed.weight.copy_(weights)

if __name__ == "__main__":
    # Test the module
    print("Testing TokenEmbed module...")
    
    # Create model
    model = TokenEmbed(vocab=32000, d=2048)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, 2048]")
    
    # Test LoRA version
    print("\nTesting TokenEmbedWithLoRA...")
    lora_model = TokenEmbedWithLoRA(vocab=32000, d=2048, lora_rank=16)
    lora_output = lora_model(input_ids)
    print(f"LoRA output shape: {lora_output.shape}")
    
    print("\nTokenEmbed module test complete!")