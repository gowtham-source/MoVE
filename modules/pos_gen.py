#!/usr/bin/env python3
"""
Positional Vector Generator Module for MoVE

This module provides positional encoding generation that can be trained
to match RoPE-style output from TinyLlama.
"""

import torch
import torch.nn as nn
import math

class PosGen(nn.Module):
    """Positional vector generator using MLP.
    
    Args:
        d (int): Hidden dimension (default: 2048)
        max_len (int): Maximum sequence length (default: 1024)
    """
    
    def __init__(self, d=2048, max_len=1024):
        super().__init__()
        self.d = d
        self.max_len = max_len
        
        # MLP for position encoding
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, pos):
        """Generate positional encodings.
        
        Args:
            pos (torch.Tensor): Position indices [seq_len] or [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Positional encodings [seq_len, d] or [batch_size, seq_len, d]
        """
        # Handle different input shapes
        original_shape = pos.shape
        if pos.dim() == 1:
            # [seq_len] -> [seq_len, 1]
            pos_input = pos.unsqueeze(-1).float()
        else:
            # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            pos_input = pos.unsqueeze(-1).float()
        
        # Generate positional encodings
        pos_encodings = self.mlp(pos_input)
        
        return pos_encodings

class RoPEPosGen(nn.Module):
    """RoPE-style positional generator with learnable parameters.
    
    This version mimics RoPE (Rotary Position Embedding) but with
    learnable frequency parameters.
    """
    
    def __init__(self, d=2048, max_len=1024, base=10000):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.base = base
        
        # Learnable frequency scaling
        self.freq_scale = nn.Parameter(torch.ones(d // 2))
        
        # Pre-compute base frequencies
        self.register_buffer('base_freqs', self._compute_base_freqs())
        
        # Optional MLP for additional transformation
        self.transform = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        
    def _compute_base_freqs(self):
        """Compute base frequencies for RoPE."""
        freqs = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        return freqs
    
    def forward(self, pos):
        """Generate RoPE-style positional encodings.
        
        Args:
            pos (torch.Tensor): Position indices [seq_len] or [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Positional encodings [seq_len, d] or [batch_size, seq_len, d]
        """
        # Handle different input shapes
        if pos.dim() == 1:
            seq_len = pos.size(0)
            batch_size = None
            pos_expanded = pos.unsqueeze(0)  # [1, seq_len]
        else:
            batch_size, seq_len = pos.shape
            pos_expanded = pos  # [batch_size, seq_len]
        
        # Compute scaled frequencies
        freqs = self.base_freqs * self.freq_scale  # [d//2]
        
        # Compute angles
        angles = pos_expanded.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)  # [batch_size, seq_len, d//2]
        
        # Compute sin and cos
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Interleave cos and sin
        pos_encodings = torch.stack([cos_vals, sin_vals], dim=-1)  # [batch_size, seq_len, d//2, 2]
        pos_encodings = pos_encodings.flatten(-2)  # [batch_size, seq_len, d]
        
        # Apply additional transformation
        pos_encodings = self.transform(pos_encodings)
        
        # Remove batch dimension if input was 1D
        if batch_size is None:
            pos_encodings = pos_encodings.squeeze(0)  # [seq_len, d]
        
        return pos_encodings

class LearnablePosGen(nn.Module):
    """Learnable positional embeddings with extrapolation capability."""
    
    def __init__(self, d=2048, max_len=1024):
        super().__init__()
        self.d = d
        self.max_len = max_len
        
        # Learnable position embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(max_len, d) * 0.02)
        
        # Extrapolation network for positions beyond max_len
        self.extrapolate = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d)
        )
        
    def forward(self, pos):
        """Generate positional encodings with extrapolation.
        
        Args:
            pos (torch.Tensor): Position indices [seq_len] or [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Positional encodings [seq_len, d] or [batch_size, seq_len, d]
        """
        # Handle different input shapes
        original_shape = pos.shape
        if pos.dim() == 1:
            pos_flat = pos
            batch_size = None
        else:
            batch_size = pos.size(0)
            pos_flat = pos.flatten()
        
        # Separate positions within and beyond max_len
        within_range = pos_flat < self.max_len
        beyond_range = ~within_range
        
        # Initialize output
        output = torch.zeros(pos_flat.size(0), self.d, device=pos.device, dtype=pos.dtype)
        
        # Handle positions within range
        if within_range.any():
            valid_pos = pos_flat[within_range]
            output[within_range] = self.pos_embeddings[valid_pos]
        
        # Handle positions beyond range with extrapolation
        if beyond_range.any():
            beyond_pos = pos_flat[beyond_range].float().unsqueeze(-1)
            extrapolated = self.extrapolate(beyond_pos)
            output[beyond_range] = extrapolated
        
        # Reshape to original shape
        if batch_size is None:
            output = output.view(original_shape[0], self.d)
        else:
            output = output.view(batch_size, original_shape[1], self.d)
        
        return output

def create_position_indices(seq_len, batch_size=None, device=None):
    """Create position indices for testing.
    
    Args:
        seq_len (int): Sequence length
        batch_size (int, optional): Batch size
        device (torch.device, optional): Device
        
    Returns:
        torch.Tensor: Position indices
    """
    pos = torch.arange(seq_len, device=device)
    if batch_size is not None:
        pos = pos.unsqueeze(0).expand(batch_size, -1)
    return pos

if __name__ == "__main__":
    # Test the modules
    print("Testing PosGen modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 10
    batch_size = 2
    d = 2048
    
    # Test basic PosGen
    print("\n1. Testing basic PosGen:")
    pos_gen = PosGen(d=d).to(device)
    pos_indices = create_position_indices(seq_len, device=device)
    pos_encodings = pos_gen(pos_indices)
    print(f"Input shape: {pos_indices.shape}")
    print(f"Output shape: {pos_encodings.shape}")
    print(f"Expected: [{seq_len}, {d}]")
    
    # Test with batch
    pos_indices_batch = create_position_indices(seq_len, batch_size, device=device)
    pos_encodings_batch = pos_gen(pos_indices_batch)
    print(f"Batch input shape: {pos_indices_batch.shape}")
    print(f"Batch output shape: {pos_encodings_batch.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {d}]")
    
    # Test RoPE-style PosGen
    print("\n2. Testing RoPE-style PosGen:")
    rope_gen = RoPEPosGen(d=d).to(device)
    rope_encodings = rope_gen(pos_indices)
    print(f"RoPE output shape: {rope_encodings.shape}")
    
    # Test Learnable PosGen
    print("\n3. Testing Learnable PosGen:")
    learnable_gen = LearnablePosGen(d=d, max_len=1024).to(device)
    learnable_encodings = learnable_gen(pos_indices)
    print(f"Learnable output shape: {learnable_encodings.shape}")
    
    # Test extrapolation
    long_pos = create_position_indices(1200, device=device)  # Beyond max_len
    extrapolated = learnable_gen(long_pos)
    print(f"Extrapolated shape: {extrapolated.shape}")
    
    print("\nPosGen module tests complete!")