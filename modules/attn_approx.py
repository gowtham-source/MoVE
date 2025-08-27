#!/usr/bin/env python3
"""
Attention Approximator Module for MoVE

This module provides low-rank attention approximation that can be trained
to match layer-1 hidden states from TinyLlama.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttnApprox(nn.Module):
    """Low-rank attention approximator.
    
    Args:
        d (int): Hidden dimension (default: 2048)
        k (int): Low-rank dimension (default: 64)
        num_heads (int): Number of attention heads (default: 32)
    """
    
    def __init__(self, d=2048, k=64, num_heads=32):
        super().__init__()
        self.d = d
        self.k = k
        self.num_heads = num_heads
        self.head_dim = k // num_heads
        
        assert k % num_heads == 0, f"k ({k}) must be divisible by num_heads ({num_heads})"
        
        # Low-rank QKV projection
        self.qkv = nn.Linear(d, 3 * k)
        self.out = nn.Linear(k, d)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x, mask=None):
        """Forward pass with low-rank attention.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d]
            mask (torch.Tensor, optional): Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer norm first (pre-norm)
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm)  # [batch_size, seq_len, 3*k]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch_size, seq_len, k]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores + mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.k
        )
        
        # Final output projection
        output = self.out(attn_output)
        
        # Residual connection
        return x + output

class GraphAttnApprox(nn.Module):
    """Graph-based attention approximator using sparse connections."""
    
    def __init__(self, d=2048, k=64, num_heads=32, sparsity=0.1):
        super().__init__()
        self.d = d
        self.k = k
        self.num_heads = num_heads
        self.head_dim = k // num_heads
        self.sparsity = sparsity
        
        # Graph construction network
        self.graph_net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Attention components
        self.qkv = nn.Linear(d, 3 * k)
        self.out = nn.Linear(k, d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.graph_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out.bias)
    
    def _construct_sparse_graph(self, x):
        """Construct sparse attention graph.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d]
            
        Returns:
            torch.Tensor: Sparse attention mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute pairwise similarities
        similarities = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    # Compute similarity between positions i and j
                    diff = x[:, i] - x[:, j]  # [batch_size, d]
                    sim = self.graph_net(diff).squeeze(-1)  # [batch_size]
                    similarities[:, i, j] = sim
        
        # Keep only top-k connections (sparse graph)
        k_sparse = max(1, int(seq_len * self.sparsity))
        
        # Get top-k similarities for each position
        _, top_indices = torch.topk(similarities, k_sparse, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.full_like(similarities, float('-inf'))
        for b in range(batch_size):
            for i in range(seq_len):
                sparse_mask[b, i, top_indices[b, i]] = 0.0
        
        return sparse_mask
    
    def forward(self, x, mask=None):
        """Forward pass with graph-based sparse attention.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d]
            mask (torch.Tensor, optional): Additional attention mask
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # Construct sparse graph
        graph_mask = self._construct_sparse_graph(x_norm)
        
        # Combine with provided mask
        if mask is not None:
            graph_mask = graph_mask + mask
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply graph mask
        if graph_mask.dim() == 3:
            graph_mask = graph_mask.unsqueeze(1)  # Add head dimension
        scores = scores + graph_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.k
        )
        output = self.out(attn_output)
        
        # Residual connection
        return x + output

class EfficientAttnApprox(nn.Module):
    """Efficient attention approximator using linear attention."""
    
    def __init__(self, d=2048, k=64, num_heads=32):
        super().__init__()
        self.d = d
        self.k = k
        self.num_heads = num_heads
        self.head_dim = k // num_heads
        
        # Feature mapping for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU()
        )
        
        self.qkv = nn.Linear(d, 3 * k)
        self.out = nn.Linear(k, d)
        self.norm = nn.LayerNorm(d)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x, mask=None):
        """Forward pass with linear attention.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d]
            mask (torch.Tensor, optional): Attention mask (not used in linear attention)
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature mapping for linear attention
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Linear attention: O(n) complexity
        # Compute K^T V first: [batch_size, num_heads, head_dim, head_dim]
        kv = torch.matmul(k.transpose(-2, -1), v)
        
        # Then Q (K^T V): [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(q, kv)
        
        # Normalize by sum of keys
        k_sum = k.sum(dim=-2, keepdim=True)  # [batch_size, num_heads, 1, head_dim]
        q_k_sum = torch.matmul(q, k_sum.transpose(-2, -1))  # [batch_size, num_heads, seq_len, 1]
        attn_output = attn_output / (q_k_sum + 1e-8)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.k
        )
        output = self.out(attn_output)
        
        # Residual connection
        return x + output

def create_causal_mask(seq_len, device=None):
    """Create causal attention mask.
    
    Args:
        seq_len (int): Sequence length
        device (torch.device, optional): Device
        
    Returns:
        torch.Tensor: Causal mask [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

if __name__ == "__main__":
    # Test the attention modules
    print("Testing AttnApprox modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 10
    d = 2048
    k = 64
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d, device=device)
    causal_mask = create_causal_mask(seq_len, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    
    # Test basic AttnApprox
    print("\n1. Testing basic AttnApprox:")
    attn_approx = AttnApprox(d=d, k=k).to(device)
    output1 = attn_approx(x, mask=causal_mask)
    print(f"Output shape: {output1.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {d}]")
    
    # Test GraphAttnApprox
    print("\n2. Testing GraphAttnApprox:")
    graph_attn = GraphAttnApprox(d=d, k=k, sparsity=0.3).to(device)
    output2 = graph_attn(x)
    print(f"Graph attention output shape: {output2.shape}")
    
    # Test EfficientAttnApprox
    print("\n3. Testing EfficientAttnApprox:")
    efficient_attn = EfficientAttnApprox(d=d, k=k).to(device)
    output3 = efficient_attn(x)
    print(f"Efficient attention output shape: {output3.shape}")
    
    # Test parameter counts
    print("\n4. Parameter counts:")
    print(f"Basic AttnApprox: {sum(p.numel() for p in attn_approx.parameters()):,}")
    print(f"Graph AttnApprox: {sum(p.numel() for p in graph_attn.parameters()):,}")
    print(f"Efficient AttnApprox: {sum(p.numel() for p in efficient_attn.parameters()):,}")
    
    print("\nAttnApprox module tests complete!")