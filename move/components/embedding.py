"""Sparse Embedding Model - Token Embedding Replacement

This module implements a lightweight encoder that replaces traditional token embeddings
with a more efficient sparse representation, inspired by Sentence-T5 and fastText approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

class SparseEmbeddingModel(nn.Module):
    """Sparse embedding model that replaces traditional token embeddings.
    
    This model uses a combination of:
    1. Sparse hash-based embeddings for efficiency
    2. Learned positional-aware token representations
    3. Dynamic vocabulary adaptation
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        sparse_ratio: float = 0.1,
        hash_buckets: int = 10000,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.hash_buckets = hash_buckets
        
        # Core embedding components
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Sparse hash embeddings for rare tokens
        self.hash_embeddings = nn.Embedding(hash_buckets, embed_dim)
        
        # Frequency-based embedding selection
        self.frequency_threshold = nn.Parameter(torch.tensor(100.0))
        
        # Learned sparse gates
        self.sparse_gate = nn.Linear(embed_dim, 1)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.token_embeddings.weight)
        nn.init.xavier_uniform_(self.hash_embeddings.weight)
        
        # Initialize sparse gate to favor dense embeddings initially
        nn.init.constant_(self.sparse_gate.bias, 2.0)
        
    def _hash_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Hash token IDs to bucket indices.
        
        Args:
            token_ids: Token IDs tensor [batch_size, seq_len]
            
        Returns:
            Hash bucket indices [batch_size, seq_len]
        """
        # Simple hash function: (token_id * prime) % hash_buckets
        prime = 31
        return (token_ids * prime) % self.hash_buckets
    
    def _compute_sparsity_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute sparsity mask based on learned gates.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Sparsity mask [batch_size, seq_len, embed_dim]
        """
        # Compute gate values
        gate_logits = self.sparse_gate(embeddings)  # [batch_size, seq_len, 1]
        gate_probs = torch.sigmoid(gate_logits)
        
        # Create sparse mask
        if self.training:
            # During training, use Gumbel-Softmax for differentiable sampling
            noise = torch.rand_like(gate_probs)
            sparse_mask = (gate_probs + 1e-8).log() + (-(-noise.log()).log())
            sparse_mask = torch.sigmoid(sparse_mask / 0.1)  # Temperature = 0.1
        else:
            # During inference, use hard thresholding
            sparse_mask = (gate_probs > 0.5).float()
        
        return sparse_mask
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        token_frequencies: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of sparse embedding model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_frequencies: Optional token frequency information
            
        Returns:
            Tuple of (embeddings, auxiliary_outputs)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get standard token embeddings
        token_embeds = self.token_embeddings(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Get hash-based embeddings for rare tokens
        hash_indices = self._hash_tokens(input_ids)
        hash_embeds = self.hash_embeddings(hash_indices)
        
        # Combine embeddings based on token frequency (if available)
        if token_frequencies is not None:
            # Use frequency-based mixing
            freq_mask = (token_frequencies < self.frequency_threshold).float().unsqueeze(-1)
            combined_embeds = (1 - freq_mask) * token_embeds + freq_mask * hash_embeds
        else:
            # Simple averaging for unknown frequencies
            combined_embeds = 0.7 * token_embeds + 0.3 * hash_embeds
        
        # Apply sparsity
        sparsity_mask = self._compute_sparsity_mask(combined_embeds)
        sparse_embeds = combined_embeds * sparsity_mask
        
        # Normalization and dropout
        output_embeds = self.layer_norm(sparse_embeds)
        output_embeds = self.dropout(output_embeds)
        
        # Auxiliary outputs for monitoring
        aux_outputs = {
            'sparsity_ratio': sparsity_mask.mean(),
            'gate_entropy': self._compute_gate_entropy(sparsity_mask),
            'embedding_norm': output_embeds.norm(dim=-1).mean()
        }
        
        return output_embeds, aux_outputs
    
    def _compute_gate_entropy(self, sparsity_mask: torch.Tensor) -> torch.Tensor:
        """Compute entropy of sparsity gates for regularization.
        
        Args:
            sparsity_mask: Sparsity mask tensor
            
        Returns:
            Gate entropy scalar
        """
        p = sparsity_mask.mean(dim=-1)  # Average over embedding dimension
        p = torch.clamp(p, 1e-8, 1 - 1e-8)  # Avoid log(0)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        return entropy
    
    def get_embedding_stats(self) -> Dict[str, float]:
        """Get embedding statistics for monitoring.
        
        Returns:
            Dictionary of embedding statistics
        """
        with torch.no_grad():
            token_norm = self.token_embeddings.weight.norm(dim=1).mean().item()
            hash_norm = self.hash_embeddings.weight.norm(dim=1).mean().item()
            
            return {
                'token_embedding_norm': token_norm,
                'hash_embedding_norm': hash_norm,
                'frequency_threshold': self.frequency_threshold.item()
            }

class PositionalPredictor(nn.Module):
    """Positional Predictor - Replaces static positional encoding.
    
    This model learns dynamic positional representations that can adapt
    to different sequence lengths and contexts, inspired by RoPE but more flexible.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope_style: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope_style = use_rope_style
        
        # Learned positional parameters
        if use_rope_style:
            # RoPE-style rotary embeddings
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            )
        else:
            # Learned positional embeddings
            self.pos_embeddings = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Context-aware position adjustment
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def _apply_rotary_pos_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary positional embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            seq_len: Sequence length
            
        Returns:
            Tensor with rotary positional embeddings applied
        """
        # Generate position indices
        position_ids = torch.arange(seq_len, device=x.device, dtype=torch.float)
        
        # Compute frequencies
        freqs = torch.outer(position_ids, self.inv_freq)  # [seq_len, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        
        # Reshape for multi-head
        emb = emb.unsqueeze(0).expand(x.shape[0], -1, -1)  # [batch_size, seq_len, head_dim]
        
        # Apply rotation
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Reshape input for rotation
        x_reshaped = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        
        # Apply rotation to each head
        x_rot = torch.zeros_like(x_reshaped)
        for h in range(self.num_heads):
            x_h = x_reshaped[:, :, h, :]  # [batch_size, seq_len, head_dim]
            
            # Split into pairs for rotation
            x1 = x_h[..., ::2]   # Even indices
            x2 = x_h[..., 1::2]  # Odd indices
            
            # Apply rotation
            cos_h = cos_emb[..., :x1.shape[-1]]
            sin_h = sin_emb[..., :x1.shape[-1]]
            
            x_rot[:, :, h, ::2] = x1 * cos_h - x2 * sin_h
            x_rot[:, :, h, 1::2] = x1 * sin_h + x2 * cos_h
        
        return x_rot.view(x.shape)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass of positional predictor.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Embeddings with positional information [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        if self.use_rope_style:
            # Apply RoPE-style rotary embeddings
            pos_embeddings = self._apply_rotary_pos_emb(embeddings, seq_len)
        else:
            # Use learned positional embeddings
            pos_emb = self.pos_embeddings[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            pos_embeddings = embeddings + pos_emb
        
        # Context-aware position adjustment
        context_features = self.context_proj(embeddings.mean(dim=1, keepdim=True))
        context_features = context_features.expand(-1, seq_len, -1)
        
        # Combine with positional information
        combined = pos_embeddings + context_features
        
        # Apply MLP for final position encoding
        output = self.position_mlp(combined)
        output = self.layer_norm(output + pos_embeddings)  # Residual connection
        
        return output