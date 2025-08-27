"""Attention Approximator - Graph-based Attention Replacement

This module implements a Graph Neural Network-based attention mechanism that
approximates traditional multi-head attention with improved efficiency and interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer.
    
    Implements attention as message passing on a dynamic graph where
    nodes are tokens and edges represent attention relationships.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_threshold: float = 0.1,
        use_edge_features: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.edge_threshold = edge_threshold
        self.use_edge_features = use_edge_features
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Node feature transformations (Q, K, V)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge feature computation
        if use_edge_features:
            self.edge_proj = nn.Linear(embed_dim * 2, num_heads)
        
        # Graph message passing
        self.message_mlp = nn.Sequential(
            nn.Linear(embed_dim + (num_heads if use_edge_features else 0), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Normalization and regularization
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _compute_edge_weights(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute edge weights and adjacency matrix.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (edge_weights, adjacency_matrix)
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Compute edge weights (attention probabilities)
        edge_weights = F.softmax(scores, dim=2)  # [batch_size, seq_len, seq_len, num_heads]
        
        # Create sparse adjacency matrix based on threshold
        adjacency = (edge_weights.mean(dim=-1) > self.edge_threshold).float()
        
        return edge_weights, adjacency
    
    def _message_passing(
        self,
        node_features: torch.Tensor,
        edge_weights: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Perform message passing on the attention graph.
        
        Args:
            node_features: Node features [batch_size, seq_len, embed_dim]
            edge_weights: Edge weights [batch_size, seq_len, seq_len, num_heads]
            adjacency: Adjacency matrix [batch_size, seq_len, seq_len]
            
        Returns:
            Updated node features [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = node_features.shape
        
        # Prepare messages
        messages = torch.zeros_like(node_features)
        
        for i in range(seq_len):
            # Find neighbors for node i
            neighbors = adjacency[:, i, :].bool()  # [batch_size, seq_len]
            
            if neighbors.any():
                # Get neighbor features
                neighbor_features = node_features  # [batch_size, seq_len, embed_dim]
                
                # Weight by edge importance
                edge_weights_i = edge_weights[:, i, :, :].mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                weighted_features = neighbor_features * edge_weights_i
                
                # Aggregate messages (masked sum)
                neighbor_mask = neighbors.unsqueeze(-1).float()
                aggregated = (weighted_features * neighbor_mask).sum(dim=1, keepdim=True)
                
                # Normalize by number of neighbors
                num_neighbors = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)
                messages[:, i:i+1, :] = aggregated / num_neighbors
        
        # Apply message MLP
        if self.use_edge_features:
            # Concatenate edge features (simplified)
            edge_features = edge_weights.mean(dim=2)  # [batch_size, seq_len, num_heads]
            combined_features = torch.cat([node_features, edge_features], dim=-1)
        else:
            combined_features = node_features
        
        updated_features = self.message_mlp(combined_features)
        
        return updated_features + messages  # Residual connection
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of graph attention layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output_states, attention_info)
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute edge weights and adjacency
        edge_weights, adjacency = self._compute_edge_weights(q, k, attention_mask)
        
        # Perform message passing
        updated_features = self._message_passing(hidden_states, edge_weights, adjacency)
        
        # Apply value transformation with attention
        v_flat = v.view(batch_size, seq_len, embed_dim)
        attended_values = torch.einsum('bij,bjd->bid', edge_weights.mean(dim=-1), v_flat)
        
        # Combine updated features with attended values
        combined_output = updated_features + attended_values
        
        # Final output projection
        output = self.out_proj(combined_output)
        output = self.dropout(output)
        
        # Attention info for analysis
        attention_info = {
            'edge_weights': edge_weights,
            'adjacency': adjacency,
            'sparsity': (adjacency == 0).float().mean(),
            'avg_degree': adjacency.sum(dim=-1).mean()
        }
        
        return output, attention_info

class AttentionApproximator(nn.Module):
    """Multi-layer Graph-based Attention Approximator.
    
    Stacks multiple graph attention layers to approximate the behavior
    of traditional multi-head attention with improved efficiency.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_threshold: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Stack of graph attention layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                edge_threshold=edge_threshold,
                use_edge_features=(i == 0)  # Only first layer uses edge features
            )
            for i in range(num_layers)
        ])
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None
        
        # Global attention pooling for sequence representation
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Forward pass of attention approximator.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            return_attention_info: Whether to return attention analysis info
            
        Returns:
            Tuple of (output_states, optional_attention_info)
        """
        current_states = hidden_states
        all_attention_info = [] if return_attention_info else None
        
        # Pass through each graph attention layer
        for i, layer in enumerate(self.layers):
            # Apply layer
            layer_output, attention_info = layer(current_states, attention_mask)
            
            # Residual connection
            if self.use_residual:
                layer_output = layer_output + current_states
            
            # Layer normalization
            if self.layer_norms is not None:
                layer_output = self.layer_norms[i](layer_output)
            
            current_states = layer_output
            
            if return_attention_info:
                all_attention_info.append(attention_info)
        
        # Compute global attention weights for sequence pooling
        global_weights = self.global_pool(current_states)  # [batch_size, seq_len, 1]
        global_weights = F.softmax(global_weights, dim=1)
        
        # Create final attention info
        final_attention_info = None
        if return_attention_info:
            final_attention_info = {
                'layer_info': all_attention_info,
                'global_weights': global_weights,
                'output_norm': current_states.norm(dim=-1).mean()
            }
        
        return current_states, final_attention_info
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get attention statistics for monitoring.
        
        Returns:
            Dictionary of attention statistics
        """
        stats = {}
        
        for i, layer in enumerate(self.layers):
            # Get layer-specific stats
            layer_stats = {
                f'layer_{i}_edge_threshold': layer.edge_threshold,
                f'layer_{i}_num_heads': layer.num_heads
            }
            stats.update(layer_stats)
        
        return stats

class LowRankAttention(nn.Module):
    """Low-rank attention approximation as an alternative to graph attention.
    
    Uses low-rank matrix factorization to approximate attention matrices
    with reduced computational complexity.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        rank: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank = rank
        
        # Low-rank projections
        self.q_low = nn.Linear(embed_dim, rank * num_heads)
        self.k_low = nn.Linear(embed_dim, rank * num_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = 1.0 / math.sqrt(rank)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of low-rank attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output states [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Low-rank Q and K projections
        q_low = self.q_low(hidden_states).view(batch_size, seq_len, self.num_heads, self.rank)
        k_low = self.k_low(hidden_states).view(batch_size, seq_len, self.num_heads, self.rank)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute low-rank attention scores
        scores = torch.einsum('bihd,bjhd->bijh', q_low, k_low) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=2)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.einsum('bijh,bjhd->bihd', attn_weights, v)
        output = output.contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output