"""Expert FFN Ensemble - Mixture-of-Experts Implementation

This module implements a Mixture-of-Experts system that replaces traditional
feedforward layers with domain-specialized expert networks, inspired by LLaMA 4's
MoE architecture with routing, load balancing, and shared experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import random

class ExpertFFN(nn.Module):
    """Individual Expert Feedforward Network.
    
    Each expert is a specialized FFN that can learn domain-specific patterns.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        expert_id: int = 0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.expert_id = expert_id
        
        # Expert-specific layers
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Expert specialization parameters
        self.specialization_gate = nn.Parameter(torch.ones(ffn_dim))
        
        # Initialize weights with expert-specific variance
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with expert-specific patterns."""
        # Use different initialization for each expert
        init_std = 0.02 * (1 + 0.1 * self.expert_id)
        nn.init.normal_(self.fc1.weight, std=init_std)
        nn.init.normal_(self.fc2.weight, std=init_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of expert FFN.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        # First linear layer
        hidden = self.fc1(x)
        
        # Apply specialization gate
        hidden = hidden * self.specialization_gate.unsqueeze(0).unsqueeze(0)
        
        # Activation and dropout
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Second linear layer
        output = self.fc2(hidden)
        
        return output

class RouterNetwork(nn.Module):
    """Router Network for Expert Selection.
    
    Implements the routing mechanism that decides which experts to activate
    for each token, with load balancing and noise injection.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 2,
        router_bias: bool = True,
        noise_std: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight
        
        # Router projection
        self.router = nn.Linear(embed_dim, num_experts, bias=router_bias)
        
        # Expert capacity tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Add noise to router logits for exploration.
        
        Args:
            logits: Router logits [batch_size, seq_len, num_experts]
            
        Returns:
            Noisy logits
        """
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            return logits + noise
        return logits
    
    def _compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage.
        
        Args:
            router_probs: Router probabilities [batch_size, seq_len, num_experts]
            
        Returns:
            Load balance loss scalar
        """
        # Compute expert usage frequencies
        expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # Target uniform distribution
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # KL divergence loss
        kl_loss = F.kl_div(
            expert_usage.log(),
            target_usage,
            reduction='batchmean'
        )
        
        return kl_loss * self.load_balance_weight
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_capacity: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of router network.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, embed_dim]
            expert_capacity: Maximum tokens per expert (for load balancing)
            
        Returns:
            Tuple of (expert_indices, expert_weights, combine_weights, router_info)
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # Add noise for exploration
        noisy_logits = self._add_noise(router_logits)
        
        # Compute router probabilities
        router_probs = F.softmax(noisy_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Expert capacity enforcement
        if expert_capacity is not None and self.training:
            # Implement capacity-based routing
            expert_mask = self._enforce_expert_capacity(
                top_k_indices, expert_capacity, batch_size, seq_len
            )
            top_k_probs = top_k_probs * expert_mask
            
            # Renormalize after masking
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute load balance loss
        load_balance_loss = self._compute_load_balance_loss(router_probs)
        
        # Router info for monitoring
        router_info = {
            'load_balance_loss': load_balance_loss,
            'router_entropy': self._compute_router_entropy(router_probs),
            'expert_usage': router_probs.mean(dim=[0, 1]),
            'top_k_confidence': top_k_probs.max(dim=-1)[0].mean()
        }
        
        return top_k_indices, top_k_probs, router_probs, router_info
    
    def _enforce_expert_capacity(
        self, 
        expert_indices: torch.Tensor, 
        capacity: int, 
        batch_size: int, 
        seq_len: int
    ) -> torch.Tensor:
        """Enforce expert capacity constraints.
        
        Args:
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            capacity: Maximum tokens per expert
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Capacity mask [batch_size, seq_len, top_k]
        """
        device = expert_indices.device
        mask = torch.ones_like(expert_indices, dtype=torch.float, device=device)
        
        # Track expert token counts
        expert_counts = torch.zeros(self.num_experts, device=device)
        
        # Process tokens sequentially to enforce capacity
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.top_k):
                    expert_id = expert_indices[b, s, k].item()
                    if expert_counts[expert_id] >= capacity:
                        mask[b, s, k] = 0.0
                    else:
                        expert_counts[expert_id] += 1
        
        return mask
    
    def _compute_router_entropy(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of router probabilities.
        
        Args:
            router_probs: Router probabilities [batch_size, seq_len, num_experts]
            
        Returns:
            Router entropy scalar
        """
        # Avoid log(0) by adding small epsilon
        log_probs = (router_probs + 1e-8).log()
        entropy = -(router_probs * log_probs).sum(dim=-1).mean()
        return entropy

class SharedExpert(nn.Module):
    """Shared Expert that processes all tokens.
    
    Acts as a stabilizing fallback and ensures consistent processing
    for all tokens regardless of routing decisions.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ffn = ExpertFFN(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            expert_id=-1  # Special ID for shared expert
        )
        
        # Shared expert weight (learnable)
        self.shared_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of shared expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Weighted output tensor [batch_size, seq_len, embed_dim]
        """
        output = self.ffn(x)
        return output * torch.sigmoid(self.shared_weight)

class ExpertFFNEnsemble(nn.Module):
    """Complete Mixture-of-Experts Ensemble.
    
    Combines multiple expert FFNs with routing, load balancing,
    and a shared expert for stable performance.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_shared_expert: bool = True,
        expert_capacity_factor: float = 1.25,
        load_balance_weight: float = 0.01,
        router_noise_std: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.expert_capacity_factor = expert_capacity_factor
        
        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertFFN(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                expert_id=i
            )
            for i in range(num_experts)
        ])
        
        # Router network
        self.router = RouterNetwork(
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=router_noise_std,
            load_balance_weight=load_balance_weight
        )
        
        # Shared expert (optional)
        if use_shared_expert:
            self.shared_expert = SharedExpert(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim // 2,  # Smaller shared expert
                dropout=dropout
            )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_router_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Forward pass of MoE ensemble.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, embed_dim]
            return_router_info: Whether to return routing information
            
        Returns:
            Tuple of (output_states, optional_router_info)
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Compute expert capacity
        total_tokens = batch_size * seq_len
        expert_capacity = int(total_tokens * self.expert_capacity_factor / self.num_experts)
        
        # Route tokens to experts
        expert_indices, expert_weights, router_probs, router_info = self.router(
            hidden_states, expert_capacity
        )
        
        # Initialize output
        moe_output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_id).any(dim=-1)  # [batch_size, seq_len]
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_tokens = hidden_states[expert_mask]  # [num_tokens, embed_dim]
                
                if expert_tokens.numel() > 0:
                    # Process through expert
                    expert_output = self.experts[expert_id](expert_tokens.unsqueeze(1))
                    expert_output = expert_output.squeeze(1)
                    
                    # Get weights for this expert
                    expert_weight_mask = (expert_indices == expert_id).float()
                    expert_token_weights = (expert_weights * expert_weight_mask).sum(dim=-1)[expert_mask]
                    
                    # Weight and accumulate output
                    weighted_output = expert_output * expert_token_weights.unsqueeze(-1)
                    moe_output[expert_mask] += weighted_output
        
        # Add shared expert output
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            moe_output = moe_output + shared_output
        
        # Normalize output
        moe_output = self.output_norm(moe_output)
        
        # Prepare router info
        final_router_info = None
        if return_router_info:
            final_router_info = {
                **router_info,
                'expert_capacity': expert_capacity,
                'total_tokens': total_tokens,
                'output_norm': moe_output.norm(dim=-1).mean()
            }
        
        return moe_output, final_router_info
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get expert statistics for monitoring.
        
        Returns:
            Dictionary of expert statistics
        """
        stats = {
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'expert_capacity_factor': self.expert_capacity_factor
        }
        
        # Add expert-specific stats
        for i, expert in enumerate(self.experts):
            expert_norm = expert.fc1.weight.norm().item()
            stats[f'expert_{i}_weight_norm'] = expert_norm
        
        return stats