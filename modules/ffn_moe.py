#!/usr/bin/env python3
"""
Expert FFN Ensemble (MoE) Module for MoVE

This module provides a Mixture-of-Experts feedforward network
with router loss and reconstruction loss for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNMoE(nn.Module):
    """Mixture-of-Experts FFN module.
    
    Args:
        d (int): Hidden dimension (default: 2048)
        experts (int): Number of experts (default: 8)
        topk (int): Number of top experts to use (default: 2)
        expert_dim (int): Expert hidden dimension (default: None, uses 4*d)
    """
    
    def __init__(self, d=2048, experts=8, topk=2, expert_dim=None):
        super().__init__()
        self.d = d
        self.num_experts = experts
        self.topk = topk
        self.expert_dim = expert_dim or 4 * d
        
        # Router/gate network
        self.gate = nn.Linear(d, experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, self.expert_dim),
                nn.GELU(),
                nn.Linear(self.expert_dim, d)
            ) for _ in range(experts)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Load balancing
        self.load_balancing_loss_coef = 0.01
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, return_router_loss=False):
        """Forward pass through MoE.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d]
            return_router_loss (bool): Whether to return router loss
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d]
            torch.Tensor (optional): Router loss if return_router_loss=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # Flatten for expert routing
        x_flat = x_norm.view(-1, self.d)  # [batch_size * seq_len, d]
        
        # Compute router scores
        router_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        # topk_probs: [batch_size * seq_len, topk]
        # topk_indices: [batch_size * seq_len, topk]
        
        # Normalize top-k probabilities
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_tokens = x_flat[expert_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Get weights for this expert
                expert_weights = torch.zeros(expert_mask.sum(), device=x.device)
                for i, token_idx in enumerate(torch.where(expert_mask)[0]):
                    # Find which position in topk this expert appears
                    expert_positions = (topk_indices[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(expert_positions) > 0:
                        expert_weights[i] = topk_probs[token_idx, expert_positions[0]]
                
                # Add weighted expert output
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, self.d)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Residual connection
        final_output = x + output
        
        if return_router_loss:
            # Compute load balancing loss
            router_loss = self._compute_router_loss(router_probs, topk_indices)
            return final_output, router_loss
        
        return final_output
    
    def _compute_router_loss(self, router_probs, topk_indices):
        """Compute load balancing loss for router.
        
        Args:
            router_probs (torch.Tensor): Router probabilities [batch_size * seq_len, num_experts]
            topk_indices (torch.Tensor): Top-k expert indices [batch_size * seq_len, topk]
            
        Returns:
            torch.Tensor: Router loss scalar
        """
        # Compute expert usage frequency
        num_tokens = router_probs.size(0)
        expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
        
        for expert_idx in range(self.num_experts):
            expert_usage[expert_idx] = (topk_indices == expert_idx).float().sum()
        
        # Normalize usage
        expert_usage = expert_usage / (num_tokens * self.topk)
        
        # Compute average router probability for each expert
        avg_router_probs = router_probs.mean(dim=0)
        
        # Load balancing loss: encourage uniform distribution
        load_loss = (expert_usage * avg_router_probs).sum() * self.num_experts
        
        return self.load_balancing_loss_coef * load_loss

class AdaptiveFFNMoE(nn.Module):
    """Adaptive MoE with dynamic expert selection."""
    
    def __init__(self, d=2048, experts=8, topk=2, expert_dim=None, adaptive_topk=True):
        super().__init__()
        self.d = d
        self.num_experts = experts
        self.base_topk = topk
        self.expert_dim = expert_dim or 4 * d
        self.adaptive_topk = adaptive_topk
        
        # Router with adaptive top-k prediction
        self.gate = nn.Linear(d, experts)
        if adaptive_topk:
            self.topk_predictor = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(),
                nn.Linear(64, topk),
                nn.Sigmoid()
            )
        
        # Experts with different capacities
        self.experts = nn.ModuleList()
        for i in range(experts):
            # Vary expert capacity
            capacity_factor = 0.5 + (i / experts)  # 0.5 to 1.5
            expert_hidden = int(self.expert_dim * capacity_factor)
            
            expert = nn.Sequential(
                nn.Linear(d, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, d)
            )
            self.experts.append(expert)
        
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
        if self.adaptive_topk:
            for layer in self.topk_predictor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, return_router_loss=False):
        """Forward pass with adaptive expert selection.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d]
            return_router_loss (bool): Whether to return router loss
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d]
            torch.Tensor (optional): Router loss if return_router_loss=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer norm
        x_norm = self.norm(x)
        x_flat = x_norm.view(-1, self.d)
        
        # Compute router scores
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Adaptive top-k selection
        if self.adaptive_topk:
            topk_weights = self.topk_predictor(x_flat)  # [batch_size * seq_len, base_topk]
            # Use weights to determine effective top-k
            effective_topk = torch.clamp(torch.round(topk_weights.sum(dim=-1)), min=1, max=self.base_topk).int()
        else:
            effective_topk = torch.full((x_flat.size(0),), self.base_topk, device=x.device)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process tokens with different top-k values
        for k in range(1, self.base_topk + 1):
            mask = (effective_topk == k)
            if mask.any():
                # Select top-k experts for these tokens
                topk_probs, topk_indices = torch.topk(router_probs[mask], k, dim=-1)
                topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Process each expert
                for expert_idx in range(self.num_experts):
                    expert_mask = (topk_indices == expert_idx).any(dim=-1)
                    if expert_mask.any():
                        # Get tokens for this expert
                        token_indices = torch.where(mask)[0][expert_mask]
                        expert_tokens = x_flat[token_indices]
                        
                        # Process through expert
                        expert_output = self.experts[expert_idx](expert_tokens)
                        
                        # Get weights
                        expert_weights = torch.zeros(expert_mask.sum(), device=x.device)
                        for i, global_idx in enumerate(token_indices):
                            local_idx = torch.where(torch.where(mask)[0] == global_idx)[0][0]
                            expert_positions = (topk_indices[local_idx] == expert_idx).nonzero(as_tuple=True)[0]
                            if len(expert_positions) > 0:
                                expert_weights[i] = topk_probs[local_idx, expert_positions[0]]
                        
                        # Add weighted output
                        output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
        
        # Reshape and apply dropout
        output = output.view(batch_size, seq_len, self.d)
        output = self.dropout(output)
        
        # Residual connection
        final_output = x + output
        
        if return_router_loss:
            # Simplified router loss for adaptive case
            avg_probs = router_probs.mean(dim=0)
            router_loss = 0.01 * (avg_probs * torch.log(avg_probs + 1e-8)).sum()
            return final_output, router_loss
        
        return final_output

def compute_reconstruction_loss(original, reconstructed, reduction='mean'):
    """Compute reconstruction loss between original and reconstructed tensors.
    
    Args:
        original (torch.Tensor): Original tensor
        reconstructed (torch.Tensor): Reconstructed tensor
        reduction (str): Reduction method ('mean', 'sum', 'none')
        
    Returns:
        torch.Tensor: Reconstruction loss
    """
    mse_loss = F.mse_loss(reconstructed, original, reduction=reduction)
    return mse_loss

if __name__ == "__main__":
    # Test the MoE modules
    print("Testing FFNMoE modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 10
    d = 2048
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d, device=device)
    print(f"Input shape: {x.shape}")
    
    # Test basic FFNMoE
    print("\n1. Testing basic FFNMoE:")
    moe = FFNMoE(d=d, experts=8, topk=2).to(device)
    output1, router_loss1 = moe(x, return_router_loss=True)
    print(f"Output shape: {output1.shape}")
    print(f"Router loss: {router_loss1.item():.6f}")
    
    # Test reconstruction loss
    recon_loss = compute_reconstruction_loss(x, output1)
    print(f"Reconstruction loss: {recon_loss.item():.6f}")
    
    # Test AdaptiveFFNMoE
    print("\n2. Testing AdaptiveFFNMoE:")
    adaptive_moe = AdaptiveFFNMoE(d=d, experts=8, topk=3, adaptive_topk=True).to(device)
    output2, router_loss2 = adaptive_moe(x, return_router_loss=True)
    print(f"Adaptive output shape: {output2.shape}")
    print(f"Adaptive router loss: {router_loss2.item():.6f}")
    
    # Test parameter counts
    print("\n3. Parameter counts:")
    print(f"Basic FFNMoE: {sum(p.numel() for p in moe.parameters()):,}")
    print(f"Adaptive FFNMoE: {sum(p.numel() for p in adaptive_moe.parameters()):,}")
    
    # Test expert utilization
    print("\n4. Testing expert utilization:")
    with torch.no_grad():
        x_flat = x.view(-1, d)
        router_logits = moe.gate(moe.norm(x_flat))
        router_probs = F.softmax(router_logits, dim=-1)
        expert_usage = router_probs.mean(dim=0)
        print(f"Expert usage distribution: {expert_usage.cpu().numpy()}")
    
    print("\nFFNMoE module tests complete!")