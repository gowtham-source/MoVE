"""Output Decoder - Task-specific Output Generation

This module implements task-specific decoders that handle the final transformation
from hidden states to output tokens, supporting both classification and generation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import math

class TaskSpecificHead(nn.Module):
    """Base class for task-specific output heads."""
    
    def __init__(self, embed_dim: int, task_type: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_type = task_type
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class LanguageModelingHead(TaskSpecificHead):
    """Language modeling head for next-token prediction."""
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        bias: bool = False,
        tie_weights: bool = True,
        embedding_layer: Optional[nn.Module] = None
    ):
        super().__init__(embed_dim, "language_modeling")
        
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        if tie_weights and embedding_layer is not None:
            # Tie weights with embedding layer
            self.weight = embedding_layer.weight
            self.bias = nn.Parameter(torch.zeros(vocab_size)) if bias else None
        else:
            # Independent projection layer
            self.projection = nn.Linear(embed_dim, vocab_size, bias=bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embed_dim]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        if hasattr(self, 'projection'):
            return self.projection(hidden_states)
        else:
            # Manual computation with tied weights
            logits = F.linear(hidden_states, self.weight, self.bias)
            return logits

class ClassificationHead(TaskSpecificHead):
    """Classification head for sequence classification tasks."""
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        pooling_strategy: str = "cls"  # "cls", "mean", "max", "attention"
    ):
        super().__init__(embed_dim, "classification")
        
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        
        # Pooling components
        if pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1)
            )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def _pool_sequence(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence representations.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, embed_dim]
        """
        if self.pooling_strategy == "cls":
            # Use first token (CLS token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask
                pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)
            return pooled
        
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                # Masked max pooling
                mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask + (1 - mask) * (-1e9)
                pooled = masked_hidden.max(dim=1)[0]
            else:
                pooled = hidden_states.max(dim=1)[0]
            return pooled
        
        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            attention_weights = self.attention_pool(hidden_states)  # [batch_size, seq_len, 1]
            
            if attention_mask is not None:
                # Apply attention mask
                attention_weights = attention_weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
            
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled = (hidden_states * attention_weights).sum(dim=1)
            return pooled
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for classification.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Pool sequence representation
        pooled = self._pool_sequence(hidden_states, attention_mask)
        
        # Apply classifier
        logits = self.classifier(pooled)
        
        return logits

class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that can switch between different output modes.
    
    This decoder can dynamically adapt its output strategy based on the task
    and input characteristics.
    """
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        num_classes: Optional[int] = None,
        adaptive_threshold: float = 0.5,
        use_mixture: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.adaptive_threshold = adaptive_threshold
        self.use_mixture = use_mixture
        
        # Language modeling head
        self.lm_head = LanguageModelingHead(embed_dim, vocab_size)
        
        # Classification head (if specified)
        if num_classes is not None:
            self.cls_head = ClassificationHead(embed_dim, num_classes)
        
        # Task prediction head
        self.task_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),  # [generation, classification]
            nn.Softmax(dim=-1)
        )
        
        # Mixture weights (if using mixture)
        if use_mixture:
            self.mixture_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # [lm, cls]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_type: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of adaptive decoder.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embed_dim]
            task_type: Optional explicit task type
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing outputs for different tasks
        """
        outputs = {}
        
        # Always compute language modeling output
        lm_logits = self.lm_head(hidden_states)
        outputs['lm_logits'] = lm_logits
        
        # Compute classification output if available
        if hasattr(self, 'cls_head'):
            cls_logits = self.cls_head(hidden_states, attention_mask)
            outputs['cls_logits'] = cls_logits
        
        # Predict task type if not specified
        if task_type is None:
            # Use pooled representation for task prediction
            pooled_repr = hidden_states.mean(dim=1)  # [batch_size, embed_dim]
            task_probs = self.task_predictor(pooled_repr)  # [batch_size, 2]
            outputs['task_probs'] = task_probs
            
            # Determine primary task
            primary_task_idx = task_probs.argmax(dim=-1)  # [batch_size]
            outputs['predicted_task'] = primary_task_idx
        
        # Compute mixture output if enabled
        if self.use_mixture and hasattr(self, 'cls_head'):
            # Normalize mixture weights
            mix_weights = F.softmax(self.mixture_weights, dim=0)
            
            # For mixture, we need to align dimensions
            # This is a simplified approach - in practice, you'd need more sophisticated mixing
            outputs['mixture_weights'] = mix_weights
        
        return outputs

class OutputDecoder(nn.Module):
    """Main Output Decoder that coordinates different output strategies.
    
    This is the primary interface for the output decoding component of MoVE.
    """
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        num_classes: Optional[int] = None,
        decoder_type: str = "adaptive",  # "lm", "classification", "adaptive"
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        tie_weights: bool = True,
        embedding_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_type = decoder_type
        
        # Pre-decoder normalization
        if use_layer_norm:
            self.pre_norm = nn.LayerNorm(embed_dim)
        else:
            self.pre_norm = nn.Identity()
        
        # Decoder selection
        if decoder_type == "lm":
            self.decoder = LanguageModelingHead(
                embed_dim, vocab_size, tie_weights=tie_weights, embedding_layer=embedding_layer
            )
        elif decoder_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification decoder")
            self.decoder = ClassificationHead(embed_dim, num_classes, dropout)
        elif decoder_type == "adaptive":
            self.decoder = AdaptiveDecoder(embed_dim, vocab_size, num_classes)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of output decoder.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            task_type: Optional task type specification
            return_dict: Whether to return dictionary output
            
        Returns:
            Output logits or dictionary of outputs
        """
        # Pre-processing
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Decode based on decoder type
        if self.decoder_type == "lm":
            logits = self.decoder(hidden_states)
            if return_dict:
                return {'logits': logits}
            return logits
        
        elif self.decoder_type == "classification":
            logits = self.decoder(hidden_states, attention_mask)
            if return_dict:
                return {'logits': logits}
            return logits
        
        elif self.decoder_type == "adaptive":
            outputs = self.decoder(hidden_states, task_type, attention_mask)
            if return_dict:
                return outputs
            # Return primary output based on task
            if task_type == "classification" and 'cls_logits' in outputs:
                return outputs['cls_logits']
            return outputs['lm_logits']
        
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
    
    def generate(
        self,
        hidden_states: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate tokens using the decoder.
        
        Args:
            hidden_states: Initial hidden states [batch_size, seq_len, embed_dim]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        if self.decoder_type not in ["lm", "adaptive"]:
            raise ValueError("Generation only supported for language modeling decoders")
        
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Initialize with last hidden state
        current_hidden = hidden_states[:, -1:, :]  # [batch_size, 1, embed_dim]
        generated_ids = []
        
        for _ in range(max_length):
            # Get logits for current position
            if self.decoder_type == "adaptive":
                outputs = self.decoder(current_hidden, task_type="generation")
                logits = outputs['lm_logits'][:, -1, :]  # [batch_size, vocab_size]
            else:
                logits = self.decoder(current_hidden)[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or select next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            
            # Update hidden state (simplified - in practice, you'd need the full model)
            # This is a placeholder - actual implementation would require the full forward pass
            break
        
        if generated_ids:
            return torch.cat(generated_ids, dim=1)
        else:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)
    
    def get_decoder_stats(self) -> Dict[str, Any]:
        """Get decoder statistics for monitoring.
        
        Returns:
            Dictionary of decoder statistics
        """
        stats = {
            'decoder_type': self.decoder_type,
            'embed_dim': self.embed_dim,
            'vocab_size': self.vocab_size
        }
        
        # Add decoder-specific stats
        if hasattr(self.decoder, 'get_stats'):
            stats.update(self.decoder.get_stats())
        
        return stats