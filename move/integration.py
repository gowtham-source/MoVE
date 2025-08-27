"""MoVE Integration - Complete System Integration

This module demonstrates how to integrate all MoVE components into a
complete modular vector engine that replicates Llama-style behavior.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Import MoVE components
from .components import (
    SparseEmbeddingModel,
    PositionalPredictor,
    AttentionApproximator,
    ExpertFFNEnsemble,
    OutputDecoder,
    VectorProtocolCoordinator,
    ComponentInterface,
    VectorState,
    ComponentType
)
from .evaluation import MoVEValidator, create_test_dataset

class MoVEModel(nn.Module):
    """Complete Modular Vector Engine Model.
    
    This class integrates all MoVE components into a unified model that
    can replace traditional transformer architectures.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 2048,
        max_seq_length: int = 2048,
        num_attention_heads: int = 32,
        num_experts: int = 8,
        expert_capacity: int = 2,
        dropout: float = 0.1,
        use_sparse_embedding: bool = True,
        use_dynamic_positional: bool = True,
        use_graph_attention: bool = True,
        use_moe_ffn: bool = True,
        use_adaptive_decoder: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Initialize coordinator
        self.coordinator = VectorProtocolCoordinator(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            use_residual_connections=True,
            use_gradient_checkpointing=False,
            component_dropout=dropout
        )
        
        # Initialize components
        self._init_components(
            vocab_size, embed_dim, max_seq_length, num_attention_heads,
            num_experts, expert_capacity, dropout,
            use_sparse_embedding, use_dynamic_positional, use_graph_attention,
            use_moe_ffn, use_adaptive_decoder
        )
        
        # Register components with coordinator
        self._register_components()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("MoVE model initialized successfully")
    
    def _init_components(
        self, vocab_size, embed_dim, max_seq_length, num_attention_heads,
        num_experts, expert_capacity, dropout, use_sparse_embedding,
        use_dynamic_positional, use_graph_attention, use_moe_ffn, use_adaptive_decoder
    ):
        """Initialize all MoVE components."""
        
        # 1. Token Embedding Component
        if use_sparse_embedding:
            self.embedding_component = SparseEmbeddingModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hash_size=vocab_size // 4,
                num_hash_functions=3,
                dropout=dropout
            )
        else:
            # Fallback to standard embedding
            self.embedding_component = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Positional Predictor Component
        if use_dynamic_positional:
            self.positional_component = PositionalPredictor(
                embed_dim=embed_dim,
                max_seq_length=max_seq_length,
                predictor_type="learned",
                use_rope_style=True,
                dropout=dropout
            )
        else:
            # Fallback to standard positional encoding
            self.positional_component = nn.Parameter(
                torch.randn(max_seq_length, embed_dim) * 0.02
            )
        
        # 3. Attention Approximator Component
        if use_graph_attention:
            self.attention_component = AttentionApproximator(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                approximation_method="graph_attention",
                graph_k=8,
                dropout=dropout
            )
        else:
            # Fallback to standard attention
            self.attention_component = nn.MultiheadAttention(
                embed_dim, num_attention_heads, dropout=dropout, batch_first=True
            )
        
        # 4. Expert FFN Ensemble Component
        if use_moe_ffn:
            self.ffn_component = ExpertFFNEnsemble(
                embed_dim=embed_dim,
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                hidden_dim=embed_dim * 4,
                dropout=dropout,
                use_shared_expert=True
            )
        else:
            # Fallback to standard FFN
            self.ffn_component = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )
        
        # 5. Output Decoder Component
        if use_adaptive_decoder:
            self.decoder_component = OutputDecoder(
                embed_dim=embed_dim,
                vocab_size=vocab_size,
                decoder_type="adaptive",
                dropout=dropout,
                tie_weights=True,
                embedding_layer=self.embedding_component if hasattr(self.embedding_component, 'weight') else None
            )
        else:
            # Fallback to standard linear projection
            self.decoder_component = nn.Linear(embed_dim, vocab_size)
    
    def _register_components(self):
        """Register components with the coordinator."""
        self.coordinator.register_component(ComponentType.EMBEDDING, self.embedding_component)
        self.coordinator.register_component(ComponentType.POSITIONAL, self.positional_component)
        self.coordinator.register_component(ComponentType.ATTENTION, self.attention_component)
        self.coordinator.register_component(ComponentType.FFN, self.ffn_component)
        self.coordinator.register_component(ComponentType.DECODER, self.decoder_component)
        
        # Set execution order
        self.coordinator.set_component_order([
            ComponentType.EMBEDDING,
            ComponentType.POSITIONAL,
            ComponentType.ATTENTION,
            ComponentType.FFN,
            ComponentType.DECODER
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete MoVE model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            return_dict: Whether to return dictionary output
            output_hidden_states: Whether to return intermediate states
            
        Returns:
            Model outputs including logits and hidden states
        """
        # Use coordinator to process through all components
        outputs = self.coordinator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=output_hidden_states
        )
        
        # Extract final logits from decoder output
        if 'component_outputs' in outputs and ComponentType.DECODER.value in outputs['component_outputs']:
            decoder_output = outputs['component_outputs'][ComponentType.DECODER.value]
            if isinstance(decoder_output, dict) and 'logits' in decoder_output:
                logits = decoder_output['logits']
            else:
                logits = decoder_output
        else:
            # Fallback: apply decoder to final hidden state
            logits = self.decoder_component(outputs['last_hidden_state'])
        
        # Prepare final output
        final_output = {
            'logits': logits,
            'last_hidden_state': outputs['last_hidden_state'],
            'component_outputs': outputs.get('component_outputs', {}),
            'metadata': outputs.get('metadata', {})
        }
        
        if output_hidden_states and 'hidden_states' in outputs:
            final_output['hidden_states'] = outputs['hidden_states']
        
        if not return_dict:
            return logits
        
        return final_output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the MoVE model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token sequences
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated, return_dict=True)
                logits = outputs['logits'][:, -1, :]  # Get last token logits
                
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
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS tokens
                if eos_token_id is not None:
                    finished = finished | (next_token.squeeze(-1) == eos_token_id)
                    if finished.all():
                        break
        
        return generated
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = {
            'model_config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'max_seq_length': self.max_seq_length
            },
            'component_stats': self.coordinator.get_component_stats(),
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        return stats
    
    def validate_against_baseline(
        self,
        baseline_model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        num_test_samples: int = 50,
        seq_length: int = 256
    ) -> Dict[str, Any]:
        """Validate MoVE model against baseline Llama model.
        
        Args:
            baseline_model_name: Name of baseline model
            num_test_samples: Number of test samples
            seq_length: Sequence length for testing
            
        Returns:
            Validation results
        """
        # Initialize validator
        validator = MoVEValidator(baseline_model_name=baseline_model_name)
        
        # Create test dataset
        test_inputs = create_test_dataset(
            validator.baseline_extractor.tokenizer,
            num_samples=num_test_samples,
            seq_length=seq_length,
            dataset_type="real_text"
        )
        
        # Run validation
        results = validator.validate_full_model(self, test_inputs)
        
        # Create visualizations
        validator.plot_validation_results(results)
        
        return results

def create_move_model_config(
    model_size: str = "small"
) -> Dict[str, Any]:
    """Create MoVE model configuration for different sizes.
    
    Args:
        model_size: Model size ('small', 'medium', 'large')
        
    Returns:
        Model configuration dictionary
    """
    configs = {
        "small": {
            "vocab_size": 32000,
            "embed_dim": 1024,
            "max_seq_length": 1024,
            "num_attention_heads": 16,
            "num_experts": 4,
            "expert_capacity": 2,
            "dropout": 0.1
        },
        "medium": {
            "vocab_size": 32000,
            "embed_dim": 2048,
            "max_seq_length": 2048,
            "num_attention_heads": 32,
            "num_experts": 8,
            "expert_capacity": 2,
            "dropout": 0.1
        },
        "large": {
            "vocab_size": 32000,
            "embed_dim": 4096,
            "max_seq_length": 4096,
            "num_attention_heads": 64,
            "num_experts": 16,
            "expert_capacity": 2,
            "dropout": 0.1
        }
    }
    
    return configs.get(model_size, configs["small"])

def demo_move_model():
    """Demonstrate MoVE model functionality."""
    print("🚀 MoVE Model Demo")
    print("=" * 50)
    
    # Create model
    config = create_move_model_config("small")
    model = MoVEModel(**config)
    
    print(f"✅ Created MoVE model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    print(f"🔄 Testing forward pass with input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        print(f"✅ Forward pass successful! Output logits shape: {outputs['logits'].shape}")
    
    # Test generation
    print("🎯 Testing text generation...")
    prompt_ids = torch.randint(0, config["vocab_size"], (1, 10))
    
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_length=20,
            temperature=0.8,
            do_sample=True
        )
        print(f"✅ Generation successful! Generated sequence length: {generated.shape[1]}")
    
    # Get model statistics
    stats = model.get_model_stats()
    print(f"📊 Model Statistics:")
    print(f"   - Parameters: {stats['parameter_count']:,}")
    print(f"   - Trainable: {stats['trainable_parameters']:,}")
    print(f"   - Components: {len(stats['component_stats'])}")
    
    # Validate vector flow
    print("🔍 Validating vector flow...")
    validation_results = model.coordinator.validate_vector_flow(input_ids)
    
    if validation_results['success']:
        print("✅ Vector flow validation passed!")
    else:
        print("❌ Vector flow validation failed:")
        for error in validation_results['errors']:
            print(f"   - {error}")
    
    print("\n🎉 MoVE Model Demo completed successfully!")
    return model

if __name__ == "__main__":
    # Run demo
    model = demo_move_model()