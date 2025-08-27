"""MoVE Components Module

This module houses the core modular transformer components:
- Token Embedding Model: Sparse, hash-based token representations
- Positional Predictor: Dynamic, context-aware positional encoding
- Attention Approximator: Graph-based attention mechanisms
- Expert FFN Ensemble: Mixture-of-Experts feedforward networks
- Output Decoder: Task-specific output generation
"""

from .embedding import SparseEmbeddingModel, PositionalPredictor
from .attention import AttentionApproximator, GraphAttentionLayer, LowRankAttention
from .moe import ExpertFFNEnsemble, ExpertFFN, RouterNetwork, SharedExpert
from .decoder import OutputDecoder, LanguageModelingHead, ClassificationHead, AdaptiveDecoder
from .coordinator import (
    VectorProtocolCoordinator, 
    ComponentInterface, 
    VectorState, 
    VectorNormalizer,
    ComponentType
)

__all__ = [
    # Embedding components
    'SparseEmbeddingModel',
    'PositionalPredictor',
    
    # Attention components
    'AttentionApproximator',
    'GraphAttentionLayer', 
    'LowRankAttention',
    
    # MoE components
    'ExpertFFNEnsemble',
    'ExpertFFN',
    'RouterNetwork',
    'SharedExpert',
    
    # Decoder components
    'OutputDecoder',
    'LanguageModelingHead',
    'ClassificationHead',
    'AdaptiveDecoder',
    
    # Coordination system
    'VectorProtocolCoordinator',
    'ComponentInterface',
    'VectorState',
    'VectorNormalizer',
    'ComponentType'
]