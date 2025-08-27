"""Modular Vector Engine (MoVE) - A Deconstructed LLM

MoVE is a novel approach to large language models that deconstructs
traditional transformer architectures into modular, specialized components.

Inspired by Llama-style transformers, MoVE replaces monolithic transformer
blocks with independent ML models that collaboratively compute vector
transformations with greater efficiency, interpretability, and adaptability.

Core Components:
- Token Embedding Model: Sparse, hash-based token representations
- Positional Predictor: Dynamic, context-aware positional encoding  
- Attention Approximator: Graph-based attention mechanisms
- Expert FFN Ensemble: Mixture-of-Experts feedforward networks
- Output Decoder: Task-specific output generation

Author: MoVE Research Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "MoVE Research Team"
__email__ = "move-research@example.com"

# Import core modules
from . import components
from . import training  
from . import evaluation
from .integration import MoVEModel, create_move_model_config, demo_move_model

# Import key classes for easy access
from .components import (
    SparseEmbeddingModel,
    PositionalPredictor,
    AttentionApproximator,
    ExpertFFNEnsemble,
    OutputDecoder,
    VectorProtocolCoordinator,
    ComponentType
)
from .evaluation import MoVEValidator, ValidationMetrics

__all__ = [
    # Main model
    'MoVEModel',
    'create_move_model_config',
    'demo_move_model',
    
    # Core components
    'SparseEmbeddingModel',
    'PositionalPredictor', 
    'AttentionApproximator',
    'ExpertFFNEnsemble',
    'OutputDecoder',
    'VectorProtocolCoordinator',
    'ComponentType',
    
    # Evaluation
    'MoVEValidator',
    'ValidationMetrics',
    
    # Modules
    'components',
    'training',
    'evaluation'
]