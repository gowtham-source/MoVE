"""Vector Protocol Coordinator - Component Communication System

This module implements the coordination system that manages communication
between modular components using a shared vector protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

class ComponentType(Enum):
    """Enumeration of component types in the MoVE system."""
    EMBEDDING = "embedding"
    POSITIONAL = "positional"
    ATTENTION = "attention"
    FFN = "ffn"
    DECODER = "decoder"

@dataclass
class VectorState:
    """Represents the state of vectors flowing through the system."""
    hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    component_outputs: Optional[Dict[str, torch.Tensor]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.component_outputs is None:
            self.component_outputs = {}
    
    def clone(self) -> 'VectorState':
        """Create a deep copy of the vector state."""
        return VectorState(
            hidden_states=self.hidden_states.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            position_ids=self.position_ids.clone() if self.position_ids is not None else None,
            metadata=self.metadata.copy(),
            component_outputs={k: v.clone() if isinstance(v, torch.Tensor) else v 
                             for k, v in self.component_outputs.items()}
        )
    
    def update_hidden_states(self, new_states: torch.Tensor, component_name: str):
        """Update hidden states and track the component that produced them."""
        self.hidden_states = new_states
        self.component_outputs[component_name] = new_states
        self.metadata[f'last_updated_by'] = component_name

class VectorNormalizer(nn.Module):
    """Normalizes vectors to maintain consistent scale across components."""
    
    def __init__(
        self,
        embed_dim: int,
        normalization_type: str = "layer_norm",  # "layer_norm", "rms_norm", "batch_norm"
        eps: float = 1e-6
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalization_type = normalization_type
        self.eps = eps
        
        if normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim, eps=eps)
        elif normalization_type == "rms_norm":
            self.norm = RMSNorm(embed_dim, eps=eps)
        elif normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(embed_dim, eps=eps)
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input tensor."""
        if self.normalization_type == "batch_norm":
            # Reshape for batch norm: [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len]
            original_shape = x.shape
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous()
            return x
        else:
            return self.norm(x)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

class ComponentInterface(nn.Module):
    """Base interface for all MoVE components."""
    
    def __init__(self, component_type: ComponentType, embed_dim: int):
        super().__init__()
        self.component_type = component_type
        self.embed_dim = embed_dim
        self.component_id = f"{component_type.value}_{id(self)}"
        
        # Input/output normalizers
        self.input_normalizer = VectorNormalizer(embed_dim)
        self.output_normalizer = VectorNormalizer(embed_dim)
        
        # Component statistics
        self.register_buffer('forward_count', torch.tensor(0))
        self.register_buffer('total_processing_time', torch.tensor(0.0))
    
    def preprocess(self, vector_state: VectorState) -> VectorState:
        """Preprocess input vector state."""
        # Normalize input
        vector_state.hidden_states = self.input_normalizer(vector_state.hidden_states)
        return vector_state
    
    def postprocess(self, vector_state: VectorState) -> VectorState:
        """Postprocess output vector state."""
        # Normalize output
        vector_state.hidden_states = self.output_normalizer(vector_state.hidden_states)
        
        # Update metadata
        vector_state.metadata[f'{self.component_type.value}_processed'] = True
        
        return vector_state
    
    def forward_with_timing(self, vector_state: VectorState) -> VectorState:
        """Forward pass with timing statistics."""
        import time
        start_time = time.time()
        
        # Preprocess
        vector_state = self.preprocess(vector_state)
        
        # Main forward pass (to be implemented by subclasses)
        vector_state = self.component_forward(vector_state)
        
        # Postprocess
        vector_state = self.postprocess(vector_state)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.forward_count += 1
        self.total_processing_time += processing_time
        
        return vector_state
    
    def component_forward(self, vector_state: VectorState) -> VectorState:
        """Component-specific forward pass. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, float]:
        """Get component processing statistics."""
        avg_time = (self.total_processing_time / self.forward_count.clamp(min=1)).item()
        return {
            'forward_count': self.forward_count.item(),
            'total_time': self.total_processing_time.item(),
            'avg_time_per_forward': avg_time
        }

class VectorProtocolCoordinator(nn.Module):
    """Main coordinator that manages the flow of vectors between components."""
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        max_seq_length: int = 2048,
        use_residual_connections: bool = True,
        use_gradient_checkpointing: bool = False,
        component_dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.use_residual_connections = use_residual_connections
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Component registry
        self.components: Dict[ComponentType, nn.Module] = {}
        self.component_order = []
        
        # Global normalizers
        self.global_input_norm = VectorNormalizer(embed_dim)
        self.global_output_norm = VectorNormalizer(embed_dim)
        
        # Residual connection weights (learnable)
        if use_residual_connections:
            self.residual_weights = nn.ParameterDict({
                comp_type.value: nn.Parameter(torch.tensor(0.5)) 
                for comp_type in ComponentType
            })
        
        # Component dropout
        self.component_dropout = nn.Dropout(component_dropout)
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_component(self, component_type: ComponentType, component: nn.Module):
        """Register a component with the coordinator."""
        self.components[component_type] = component
        if component_type not in self.component_order:
            self.component_order.append(component_type)
        
        self.logger.info(f"Registered component: {component_type.value}")
    
    def set_component_order(self, order: List[ComponentType]):
        """Set the order in which components are executed."""
        # Validate that all components are registered
        for comp_type in order:
            if comp_type not in self.components:
                raise ValueError(f"Component {comp_type.value} not registered")
        
        self.component_order = order
        self.logger.info(f"Set component order: {[c.value for c in order]}")
    
    def create_vector_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> VectorState:
        """Create initial vector state from input tokens."""
        batch_size, seq_len = input_ids.shape
        
        # Initialize with zero hidden states (will be filled by embedding component)
        hidden_states = torch.zeros(
            batch_size, seq_len, self.embed_dim,
            device=input_ids.device, dtype=torch.float32
        )
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Create vector state
        vector_state = VectorState(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata={
                'input_ids': input_ids,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'device': input_ids.device
            }
        )
        
        return vector_state
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through all components.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            return_dict: Whether to return dictionary output
            output_hidden_states: Whether to return intermediate hidden states
            
        Returns:
            Final output or dictionary of outputs
        """
        # Create initial vector state
        vector_state = self.create_vector_state(input_ids, attention_mask, position_ids)
        
        # Apply global input normalization
        vector_state.hidden_states = self.global_input_norm(vector_state.hidden_states)
        
        # Store intermediate states if requested
        hidden_states_list = []
        if output_hidden_states:
            hidden_states_list.append(vector_state.hidden_states.clone())
        
        # Process through each component
        for comp_type in self.component_order:
            if comp_type not in self.components:
                self.logger.warning(f"Component {comp_type.value} not found, skipping")
                continue
            
            component = self.components[comp_type]
            
            # Store previous state for residual connection
            prev_hidden_states = vector_state.hidden_states.clone() if self.use_residual_connections else None
            
            # Forward through component
            if self.use_gradient_checkpointing and self.training:
                vector_state = torch.utils.checkpoint.checkpoint(
                    self._component_forward_wrapper,
                    component,
                    vector_state,
                    use_reentrant=False
                )
            else:
                vector_state = self._component_forward(component, vector_state)
            
            # Apply residual connection
            if self.use_residual_connections and prev_hidden_states is not None:
                weight = torch.sigmoid(self.residual_weights[comp_type.value])
                vector_state.hidden_states = (
                    weight * vector_state.hidden_states + 
                    (1 - weight) * prev_hidden_states
                )
            
            # Apply component dropout
            if self.training:
                vector_state.hidden_states = self.component_dropout(vector_state.hidden_states)
            
            # Store intermediate state
            if output_hidden_states:
                hidden_states_list.append(vector_state.hidden_states.clone())
        
        # Apply global output normalization
        vector_state.hidden_states = self.global_output_norm(vector_state.hidden_states)
        
        # Prepare output
        if return_dict:
            output = {
                'last_hidden_state': vector_state.hidden_states,
                'component_outputs': vector_state.component_outputs,
                'metadata': vector_state.metadata
            }
            
            if output_hidden_states:
                output['hidden_states'] = hidden_states_list
            
            return output
        else:
            return vector_state.hidden_states
    
    def _component_forward(self, component: nn.Module, vector_state: VectorState) -> VectorState:
        """Forward pass through a single component."""
        if hasattr(component, 'forward_with_timing'):
            return component.forward_with_timing(vector_state)
        elif hasattr(component, 'component_forward'):
            return component.component_forward(vector_state)
        else:
            # Fallback for standard PyTorch modules
            vector_state.hidden_states = component(vector_state.hidden_states)
            return vector_state
    
    def _component_forward_wrapper(self, component: nn.Module, vector_state: VectorState) -> VectorState:
        """Wrapper for gradient checkpointing."""
        return self._component_forward(component, vector_state)
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all components."""
        stats = {}
        for comp_type, component in self.components.items():
            if hasattr(component, 'get_stats'):
                stats[comp_type.value] = component.get_stats()
        return stats
    
    def validate_vector_flow(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Validate that vectors flow correctly through all components."""
        validation_results = {
            'success': True,
            'errors': [],
            'component_outputs': {},
            'shape_consistency': True
        }
        
        try:
            # Test forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids, return_dict=True, output_hidden_states=True)
                
                # Check shape consistency
                expected_shape = (input_ids.shape[0], input_ids.shape[1], self.embed_dim)
                if outputs['last_hidden_state'].shape != expected_shape:
                    validation_results['shape_consistency'] = False
                    validation_results['errors'].append(
                        f"Output shape mismatch: expected {expected_shape}, got {outputs['last_hidden_state'].shape}"
                    )
                
                # Store component outputs for analysis
                validation_results['component_outputs'] = outputs['component_outputs']
                
        except Exception as e:
            validation_results['success'] = False
            validation_results['errors'].append(str(e))
        
        return validation_results
    
    def estimate_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Estimate memory usage for given input dimensions."""
        # Base memory for hidden states
        hidden_state_memory = batch_size * seq_len * self.embed_dim * 4  # 4 bytes per float32
        
        # Estimate component memory (simplified)
        component_memory = 0
        for comp_type, component in self.components.items():
            if hasattr(component, 'estimate_memory'):
                component_memory += component.estimate_memory(batch_size, seq_len)
            else:
                # Rough estimate based on parameters
                param_count = sum(p.numel() for p in component.parameters())
                component_memory += param_count * 4  # 4 bytes per parameter
        
        total_memory = hidden_state_memory + component_memory
        
        return {
            'hidden_states_mb': hidden_state_memory / (1024 * 1024),
            'components_mb': component_memory / (1024 * 1024),
            'total_mb': total_memory / (1024 * 1024),
            'total_gb': total_memory / (1024 * 1024 * 1024)
        }