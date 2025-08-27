"""Validation Framework - Compare MoVE outputs with Llama activations

This module implements the validation framework to ensure MoVE components
replicate Llama-style transformer behavior with high fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import wandb
from tqdm import tqdm

@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    cosine_similarity: float
    mse_loss: float
    mae_loss: float
    pearson_correlation: float
    spearman_correlation: float
    l2_distance: float
    relative_error: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'cosine_similarity': self.cosine_similarity,
            'mse_loss': self.mse_loss,
            'mae_loss': self.mae_loss,
            'pearson_correlation': self.pearson_correlation,
            'spearman_correlation': self.spearman_correlation,
            'l2_distance': self.l2_distance,
            'relative_error': self.relative_error
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        return (f"ValidationMetrics(cosine_sim={self.cosine_similarity:.4f}, "
                f"mse={self.mse_loss:.6f}, pearson={self.pearson_correlation:.4f})")

class BaselineExtractor:
    """Extracts activations from baseline Llama model for comparison."""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            output_hidden_states=True,
            output_attentions=True
        )
        self.model.eval()
        
        # Hook storage
        self.activations = {}
        self.hooks = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loaded baseline model: {model_name}")
    
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture intermediate activations."""
        if layer_names is None:
            # Default to all transformer layers
            layer_names = [f"layers.{i}" for i in range(len(self.model.layers))]
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook_fn
        
        # Register hooks
        for layer_name in layer_names:
            try:
                layer = dict(self.model.named_modules())[layer_name]
                hook = layer.register_forward_hook(create_hook(layer_name))
                self.hooks.append(hook)
                self.logger.debug(f"Registered hook for layer: {layer_name}")
            except KeyError:
                self.logger.warning(f"Layer {layer_name} not found in model")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def extract_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from baseline model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of layer activations
        """
        self.activations.clear()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None
            )
        
        # Add final hidden states
        self.activations['final_hidden_states'] = outputs.last_hidden_state.detach().cpu()
        
        # Add layer-wise hidden states if available
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            for i, hidden_state in enumerate(outputs.hidden_states):
                self.activations[f'layer_{i}_hidden_states'] = hidden_state.detach().cpu()
        
        return self.activations.copy()
    
    def get_embedding_output(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embedding layer output."""
        with torch.no_grad():
            embeddings = self.model.embed_tokens(input_ids.to(self.model.device))
            return embeddings.detach().cpu()

class MoVEValidator:
    """Main validation class for comparing MoVE with baseline Llama."""
    
    def __init__(
        self,
        baseline_model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        target_cosine_similarity: float = 0.95,
        device: str = "auto",
        save_dir: str = "validation_results"
    ):
        self.baseline_model_name = baseline_model_name
        self.target_cosine_similarity = target_cosine_similarity
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize baseline extractor
        self.baseline_extractor = BaselineExtractor(baseline_model_name, device)
        
        # Validation history
        self.validation_history = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_metrics(
        self,
        move_output: torch.Tensor,
        baseline_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> ValidationMetrics:
        """Compute validation metrics between MoVE and baseline outputs.
        
        Args:
            move_output: MoVE model output [batch_size, seq_len, embed_dim]
            baseline_output: Baseline model output [batch_size, seq_len, embed_dim]
            mask: Optional mask for valid positions [batch_size, seq_len]
            
        Returns:
            ValidationMetrics object
        """
        # Ensure same device and dtype
        move_output = move_output.float()
        baseline_output = baseline_output.float()
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            move_output = move_output * mask
            baseline_output = baseline_output * mask
        
        # Flatten for metric computation
        move_flat = move_output.view(-1, move_output.size(-1))
        baseline_flat = baseline_output.view(-1, baseline_output.size(-1))
        
        # Remove zero vectors (from masking)
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            move_flat = move_flat[mask_flat]
            baseline_flat = baseline_flat[mask_flat]
        
        # Cosine similarity
        cos_sim_matrix = cosine_similarity(move_flat.numpy(), baseline_flat.numpy())
        cosine_sim = np.mean(np.diag(cos_sim_matrix))
        
        # MSE and MAE
        mse = F.mse_loss(move_flat, baseline_flat).item()
        mae = F.l1_loss(move_flat, baseline_flat).item()
        
        # L2 distance
        l2_dist = torch.norm(move_flat - baseline_flat, dim=-1).mean().item()
        
        # Relative error
        baseline_norm = torch.norm(baseline_flat, dim=-1).mean()
        relative_error = (l2_dist / baseline_norm.item()) if baseline_norm > 0 else float('inf')
        
        # Correlation metrics
        move_mean = move_flat.mean(dim=0).numpy()
        baseline_mean = baseline_flat.mean(dim=0).numpy()
        
        pearson_corr, _ = pearsonr(move_mean, baseline_mean)
        spearman_corr, _ = spearmanr(move_mean, baseline_mean)
        
        return ValidationMetrics(
            cosine_similarity=cosine_sim,
            mse_loss=mse,
            mae_loss=mae,
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            l2_distance=l2_dist,
            relative_error=relative_error
        )
    
    def validate_component(
        self,
        move_component: nn.Module,
        input_ids: torch.Tensor,
        component_name: str,
        attention_mask: Optional[torch.Tensor] = None,
        baseline_layer: Optional[str] = None
    ) -> ValidationMetrics:
        """Validate a single MoVE component against baseline.
        
        Args:
            move_component: MoVE component to validate
            input_ids: Input token IDs
            component_name: Name of the component
            attention_mask: Optional attention mask
            baseline_layer: Specific baseline layer to compare against
            
        Returns:
            ValidationMetrics for the component
        """
        self.logger.info(f"Validating component: {component_name}")
        
        # Extract baseline activations
        if baseline_layer:
            self.baseline_extractor.register_hooks([baseline_layer])
        
        baseline_activations = self.baseline_extractor.extract_activations(input_ids, attention_mask)
        
        # Get MoVE component output
        move_component.eval()
        with torch.no_grad():
            if hasattr(move_component, 'component_forward'):
                # Use component interface
                from ..components.coordinator import VectorState
                vector_state = VectorState(
                    hidden_states=torch.zeros(input_ids.shape[0], input_ids.shape[1], move_component.embed_dim),
                    attention_mask=attention_mask,
                    metadata={'input_ids': input_ids}
                )
                output_state = move_component.component_forward(vector_state)
                move_output = output_state.hidden_states
            else:
                # Standard forward pass
                move_output = move_component(input_ids)
        
        # Select appropriate baseline output
        if baseline_layer and baseline_layer in baseline_activations:
            baseline_output = baseline_activations[baseline_layer]
        else:
            baseline_output = baseline_activations.get('final_hidden_states', 
                                                     list(baseline_activations.values())[0])
        
        # Compute metrics
        metrics = self.compute_metrics(move_output, baseline_output, attention_mask)
        
        # Clean up hooks
        self.baseline_extractor.remove_hooks()
        
        self.logger.info(f"Component {component_name} validation: {metrics}")
        return metrics
    
    def validate_full_model(
        self,
        move_model: nn.Module,
        test_inputs: List[torch.Tensor],
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """Validate full MoVE model against baseline.
        
        Args:
            move_model: Complete MoVE model
            test_inputs: List of test input tensors
            batch_size: Batch size for validation
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting full model validation")
        
        all_metrics = []
        detailed_results = {
            'per_sample_metrics': [],
            'aggregate_metrics': {},
            'validation_passed': False,
            'target_cosine_similarity': self.target_cosine_similarity
        }
        
        move_model.eval()
        
        # Process in batches
        for i in tqdm(range(0, len(test_inputs), batch_size), desc="Validating batches"):
            batch_inputs = test_inputs[i:i+batch_size]
            
            for input_ids in batch_inputs:
                # Create attention mask
                attention_mask = torch.ones_like(input_ids)
                
                # Get baseline output
                baseline_activations = self.baseline_extractor.extract_activations(input_ids, attention_mask)
                baseline_output = baseline_activations['final_hidden_states']
                
                # Get MoVE output
                with torch.no_grad():
                    if hasattr(move_model, 'forward'):
                        move_outputs = move_model(input_ids, attention_mask=attention_mask, return_dict=True)
                        move_output = move_outputs.get('last_hidden_state', move_outputs)
                    else:
                        move_output = move_model(input_ids)
                
                # Compute metrics
                metrics = self.compute_metrics(move_output, baseline_output, attention_mask)
                all_metrics.append(metrics)
                detailed_results['per_sample_metrics'].append(metrics.to_dict())
        
        # Aggregate metrics
        if all_metrics:
            detailed_results['aggregate_metrics'] = {
                'mean_cosine_similarity': np.mean([m.cosine_similarity for m in all_metrics]),
                'std_cosine_similarity': np.std([m.cosine_similarity for m in all_metrics]),
                'mean_mse': np.mean([m.mse_loss for m in all_metrics]),
                'mean_pearson': np.mean([m.pearson_correlation for m in all_metrics]),
                'min_cosine_similarity': np.min([m.cosine_similarity for m in all_metrics]),
                'max_cosine_similarity': np.max([m.cosine_similarity for m in all_metrics])
            }
            
            # Check if validation passed
            mean_cosine_sim = detailed_results['aggregate_metrics']['mean_cosine_similarity']
            detailed_results['validation_passed'] = mean_cosine_sim >= self.target_cosine_similarity
        
        # Save results
        self.save_validation_results(detailed_results)
        
        # Log results
        if detailed_results['validation_passed']:
            self.logger.info(f"✅ Validation PASSED! Mean cosine similarity: {mean_cosine_sim:.4f}")
        else:
            self.logger.warning(f"❌ Validation FAILED. Mean cosine similarity: {mean_cosine_sim:.4f} < {self.target_cosine_similarity}")
        
        return detailed_results
    
    def save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to disk."""
        timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.save_dir / f"validation_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Deep convert the results
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Validation results saved to: {results_file}")
    
    def plot_validation_results(self, results: Dict[str, Any]):
        """Create visualization plots for validation results."""
        if not results['per_sample_metrics']:
            return
        
        # Extract metrics for plotting
        cosine_sims = [m['cosine_similarity'] for m in results['per_sample_metrics']]
        mse_losses = [m['mse_loss'] for m in results['per_sample_metrics']]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cosine similarity distribution
        axes[0, 0].hist(cosine_sims, bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(self.target_cosine_similarity, color='red', linestyle='--', 
                          label=f'Target: {self.target_cosine_similarity}')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cosine Similarity Distribution')
        axes[0, 0].legend()
        
        # MSE loss distribution
        axes[0, 1].hist(mse_losses, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('MSE Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('MSE Loss Distribution')
        
        # Cosine similarity over samples
        axes[1, 0].plot(cosine_sims, alpha=0.7)
        axes[1, 0].axhline(self.target_cosine_similarity, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Cosine Similarity per Sample')
        
        # Scatter plot: Cosine sim vs MSE
        axes[1, 1].scatter(cosine_sims, mse_losses, alpha=0.6)
        axes[1, 1].set_xlabel('Cosine Similarity')
        axes[1, 1].set_ylabel('MSE Loss')
        axes[1, 1].set_title('Cosine Similarity vs MSE Loss')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.save_dir / f"validation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Validation plots saved to: {plot_file}")
    
    def log_to_wandb(self, results: Dict[str, Any], run_name: Optional[str] = None):
        """Log validation results to Weights & Biases."""
        if not wandb.run:
            wandb.init(project="move-validation", name=run_name)
        
        # Log aggregate metrics
        if 'aggregate_metrics' in results:
            wandb.log(results['aggregate_metrics'])
        
        # Log validation status
        wandb.log({
            'validation_passed': results.get('validation_passed', False),
            'target_cosine_similarity': results.get('target_cosine_similarity', self.target_cosine_similarity)
        })
        
        self.logger.info("Results logged to Weights & Biases")

def create_test_dataset(
    tokenizer,
    num_samples: int = 100,
    seq_length: int = 512,
    dataset_type: str = "random"  # "random", "real_text"
) -> List[torch.Tensor]:
    """Create test dataset for validation.
    
    Args:
        tokenizer: Tokenizer to use
        num_samples: Number of test samples
        seq_length: Sequence length
        dataset_type: Type of test data to generate
        
    Returns:
        List of input tensors
    """
    test_inputs = []
    
    if dataset_type == "random":
        # Generate random token sequences
        vocab_size = tokenizer.vocab_size
        for _ in range(num_samples):
            input_ids = torch.randint(0, vocab_size, (1, seq_length))
            test_inputs.append(input_ids)
    
    elif dataset_type == "real_text":
        # Use real text samples (simplified)
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of artificial intelligence.",
            "Large language models have revolutionized natural language processing.",
            "Deep learning architectures continue to evolve and improve.",
            "Transformer models have become the backbone of modern NLP systems."
        ] * (num_samples // 5 + 1)
        
        for i in range(num_samples):
            text = sample_texts[i % len(sample_texts)]
            tokens = tokenizer(text, return_tensors="pt", max_length=seq_length, 
                             truncation=True, padding="max_length")
            test_inputs.append(tokens['input_ids'])
    
    return test_inputs[:num_samples]

# Example usage and testing functions
def run_validation_example():
    """Example of how to use the validation framework."""
    # Initialize validator
    validator = MoVEValidator()
    
    # Create test dataset
    test_inputs = create_test_dataset(
        validator.baseline_extractor.tokenizer,
        num_samples=50,
        seq_length=256
    )
    
    # Mock MoVE model for testing
    class MockMoVEModel(nn.Module):
        def __init__(self, embed_dim=2048, vocab_size=32000):
            super().__init__()
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        def forward(self, input_ids, attention_mask=None, return_dict=True):
            x = self.embedding(input_ids)
            x = self.output_proj(x)
            if return_dict:
                return {'last_hidden_state': x}
            return x
    
    # Create and validate mock model
    mock_model = MockMoVEModel()
    results = validator.validate_full_model(mock_model, test_inputs)
    
    # Create visualizations
    validator.plot_validation_results(results)
    
    return results

if __name__ == "__main__":
    # Run example validation
    results = run_validation_example()
    print(f"Validation completed. Results: {results['aggregate_metrics']}")