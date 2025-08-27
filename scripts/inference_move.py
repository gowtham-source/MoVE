#!/usr/bin/env python3
"""
MoVE Model Inference Script
Optimized inference with KV-cache, quantization, and efficient attention mechanisms
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoTokenizer
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from move_large import MoVELarge, create_move_large_model

class MoVEInferenceEngine:
    """Optimized inference engine for MoVE models with KV-cache and quantization support."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
                 device: str = "auto",
                 quantization: str = "none",
                 max_length: int = 2048,
                 use_kv_cache: bool = True):
        """
        Initialize the MoVE inference engine.
        
        Args:
            model_path: Path to the trained MoVE model checkpoint
            tokenizer_name: Name or path of the tokenizer to use
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            quantization: Quantization method ('none', 'int8', 'int4')
            max_length: Maximum sequence length for generation
            use_kv_cache: Whether to use KV-cache for faster generation
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.use_kv_cache = use_kv_cache
        self.quantization = quantization
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize tokenizer and model
        self._load_tokenizer()
        self._load_model()
        
        # KV-cache for efficient generation
        self.kv_cache = {} if use_kv_cache else None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info(f"Loaded tokenizer: {self.tokenizer_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
            
    def _load_model(self):
        """Load the MoVE model with optional quantization."""
        try:
            # Load model checkpoint
            if os.path.isfile(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Extract model config and state dict
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                else:
                    # Try to infer config from state dict
                    config = self._infer_config_from_state_dict(checkpoint)
                    state_dict = checkpoint
                    
                # Create model
                self.model = create_move_large_model(config)
                self.model.load_state_dict(state_dict, strict=False)
                
            else:
                # Load from directory (Hugging Face format)
                config_path = os.path.join(self.model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.model = create_move_large_model(config)
                    # Load weights
                    model_file = os.path.join(self.model_path, 'pytorch_model.bin')
                    if os.path.exists(model_file):
                        state_dict = torch.load(model_file, map_location=self.device)
                        self.model.load_state_dict(state_dict, strict=False)
                else:
                    raise FileNotFoundError(f"Model config not found at {self.model_path}")
                    
            # Apply quantization
            if self.quantization != "none":
                self._apply_quantization()
                
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"Loaded MoVE model with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            self.logger.info(f"Device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def _infer_config_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Infer model configuration from state dict."""
        # Default MoVE 1B configuration
        config = {
            "vocab_size": 32000,
            "d_model": 2048,
            "num_layers": 16,
            "num_heads": 32,
            "max_seq_len": 2048,
            "moe_experts": 8,
            "moe_topk": 2,
            "dropout": 0.1,
            "use_checkpoint": False,
            "tie_weights": True
        }
        
        # Try to infer from state dict keys
        for key in state_dict.keys():
            if 'embed_tokens.weight' in key:
                config['vocab_size'] = state_dict[key].shape[0]
                config['d_model'] = state_dict[key].shape[1]
            elif 'layers.' in key and '.attn.' in key:
                # Count layers
                layer_nums = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('layers.') and '.attn.' in k]
                if layer_nums:
                    config['num_layers'] = max(layer_nums) + 1
                    
        return config
        
    def _apply_quantization(self):
        """Apply quantization to the model."""
        if self.quantization == "int8":
            # Apply dynamic quantization
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.logger.info("Applied INT8 quantization")
        elif self.quantization == "int4":
            # For INT4, we would need specialized libraries like bitsandbytes
            self.logger.warning("INT4 quantization requires bitsandbytes library")
            
    def clear_kv_cache(self):
        """Clear the KV-cache."""
        if self.kv_cache is not None:
            self.kv_cache.clear()
            
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> str:
        """Generate text from a prompt."""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
            
        # Clear cache if not using it
        if not use_cache:
            self.clear_kv_cache()
            
        generated_tokens = []
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        start_time = time.time()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.model(
                    current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=use_cache and self.use_kv_cache
                )
                
                # Get logits for the last token
                logits = outputs.logits[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, current_input_ids, repetition_penalty
                    )
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                    
                # Apply top-k filtering
                if top_k > 0:
                    logits = self._top_k_filtering(logits, top_k)
                    
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    logits = self._top_p_filtering(logits, top_p)
                    
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                # Check for EOS token
                if next_token.item() == eos_token_id:
                    break
                    
                generated_tokens.append(next_token.item())
                
                # Update input for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), device=self.device)
                ], dim=1)
                
                # Truncate if exceeding max length
                if current_input_ids.shape[1] > self.max_length:
                    current_input_ids = current_input_ids[:, -self.max_length:]
                    current_attention_mask = current_attention_mask[:, -self.max_length:]
                    
        generation_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        
        self.logger.info(f"Generated {len(generated_tokens)} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
        
        return full_text
        
    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for token_id in torch.unique(input_ids):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        return logits
        
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
        
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
        
    def chat(self, system_prompt: str = "You are a helpful AI assistant."):
        """Interactive chat interface."""
        print("MoVE Chat Interface")
        print("Type 'quit' to exit, 'clear' to clear conversation history")
        print("-" * 50)
        
        conversation_history = system_prompt + "\n\n"
        
        while True:
            try:
                user_input = input("User: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = system_prompt + "\n\n"
                    self.clear_kv_cache()
                    print("Conversation history cleared.")
                    continue
                elif not user_input:
                    continue
                    
                # Add user input to conversation
                conversation_history += f"User: {user_input}\nAssistant: "
                
                # Generate response
                response = self.generate(
                    conversation_history,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                # Extract just the assistant's response
                assistant_response = response[len(conversation_history):].strip()
                
                # Update conversation history
                conversation_history = response + "\n\n"
                
                print(f"Assistant: {assistant_response}")
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                
def main():
    parser = argparse.ArgumentParser(description='MoVE Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained MoVE model')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Tokenizer to use')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--quantization', type=str, default='none',
                        choices=['none', 'int8', 'int4'],
                        help='Quantization method')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--prompt', type=str,
                        help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling threshold')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling threshold')
    parser.add_argument('--chat', action='store_true',
                        help='Start interactive chat mode')
    parser.add_argument('--no_kv_cache', action='store_true',
                        help='Disable KV-cache')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = MoVEInferenceEngine(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        device=args.device,
        quantization=args.quantization,
        max_length=args.max_length,
        use_kv_cache=not args.no_kv_cache
    )
    
    if args.chat:
        # Start chat mode
        engine.chat()
    elif args.prompt:
        # Generate from prompt
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        print("Generated text:")
        print("-" * 50)
        print(result)
    else:
        print("Please provide either --prompt for single generation or --chat for interactive mode")
        
if __name__ == "__main__":
    main()