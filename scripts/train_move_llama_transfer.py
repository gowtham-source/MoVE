"""Transfer Learning from LLaMA 1B to MoVE Architecture

This script implements transfer learning by loading pre-trained LLaMA weights
and adapting them to our MoVE architecture for efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import argparse
import os
import json
import math
from datasets import load_from_disk
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from move_large import MoVELarge
from modules.token_embed import TokenEmbed
from modules.pos_gen import PosGen
from modules.attn_approx import AttnApprox
from modules.ffn_moe import AdaptiveFFNMoE

class LLaMAToMoVETransfer:
    """Handles transfer learning from LLaMA to MoVE architecture."""
    
    def __init__(self, llama_model_name: str = "unsloth/Llama-3.2-1B-bnb-4bit"):
        self.llama_model_name = llama_model_name
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_llama_model(self) -> tuple:
        """Load pre-trained LLaMA model and tokenizer."""
        self.logger.info(f"Loading LLaMA model: {self.llama_model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            llama_model = AutoModelForCausalLM.from_pretrained(
                self.llama_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True  # Enable 4-bit quantization for memory efficiency
            )
            
            self.logger.info(f"Successfully loaded LLaMA model with {llama_model.num_parameters():,} parameters")
            return llama_model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load LLaMA model: {e}")
            raise
    
    def create_move_config_from_llama(self, llama_config: LlamaConfig) -> Dict[str, Any]:
        """Create MoVE configuration based on LLaMA config."""
        # Map LLaMA config to MoVE config
        move_config = {
            'vocab_size': llama_config.vocab_size,
            'd_model': llama_config.hidden_size,
            'num_layers': llama_config.num_hidden_layers,
            'num_heads': llama_config.num_attention_heads,
            'max_seq_len': llama_config.max_position_embeddings,
            'moe_experts': 8,  # MoVE-specific: number of experts
            'moe_topk': 2,     # MoVE-specific: top-k routing
            'use_lora': True,  # Enable LoRA for efficient fine-tuning
            'dropout': 0.1,
            'use_checkpoint': True,
            'tie_weights': True
        }
        
        self.logger.info(f"Created MoVE config: {move_config}")
        return move_config
    
    def transfer_weights(self, llama_model: nn.Module, move_model: MoVELarge) -> MoVELarge:
        """Transfer compatible weights from LLaMA to MoVE model."""
        self.logger.info("Starting weight transfer from LLaMA to MoVE...")
        
        llama_state_dict = llama_model.state_dict()
        move_state_dict = move_model.state_dict()
        
        transferred_weights = 0
        total_weights = len(move_state_dict)
        
        # Transfer token embeddings
        if 'model.embed_tokens.weight' in llama_state_dict:
            move_state_dict['token_embed.embed.weight'] = llama_state_dict['model.embed_tokens.weight'].clone()
            transferred_weights += 1
            self.logger.info("Transferred token embeddings")
        
        # Transfer output projection (language modeling head)
        if 'lm_head.weight' in llama_state_dict:
            move_state_dict['lm_head.weight'] = llama_state_dict['lm_head.weight'].clone()
            transferred_weights += 1
            self.logger.info("Transferred language modeling head")
        
        # Transfer layer normalization weights
        if 'model.norm.weight' in llama_state_dict:
            move_state_dict['norm.weight'] = llama_state_dict['model.norm.weight'].clone()
            transferred_weights += 1
            self.logger.info("Transferred final layer norm")
        
        if 'model.norm.bias' in llama_state_dict:
            move_state_dict['norm.bias'] = llama_state_dict['model.norm.bias'].clone()
            transferred_weights += 1
        
        # Transfer transformer layer weights
        for layer_idx in range(move_model.num_layers):
            llama_prefix = f'model.layers.{layer_idx}'
            move_prefix = f'layers.{layer_idx}'
            
            # Transfer attention layer norms
            llama_attn_norm = f'{llama_prefix}.input_layernorm.weight'
            move_attn_norm = f'{move_prefix}.norm1.weight'
            if llama_attn_norm in llama_state_dict and move_attn_norm in move_state_dict:
                move_state_dict[move_attn_norm] = llama_state_dict[llama_attn_norm].clone()
                transferred_weights += 1
            
            # Transfer FFN layer norms
            llama_ffn_norm = f'{llama_prefix}.post_attention_layernorm.weight'
            move_ffn_norm = f'{move_prefix}.norm2.weight'
            if llama_ffn_norm in llama_state_dict and move_ffn_norm in move_state_dict:
                move_state_dict[move_ffn_norm] = llama_state_dict[llama_ffn_norm].clone()
                transferred_weights += 1
            
            # Note: Attention and FFN weights require special handling due to architectural differences
            # LLaMA uses standard attention while MoVE uses AttnApprox
            # LLaMA uses standard FFN while MoVE uses MoE FFN
            # These will be initialized randomly and fine-tuned
        
        # Load the transferred weights
        move_model.load_state_dict(move_state_dict, strict=False)
        
        transfer_ratio = transferred_weights / total_weights * 100
        self.logger.info(f"Transferred {transferred_weights}/{total_weights} weights ({transfer_ratio:.1f}%)")
        self.logger.info("Weight transfer completed. Attention and FFN layers will be fine-tuned from scratch.")
        
        return move_model
    
    def setup_lora_fine_tuning(self, model: MoVELarge) -> MoVELarge:
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        self.logger.info("Setting up LoRA fine-tuning...")
        
        # Freeze transferred weights (embeddings, layer norms, output head)
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            # Freeze token embeddings and output head (transferred from LLaMA)
            if any(freeze_name in name for freeze_name in ['token_embed', 'lm_head', 'norm']):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                # Keep MoVE-specific components trainable (attention, MoE FFN)
                param.requires_grad = True
                trainable_params += param.numel()
        
        total_params = frozen_params + trainable_params
        trainable_ratio = trainable_params / total_params * 100
        
        self.logger.info(f"Frozen parameters: {frozen_params:,} ({100-trainable_ratio:.1f}%)")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.1f}%)")
        
        return model

class PreTokenizedDataset:
    """Dataset for pre-tokenized data with input_ids and attention_mask."""
    
    def __init__(self, dataset_path: str, max_length: int = 1024):
        self.dataset = load_from_disk(dataset_path)['train']
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get input_ids and attention_mask
        input_ids = torch.tensor(item['input_ids'][:self.max_length], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'][:self.max_length], dtype=torch.long)
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For language modeling
        }

def train_step(model, batch, optimizer, scaler, device):
    """Single training step with safety checks."""
    model.train()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Check for NaN/Inf in inputs
    if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
        print("Warning: NaN/Inf detected in input_ids, skipping batch")
        return None
    
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
    
    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: NaN/Inf loss detected: {loss.item()}, skipping batch")
        return None
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Check for NaN/Inf in gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Warning: NaN/Inf gradient in {name}, skipping batch")
                return None
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning from LLaMA to MoVE')
    parser.add_argument('--llama_model', type=str, default='unsloth/Llama-3.2-1B-bnb-4bit',
                        help='LLaMA model name or path (using unsloth 4-bit quantized version)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to pre-tokenized dataset')
    parser.add_argument('--output_dir', type=str, default='./models/move_llama_transfer',
                        help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_steps', type=int, default=100,
                        help='Log training progress every N steps')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Initialize transfer learning
    transfer = LLaMAToMoVETransfer(args.llama_model)
    
    # Load LLaMA model
    llama_model, tokenizer = transfer.load_llama_model()
    
    # Create MoVE model with LLaMA-compatible config
    move_config = transfer.create_move_config_from_llama(llama_model.config)
    move_model = MoVELarge(**move_config)
    
    # Transfer weights from LLaMA to MoVE
    move_model = transfer.transfer_weights(llama_model, move_model)
    
    # Setup LoRA fine-tuning
    move_model = transfer.setup_lora_fine_tuning(move_model)
    
    # Move model to device
    move_model = move_model.to(device)
    
    # Clear LLaMA model from memory
    del llama_model
    torch.cuda.empty_cache()
    
    print(f"MoVE model created with {move_model.count_parameters()['total']:,} total parameters")
    print(f"Trainable parameters: {move_model.count_parameters()['trainable']:,}")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = PreTokenizedDataset(args.dataset_path, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(move_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(move_config, f, indent=2)
    
    # Training loop
    print("Starting transfer learning training...")
    global_step = 0
    total_loss = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = train_step(move_model, batch, optimizer, scaler, device)
            
            if loss is not None:
                total_loss += loss
                global_step += 1
                
                # Log progress
                if global_step % args.log_steps == 0:
                    avg_loss = total_loss / global_step
                    print(f"Step {global_step}, Average Loss: {avg_loss:.4f}, Current Loss: {loss:.4f}")
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    torch.save({
                        'model_state_dict': move_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'loss': loss,
                        'config': move_config
                    }, os.path.join(checkpoint_path, 'pytorch_model.bin'))
                    
                    print(f"Checkpoint saved at step {global_step}")
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model')
    os.makedirs(final_path, exist_ok=True)
    
    torch.save({
        'model_state_dict': move_model.state_dict(),
        'config': move_config,
        'tokenizer_config': tokenizer.get_vocab()
    }, os.path.join(final_path, 'pytorch_model.bin'))
    
    print(f"\nTraining completed! Final model saved to {final_path}")
    print(f"Total steps: {global_step}")
    print(f"Final average loss: {total_loss / max(global_step, 1):.4f}")

if __name__ == '__main__':
    main()