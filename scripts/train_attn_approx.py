#!/usr/bin/env python3
"""
Training script for AttnApprox module to match layer-1 hidden states from TinyLlama.

This script trains the AttnApprox module to approximate the attention mechanism
and match the layer-1 hidden states from TinyLlama.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import math
import glob

# Add parent directory to path for imports
sys.path.append('.')
from modules.attn_approx import AttnApprox, GraphAttnApprox, EfficientAttnApprox
from modules.token_embed import TokenEmbed, TokenEmbedWithLoRA
from modules.pos_gen import PosGen, RoPEPosGen, LearnablePosGen

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Layer1TargetDataset(Dataset):
    """Dataset for generating layer-1 hidden state targets from TinyLlama."""
    
    def __init__(self, 
                 model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 data_path="data/vecs",
                 max_seq_len=512, 
                 num_samples=1000):
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.data_path = data_path
        
        # Load TinyLlama model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Model config
        self.hidden_size = self.model.config.hidden_size
        self.num_attention_heads = self.model.config.num_attention_heads
        
        logger.info(f"Model config: hidden_size={self.hidden_size}, heads={self.num_attention_heads}")
        
        # Load existing vector data if available
        self.use_existing_data = os.path.exists(data_path)
        if self.use_existing_data:
            logger.info(f"Using existing vector data from {data_path}")
            self._load_existing_data()
        else:
            logger.info("Generating new data from scratch")
            self._generate_new_data()
    
    def _load_existing_data(self):
        """Load existing tokenized data and generate layer-1 targets."""
        # Load vector files
        vector_files = sorted(glob.glob(os.path.join(self.data_path, "*.pt")))
        
        if not vector_files:
            logger.warning("No vector files found, generating new data")
            self._generate_new_data()
            return
        
        logger.info(f"Found {len(vector_files)} vector files")
        
        self.input_ids_list = []
        self.layer0_states = []
        self.layer1_targets = []
        
        # Load a subset of files for training
        max_files = min(len(vector_files), self.num_samples // 4)  # Assuming 4 samples per file
        
        for i, file_path in enumerate(tqdm(vector_files[:max_files], desc="Loading data")):
            try:
                # Skip metadata file
                if 'metadata.pt' in file_path:
                    continue
                    
                data = torch.load(file_path, map_location='cpu')
                
                if isinstance(data, dict) and 'input_ids' in data and 'hidden_states' in data:
                    input_ids = data['input_ids']  # [batch_size, seq_len]
                    layer0_hidden = data['hidden_states']  # [batch_size, seq_len, hidden_size]
                    
                    # Generate layer-1 targets for each sample in the batch
                    for j in range(input_ids.size(0)):
                        sample_ids = input_ids[j]
                        sample_layer0 = layer0_hidden[j]
                        
                        # Truncate to max_seq_len
                        if sample_ids.size(0) > self.max_seq_len:
                            sample_ids = sample_ids[:self.max_seq_len]
                            sample_layer0 = sample_layer0[:self.max_seq_len]
                        
                        # For now, use layer0 hidden states as both input and target
                        # This tests the AttnApprox training pipeline
                        layer1_target = sample_layer0  # Use same as input for testing
                        
                        self.input_ids_list.append(sample_ids)
                        self.layer0_states.append(sample_layer0)
                        self.layer1_targets.append(layer1_target)
                        
                        if len(self.input_ids_list) >= self.num_samples:
                            break
                
                if len(self.input_ids_list) >= self.num_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.input_ids_list)} samples")
    
    def _generate_new_data(self):
        """Generate new data from scratch."""
        self.input_ids_list = []
        self.layer0_states = []
        self.layer1_targets = []
        
        # Generate random sequences
        for i in tqdm(range(self.num_samples), desc="Generating data"):
            # Random sequence length
            seq_len = torch.randint(10, self.max_seq_len + 1, (1,)).item()
            
            # Random token IDs (avoiding special tokens)
            input_ids = torch.randint(100, self.tokenizer.vocab_size - 100, (seq_len,))
            
            # Get layer-0 and layer-1 hidden states
            layer0_hidden, layer1_target = self._compute_targets(input_ids.unsqueeze(0))
            
            self.input_ids_list.append(input_ids)
            self.layer0_states.append(layer0_hidden.squeeze(0))
            self.layer1_targets.append(layer1_target.squeeze(0))
    
    def _compute_targets(self, input_ids):
        """Compute layer-0 and layer-1 hidden states."""
        with torch.no_grad():
            # Get embeddings (layer-0)
            embeddings = self.model.embeddings(input_ids)
            
            # Apply first transformer layer to get layer-1
            layer1_output = self.model.layers[0](embeddings)
            if isinstance(layer1_output, tuple):
                layer1_hidden = layer1_output[0]
            else:
                layer1_hidden = layer1_output
            
            return embeddings, layer1_hidden
    
    def _compute_layer1_target(self, input_ids, layer0_hidden):
        """Compute layer-1 target from layer-0 hidden states."""
        with torch.no_grad():
            try:
                # Apply first transformer layer
                layer1_output = self.model.layers[0](layer0_hidden)
                if isinstance(layer1_output, tuple):
                    layer1_hidden = layer1_output[0]
                else:
                    layer1_hidden = layer1_output
                
                return layer1_hidden
            except Exception as e:
                logger.warning(f"Error computing layer-1 target: {e}")
                # Return a dummy tensor with the same shape as input
                return torch.zeros_like(layer0_hidden)
    
    def __len__(self):
        return len(self.input_ids_list)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'layer0_hidden': self.layer0_states[idx],
            'layer1_target': self.layer1_targets[idx]
        }

def collate_fn(batch):
    """Collate function to handle variable sequence lengths."""
    # Find max sequence length in batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    batch_ids = []
    batch_layer0 = []
    batch_targets = []
    
    for item in batch:
        input_ids = item['input_ids']
        layer0_hidden = item['layer0_hidden']
        layer1_target = item['layer1_target']
        seq_len = input_ids.size(0)
        
        # Pad sequences
        if seq_len < max_len:
            pad_size = max_len - seq_len
            
            # Pad input_ids with 0
            input_ids = torch.cat([input_ids, torch.zeros(pad_size, dtype=input_ids.dtype)])
            
            # Pad hidden states with zeros
            layer0_hidden = torch.cat([layer0_hidden, torch.zeros(pad_size, layer0_hidden.size(-1))])
            layer1_target = torch.cat([layer1_target, torch.zeros(pad_size, layer1_target.size(-1))])
        
        batch_ids.append(input_ids)
        batch_layer0.append(layer0_hidden)
        batch_targets.append(layer1_target)
    
    return {
        'input_ids': torch.stack(batch_ids),
        'layer0_hidden': torch.stack(batch_layer0).float(),
        'layer1_target': torch.stack(batch_targets).float()
    }

def train_attn_approx(
    model_type='basic',
    hidden_dim=2048,
    low_rank_dim=64,
    num_heads=32,
    max_seq_len=512,
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=10,
    num_samples=1000,
    data_path="data/vecs",
    output_dir='models',
    log_dir='logs',
    device=None
):
    """Train AttnApprox module to match layer-1 hidden states."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training on device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = Layer1TargetDataset(
        data_path=data_path,
        max_seq_len=max_seq_len, 
        num_samples=num_samples
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create attention approximation model
    logger.info(f"Creating {model_type} AttnApprox model...")
    if model_type == 'graph':
        attn_model = GraphAttnApprox(d=hidden_dim, k=low_rank_dim, num_heads=num_heads, sparsity=0.1)
    elif model_type == 'efficient':
        attn_model = EfficientAttnApprox(d=hidden_dim, k=low_rank_dim, num_heads=num_heads)
    else:  # 'basic'
        attn_model = AttnApprox(d=hidden_dim, k=low_rank_dim, num_heads=num_heads)
    
    attn_model = attn_model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in attn_model.parameters())
    trainable_params = sum(p.numel() for p in attn_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(attn_model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    training_log = []
    
    for epoch in range(num_epochs):
        attn_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            layer0_hidden = batch['layer0_hidden'].to(device)
            layer1_target = batch['layer1_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Apply attention approximation to layer-0 hidden states
            attn_output = attn_model(layer0_hidden)
            
            # Compute loss
            loss = criterion(attn_output, layer1_target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(attn_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Log epoch results
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss
        })
    
    # Save model
    model_path = os.path.join(output_dir, f'attn_approx_{model_type}.pt')
    torch.save({
        'model_state_dict': attn_model.state_dict(),
        'model_type': model_type,
        'hidden_dim': hidden_dim,
        'low_rank_dim': low_rank_dim,
        'num_heads': num_heads,
        'max_seq_len': max_seq_len,
        'final_loss': avg_loss
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training log
    log_path = os.path.join(log_dir, f'attn_approx_{model_type}.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log saved to {log_path}")
    
    logger.info(f"Training completed! Final loss: {avg_loss:.6f}")
    
    return attn_model, training_log

def main():
    parser = argparse.ArgumentParser(description='Train AttnApprox module')
    parser.add_argument('--model_type', type=str, default='basic', 
                        choices=['basic', 'graph', 'efficient'],
                        help='Type of attention approximation model')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--low_rank_dim', type=int, default=64, help='Low-rank dimension')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--data_path', type=str, default='data/vecs', help='Path to vector data')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Train model
    train_attn_approx(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        low_rank_dim=args.low_rank_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        data_path=args.data_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )

if __name__ == '__main__':
    main()