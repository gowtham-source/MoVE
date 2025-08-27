#!/usr/bin/env python3
"""
Perplexity Evaluation Script for MoVE vs TinyLlama

This script evaluates the perplexity of MoVE model against TinyLlama baseline
with the goal of achieving performance within 5% of the baseline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from move.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location("move_module", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "move.py"))
move_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(move_module)
MoVE = move_module.MoVE
create_move_model = move_module.create_move_model

class EvalDataset:
    """Dataset for evaluation using real tokenized data."""
    
    def __init__(self, data_path, max_length=1024, split='validation'):
        self.max_length = max_length
        
        # Load tokenized dataset
        if os.path.exists(data_path):
            print(f"Loading dataset from {data_path}...")
            self.dataset = datasets.load_from_disk(data_path)
            
            # Create validation split if it doesn't exist
            if hasattr(self.dataset, 'train_test_split'):
                splits = self.dataset.train_test_split(test_size=0.1, seed=42)
                if split == 'validation':
                    self.dataset = splits['test']
                else:
                    self.dataset = splits['train']
            print(f"Using {len(self.dataset)} examples for {split}")
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_data.py and tokenise.py first.")
    
    def __len__(self):
        if hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        return len(self.dataset['input_ids'])
    
    def __getitem__(self, idx):
        if hasattr(self.dataset, '__getitem__'):
            item = self.dataset[idx]
        else:
            item = {
                'input_ids': self.dataset['input_ids'][idx],
                'attention_mask': self.dataset['attention_mask'][idx]
            }
        
        # Ensure proper length
        input_ids = item['input_ids'][:self.max_length]
        attention_mask = item.get('attention_mask', [1] * len(input_ids))[:self.max_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([0] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def compute_perplexity_detailed(model, dataloader, device, max_steps=None, model_name="Model"):
    """Compute detailed perplexity metrics.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        max_steps: Maximum number of steps to evaluate
        model_name: Name of the model for logging
        
    Returns:
        dict: Detailed metrics including perplexity, loss, and token counts
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    batch_losses = []
    
    print(f"Evaluating {model_name}...")
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating {model_name}")
        
        for i, batch in enumerate(progress_bar):
            if max_steps and i >= max_steps:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift for language modeling
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = attention_mask[:, 1:].contiguous()
            
            try:
                # Forward pass
                if hasattr(model, 'forward') and 'MoVE' in str(type(model)):
                    # MoVE model
                    outputs = model(inputs)
                    logits = outputs
                else:
                    # Transformers model
                    outputs = model(inputs)
                    logits = outputs.logits
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                losses = losses.view(targets.shape)
                
                # Apply mask
                masked_losses = losses * mask.float()
                batch_loss = masked_losses.sum().item()
                batch_tokens = mask.sum().item()
                
                if batch_tokens > 0:
                    total_loss += batch_loss
                    total_tokens += batch_tokens
                    batch_losses.append(batch_loss / batch_tokens)
                    total_batches += 1
                
                # Update progress bar
                if total_tokens > 0:
                    current_ppl = math.exp(total_loss / total_tokens)
                    progress_bar.set_postfix({'PPL': f'{current_ppl:.2f}'})
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
    
    if total_tokens == 0:
        return {
            'perplexity': float('inf'),
            'avg_loss': float('inf'),
            'total_tokens': 0,
            'total_batches': 0,
            'batch_losses': []
        }
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
        'total_batches': total_batches,
        'batch_losses': batch_losses,
        'std_loss': torch.std(torch.tensor(batch_losses)).item() if batch_losses else 0.0
    }

def load_move_model(model_path, device):
    """Load MoVE model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load on
        
    Returns:
        MoVE: Loaded model
    """
    print(f"Loading MoVE model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = MoVE(**config)
    else:
        # Use default config
        print("No config found in checkpoint, using default 'small' config")
        model = create_move_model('small')
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"MoVE model loaded with {model.get_num_params():,} parameters")
    
    return model

def load_baseline_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="auto"):
    """Load baseline TinyLlama model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load on
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading baseline model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device if device != "auto" else "auto"
        )
        
        if device != "auto":
            model = model.to(device)
        
        model.eval()
        
        print(f"Baseline model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        print("Creating dummy baseline model for testing...")
        
        # Create a dummy model for testing
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(2048, 32000)
            
            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                hidden = torch.randn(batch_size, seq_len, 2048, device=input_ids.device)
                logits = self.lm_head(hidden)
                
                class Output:
                    def __init__(self, logits):
                        self.logits = logits
                
                return Output(logits)
        
        dummy_model = DummyModel().to(device)
        return dummy_model, None

def compare_models(move_model, baseline_model, dataloader, device, max_steps=50):
    """Compare MoVE and baseline models.
    
    Args:
        move_model: MoVE model
        baseline_model: Baseline model
        dataloader: Evaluation dataloader
        device: Device to run on
        max_steps: Maximum evaluation steps
        
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*60)
    print("PERPLEXITY COMPARISON: MoVE vs TinyLlama")
    print("="*60)
    
    # Evaluate MoVE model
    move_results = compute_perplexity_detailed(
        move_model, dataloader, device, max_steps, "MoVE"
    )
    
    # Evaluate baseline model
    baseline_results = compute_perplexity_detailed(
        baseline_model, dataloader, device, max_steps, "TinyLlama"
    )
    
    # Compute comparison metrics
    move_ppl = move_results['perplexity']
    baseline_ppl = baseline_results['perplexity']
    
    if baseline_ppl > 0 and baseline_ppl != float('inf'):
        ppl_ratio = move_ppl / baseline_ppl
        ppl_diff_percent = ((move_ppl - baseline_ppl) / baseline_ppl) * 100
        within_5_percent = abs(ppl_diff_percent) <= 5.0
    else:
        ppl_ratio = float('inf')
        ppl_diff_percent = float('inf')
        within_5_percent = False
    
    # Print results
    print(f"\nRESULTS:")
    print(f"{'Model':<15} {'Perplexity':<12} {'Avg Loss':<10} {'Tokens':<10} {'Batches':<8}")
    print("-" * 60)
    print(f"{'MoVE':<15} {move_ppl:<12.2f} {move_results['avg_loss']:<10.4f} {move_results['total_tokens']:<10} {move_results['total_batches']:<8}")
    print(f"{'TinyLlama':<15} {baseline_ppl:<12.2f} {baseline_results['avg_loss']:<10.4f} {baseline_results['total_tokens']:<10} {baseline_results['total_batches']:<8}")
    
    print(f"\nCOMPARISON:")
    print(f"Perplexity Ratio (MoVE/TinyLlama): {ppl_ratio:.4f}")
    print(f"Perplexity Difference: {ppl_diff_percent:+.2f}%")
    print(f"Within 5% of baseline: {'✓ YES' if within_5_percent else '✗ NO'}")
    
    if within_5_percent:
        print(f"\n🎉 SUCCESS: MoVE achieves perplexity within 5% of TinyLlama!")
    else:
        print(f"\n⚠️  MoVE perplexity is {abs(ppl_diff_percent):.1f}% {'higher' if ppl_diff_percent > 0 else 'lower'} than baseline")
        if ppl_diff_percent > 5:
            print(f"   Consider: more training, better hyperparameters, or model architecture improvements")
    
    return {
        'move_results': move_results,
        'baseline_results': baseline_results,
        'ppl_ratio': ppl_ratio,
        'ppl_diff_percent': ppl_diff_percent,
        'within_5_percent': within_5_percent,
        'success': within_5_percent
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate MoVE vs TinyLlama perplexity')
    parser.add_argument('--model_path', required=True, help='Path to MoVE model checkpoint')
    parser.add_argument('--eval_split', default='data/owt_1pct_tok', help='Path to evaluation dataset')
    parser.add_argument('--baseline_model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Baseline model name')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum evaluation steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--seq_length', type=int, default=256, help='Sequence length')
    parser.add_argument('--output_file', default='perplexity_comparison.json', help='Output file for results (JSON)')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load models
    try:
        move_model = load_move_model(args.model_path, device)
    except Exception as e:
        print(f"Error loading MoVE model: {e}")
        return
    
    try:
        baseline_model, tokenizer = load_baseline_model(args.baseline_model, device)
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        return
    
    # Setup evaluation dataset
    eval_dataset = EvalDataset(args.eval_split, max_length=args.seq_length, split='validation')
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Evaluation dataset: {len(eval_dataset)} samples")
    
    # Run comparison
    start_time = time.time()
    results = compare_models(move_model, baseline_model, eval_dataloader, device, args.max_steps)
    eval_time = time.time() - start_time
    
    # Add timing info
    results['evaluation_time'] = eval_time
    results['args'] = vars(args)
    
    print(f"\nEvaluation completed in {eval_time:.2f} seconds")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output_file}")
    
    # Return success code
    return 0 if results['success'] else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)