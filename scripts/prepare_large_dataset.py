"""Large Dataset Preparation Script

Prepares large-scale datasets for training:
- Common Crawl (C4)
- OpenWebText
- RedPajama
- The Pile
- Custom dataset combinations

Optimized for efficient loading and preprocessing.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class LargeDatasetPreprocessor:
    """Preprocessor for large-scale datasets."""
    
    def __init__(self, tokenizer_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0', max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loaded tokenizer: {tokenizer_name}")
        print(f"Vocab size: {len(self.tokenizer)}")
        print(f"Max length: {max_length}")
    
    def tokenize_function(self, examples):
        """Tokenize text examples."""
        # Handle different text column names
        text_column = None
        for col in ['text', 'content', 'article', 'document']:
            if col in examples:
                text_column = col
                break
        
        if text_column is None:
            raise ValueError(f"No text column found. Available columns: {list(examples.keys())}")
        
        # Tokenize
        tokenized = self.tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_attention_mask=False
        )
        
        return tokenized
    
    def group_texts(self, examples):
        """Group texts into chunks of max_length."""
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # Split into chunks
        result = {k: [] for k in concatenated.keys()}
        
        for i in range(0, total_length, self.max_length):
            for k in concatenated.keys():
                chunk = concatenated[k][i:i + self.max_length]
                if len(chunk) == self.max_length:  # Only keep full chunks
                    result[k].append(chunk)
        
        return result

def download_c4_dataset(split='train', streaming=False, num_proc=4):
    """Download and prepare C4 dataset."""
    print("Downloading C4 dataset...")
    
    # C4 is very large, so we'll use a subset
    dataset = load_dataset(
        'c4',
        'en',
        split=f'{split}[:100000]' if not streaming else split,  # Limit to 100k examples for demo
        streaming=streaming
    )
    
    print(f"C4 dataset loaded: {len(dataset) if not streaming else 'streaming'} examples")
    return dataset

def download_openwebtext_dataset(split='train', num_proc=4):
    """Download and prepare OpenWebText dataset."""
    print("Downloading OpenWebText dataset...")
    
    dataset = load_dataset('openwebtext', split=split)
    print(f"OpenWebText dataset loaded: {len(dataset)} examples")
    return dataset

def download_redpajama_dataset(split='train', subset='sample', num_proc=4):
    """Download and prepare RedPajama dataset."""
    print("Downloading RedPajama dataset...")
    
    # RedPajama is very large, use sample subset
    dataset = load_dataset(
        'togethercomputer/RedPajama-Data-1T-Sample',
        split=split
    )
    
    print(f"RedPajama dataset loaded: {len(dataset)} examples")
    return dataset

def download_pile_dataset(split='train', subset='all', num_proc=4):
    """Download and prepare The Pile dataset."""
    print("Downloading The Pile dataset...")
    
    # The Pile is very large, use a subset
    dataset = load_dataset(
        'the_pile',
        split=f'{split}[:50000]'  # Limit to 50k examples
    )
    
    print(f"The Pile dataset loaded: {len(dataset)} examples")
    return dataset

def download_custom_datasets(dataset_configs: List[Dict[str, Any]], num_proc=4):
    """Download custom datasets based on configuration."""
    datasets_list = []
    
    for config in dataset_configs:
        name = config['name']
        path = config['path']
        split = config.get('split', 'train')
        subset = config.get('subset', None)
        limit = config.get('limit', None)
        
        print(f"Loading {name} dataset...")
        
        try:
            if subset:
                dataset = load_dataset(path, subset, split=split)
            else:
                dataset = load_dataset(path, split=split)
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            datasets_list.append(dataset)
            print(f"{name} loaded: {len(dataset)} examples")
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
    
    return datasets_list

def prepare_large_dataset(
    output_dir='data/large_dataset',
    datasets_to_use=['c4'],
    tokenizer_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    max_length=2048,
    num_proc=4,
    test_size=0.01,
    custom_configs=None,
    streaming=False
):
    """Prepare large-scale dataset for training."""
    
    print(f"Preparing large dataset with: {datasets_to_use}")
    print(f"Output directory: {output_dir}")
    print(f"Max length: {max_length}")
    print(f"Number of processes: {num_proc}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = LargeDatasetPreprocessor(tokenizer_name, max_length)
    
    # Download datasets
    datasets_list = []
    
    for dataset_name in datasets_to_use:
        if dataset_name == 'c4':
            dataset = download_c4_dataset(streaming=streaming, num_proc=num_proc)
        elif dataset_name == 'openwebtext':
            dataset = download_openwebtext_dataset(num_proc=num_proc)
        elif dataset_name == 'redpajama':
            dataset = download_redpajama_dataset(num_proc=num_proc)
        elif dataset_name == 'pile':
            dataset = download_pile_dataset(num_proc=num_proc)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        datasets_list.append(dataset)
    
    # Add custom datasets if provided
    if custom_configs:
        custom_datasets = download_custom_datasets(custom_configs, num_proc)
        datasets_list.extend(custom_datasets)
    
    if not datasets_list:
        raise ValueError("No datasets loaded successfully")
    
    # Combine datasets
    if len(datasets_list) > 1:
        print("Combining datasets...")
        combined_dataset = concatenate_datasets(datasets_list)
    else:
        combined_dataset = datasets_list[0]
    
    print(f"Combined dataset size: {len(combined_dataset)} examples")
    
    # Shuffle dataset
    print("Shuffling dataset...")
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = combined_dataset.map(
        preprocessor.tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=combined_dataset.column_names,
        desc="Tokenizing"
    )
    
    # Group texts into chunks
    print("Grouping texts into chunks...")
    grouped_dataset = tokenized_dataset.map(
        preprocessor.group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts"
    )
    
    # Split into train/validation
    print(f"Splitting dataset (test_size={test_size})...")
    split_dataset = grouped_dataset.train_test_split(
        test_size=test_size,
        seed=42
    )
    
    # Save dataset
    print(f"Saving dataset to {output_dir}...")
    split_dataset.save_to_disk(output_dir)
    
    # Save metadata
    metadata = {
        'datasets_used': datasets_to_use,
        'tokenizer': tokenizer_name,
        'max_length': max_length,
        'vocab_size': len(preprocessor.tokenizer),
        'train_size': len(split_dataset['train']),
        'validation_size': len(split_dataset['test']),
        'total_tokens_train': len(split_dataset['train']) * max_length,
        'total_tokens_validation': len(split_dataset['test']) * max_length
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset preparation completed!")
    print(f"Train examples: {metadata['train_size']:,}")
    print(f"Validation examples: {metadata['validation_size']:,}")
    print(f"Total training tokens: {metadata['total_tokens_train']:,}")
    print(f"Total validation tokens: {metadata['total_tokens_validation']:,}")
    
    return split_dataset

def estimate_training_time(dataset_size, tokens_per_example, batch_size, 
                          gradient_accumulation_steps, throughput_tokens_per_sec=1000):
    """Estimate training time based on dataset size and throughput."""
    
    total_tokens = dataset_size * tokens_per_example
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Estimate steps
    steps_per_epoch = dataset_size // effective_batch_size
    
    # Estimate time
    tokens_per_step = effective_batch_size * tokens_per_example
    seconds_per_step = tokens_per_step / throughput_tokens_per_sec
    total_seconds = steps_per_epoch * seconds_per_step
    
    hours = total_seconds / 3600
    
    print(f"\nTraining Time Estimation:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Estimated time per epoch: {hours:.2f} hours")
    print(f"  Tokens per second: {throughput_tokens_per_sec:,}")
    
    return hours

def main():
    parser = argparse.ArgumentParser(description='Prepare large dataset for training')
    
    parser.add_argument('--output_dir', type=str, default='data/large_dataset',
                       help='Output directory for processed dataset')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['c4', 'openwebtext', 'redpajama', 'pile'],
                       default=['c4'],
                       help='Datasets to use')
    parser.add_argument('--tokenizer', type=str, 
                       default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Tokenizer to use')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--num_proc', type=int, default=4,
                       help='Number of processes for data processing')
    parser.add_argument('--test_size', type=float, default=0.01,
                       help='Fraction of data to use for validation')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming for very large datasets')
    parser.add_argument('--custom_config', type=str, default=None,
                       help='Path to JSON file with custom dataset configurations')
    
    # Training estimation
    parser.add_argument('--estimate_time', action='store_true',
                       help='Estimate training time')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for time estimation')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                       help='Gradient accumulation steps for time estimation')
    parser.add_argument('--throughput', type=int, default=1000,
                       help='Estimated tokens per second throughput')
    
    args = parser.parse_args()
    
    # Load custom configurations if provided
    custom_configs = None
    if args.custom_config and os.path.exists(args.custom_config):
        with open(args.custom_config, 'r') as f:
            custom_configs = json.load(f)
    
    # Prepare dataset
    dataset = prepare_large_dataset(
        output_dir=args.output_dir,
        datasets_to_use=args.datasets,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        test_size=args.test_size,
        custom_configs=custom_configs,
        streaming=args.streaming
    )
    
    # Estimate training time if requested
    if args.estimate_time:
        estimate_training_time(
            dataset_size=len(dataset['train']),
            tokens_per_example=args.max_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            throughput_tokens_per_sec=args.throughput
        )

if __name__ == '__main__':
    main()