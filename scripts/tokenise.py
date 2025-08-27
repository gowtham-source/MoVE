#!/usr/bin/env python3
"""Tokenization Script for MoVE Project

Tokenizes OpenWebText data using Llama-2 BPE tokenizer into 1024-token blocks.
This script implements Step 2 from the experimental roadmap.
"""

import os
import sys
from pathlib import Path
from datasets import load_from_disk
from transformers import LlamaTokenizerFast
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_tokenizer():
    """Load Llama tokenizer.
    
    Returns:
        LlamaTokenizerFast: The loaded tokenizer
    """
    try:
        logger.info("Loading Llama tokenizer...")
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            cache_dir=".cache/huggingface"
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Tokenizer loaded successfully")
        logger.info(f"Vocab size: {tokenizer.vocab_size}")
        logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        raise

def tokenize_function(examples: Dict[str, Any], tokenizer: LlamaTokenizerFast, max_length: int = 1024) -> Dict[str, Any]:
    """Tokenization function for dataset mapping.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dict containing tokenized inputs
    """
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll handle padding later if needed
        return_attention_mask=True,
        return_token_type_ids=False
    )
    
    return tokenized

def tokenize_dataset(
    input_path: str = "data/owt_1pct",
    output_path: str = "data/owt_1pct_tok",
    max_length: int = 1024,
    num_proc: int = 4
):
    """Tokenize the dataset.
    
    Args:
        input_path: Path to the raw dataset
        output_path: Path to save tokenized dataset
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel processing
    """
    try:
        logger.info(f"Starting tokenization process...")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Processes: {num_proc}")
        
        # Load the dataset
        logger.info("Loading raw dataset...")
        dataset = load_from_disk(input_path)
        logger.info(f"Dataset loaded. Size: {len(dataset)} examples")
        
        # Load tokenizer
        tokenizer = load_tokenizer()
        
        # Create tokenization function with fixed parameters
        def tok_fn(examples):
            return tokenize_function(examples, tokenizer, max_length)
        
        # Apply tokenization
        logger.info("Starting tokenization...")
        tokenized_dataset = dataset.map(
            tok_fn,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Tokenizing"
        )
        
        logger.info(f"Tokenization completed. Size: {len(tokenized_dataset)} examples")
        
        # Save tokenized dataset
        logger.info(f"Saving tokenized dataset to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        tokenized_dataset.save_to_disk(output_path)
        
        logger.info("Tokenized dataset saved successfully!")
        
        # Print statistics
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
            logger.info(f"Sample input_ids: {sample['input_ids'][:20]}...")
            
            # Decode sample for verification
            decoded_sample = tokenizer.decode(sample['input_ids'][:50])
            logger.info(f"Decoded sample: {decoded_sample}")
        
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        raise

def verify_tokenized_dataset(dataset_path: str = "data/owt_1pct_tok"):
    """Verify the tokenized dataset.
    
    Args:
        dataset_path: Path to the tokenized dataset
    """
    try:
        logger.info(f"Verifying tokenized dataset at {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        
        logger.info(f"Dataset verification successful!")
        logger.info(f"Total examples: {len(dataset)}")
        logger.info(f"Features: {dataset.features}")
        
        # Check sequence lengths
        if len(dataset) > 0:
            lengths = [len(example['input_ids']) for example in dataset.select(range(min(100, len(dataset))))]
            logger.info(f"Sequence length stats (first 100 examples):")
            logger.info(f"  Min: {min(lengths)}")
            logger.info(f"  Max: {max(lengths)}")
            logger.info(f"  Avg: {sum(lengths) / len(lengths):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset verification failed: {str(e)}")
        return False

def main():
    """Main function to tokenize and verify the dataset."""
    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        logger.info("Starting MoVE tokenization process...")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check if raw dataset exists
        if not os.path.exists("data/owt_1pct"):
            logger.error("Raw dataset not found! Please run scripts/download_data.py first.")
            sys.exit(1)
        
        # Tokenize dataset
        tokenized_dataset = tokenize_dataset()
        
        # Verify tokenized dataset
        if verify_tokenized_dataset():
            logger.info("✅ Tokenization completed successfully!")
            logger.info("Next step: Download baseline model and verify VRAM usage")
        else:
            logger.error("❌ Tokenized dataset verification failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Tokenization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()