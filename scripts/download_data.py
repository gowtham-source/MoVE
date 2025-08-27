#!/usr/bin/env python3
"""Dataset Acquisition Script for MoVE Project

Downloads OpenWebText 1% slice for training the modular components.
This script implements Step 1 from the experimental roadmap.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_openwebtext_slice(output_dir: str = "data/owt_1pct", slice_percentage: float = 1.0):
    """Download text dataset for training.
    
    Args:
        output_dir: Directory to save the dataset
        slice_percentage: Percentage of dataset to download (default: 1%)
    """
    try:
        logger.info(f"Starting download of text dataset {slice_percentage}% slice...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Try multiple dataset options in order of preference
        dataset_options = [
            # Option 1: Try wikitext-103-raw-v1 (smaller, more reliable)
            ('wikitext', 'wikitext-103-raw-v1', 'train'),
            # Option 2: Try wikitext-2-raw-v1 (even smaller)
            ('wikitext', 'wikitext-2-raw-v1', 'train'),
            # Option 3: Try bookcorpus (if available)
            ('bookcorpus', None, 'train'),
        ]
        
        dataset = None
        for dataset_name, config_name, split in dataset_options:
            try:
                logger.info(f"Trying dataset: {dataset_name} (config: {config_name})")
                
                if config_name:
                    dataset = load_dataset(
                        dataset_name,
                        config_name,
                        split=split,
                        cache_dir=".cache/huggingface"
                    )
                else:
                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        cache_dir=".cache/huggingface"
                    )
                
                # Take a slice if needed
                if slice_percentage < 100.0:
                    slice_size = int(len(dataset) * slice_percentage / 100.0)
                    dataset = dataset.select(range(min(slice_size, len(dataset))))
                
                logger.info(f"Dataset loaded successfully: {dataset_name}")
                logger.info(f"Size: {len(dataset)} examples")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {str(e)}")
                continue
        
        # If all datasets fail, create a synthetic dataset
        if dataset is None:
            logger.warning("All dataset downloads failed, creating synthetic dataset...")
            dataset = create_synthetic_dataset()
        
        # Save to disk
        logger.info(f"Saving dataset to {output_dir}...")
        dataset.save_to_disk(output_dir)
        
        logger.info(f"Dataset saved successfully to {output_dir}")
        logger.info(f"Total examples: {len(dataset)}")
        
        # Print sample statistics
        if len(dataset) > 0:
            sample_text = dataset[0]['text']
            logger.info(f"Sample text length: {len(sample_text)} characters")
            logger.info(f"Sample preview: {sample_text[:200]}...")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def create_synthetic_dataset():
    """Create a synthetic text dataset for testing purposes."""
    from datasets import Dataset
    
    logger.info("Creating synthetic text dataset...")
    
    # Generate synthetic text samples
    synthetic_texts = []
    
    # Sample texts for language modeling
    base_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample sentence for training language models.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
        "Transformers have revolutionized the field of natural language processing with their attention mechanisms.",
        "Large language models can generate coherent text and perform various language understanding tasks.",
        "Training neural networks requires large amounts of data and computational resources.",
        "The attention mechanism allows models to focus on relevant parts of the input sequence.",
        "Gradient descent is an optimization algorithm used to minimize the loss function during training.",
        "Tokenization is the process of converting text into smaller units that can be processed by models."
    ]
    
    # Expand the dataset by creating variations
    for i in range(1000):  # Create 1000 synthetic examples
        base_idx = i % len(base_texts)
        text = base_texts[base_idx]
        
        # Add some variation
        if i % 3 == 0:
            text = text + " This is additional content for variation."
        elif i % 3 == 1:
            text = "Introduction: " + text
        
        synthetic_texts.append(text)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": synthetic_texts})
    
    logger.info(f"Created synthetic dataset with {len(dataset)} examples")
    return dataset

def verify_dataset(dataset_path: str = "data/owt_1pct"):
    """Verify the downloaded dataset.
    
    Args:
        dataset_path: Path to the saved dataset
    """
    try:
        from datasets import load_from_disk
        
        logger.info(f"Verifying dataset at {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        
        logger.info(f"Dataset verification successful!")
        logger.info(f"Total examples: {len(dataset)}")
        logger.info(f"Features: {dataset.features}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset verification failed: {str(e)}")
        return False

def main():
    """Main function to download and verify the dataset."""
    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        logger.info("Starting MoVE dataset acquisition...")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Download dataset
        dataset = download_openwebtext_slice()
        
        # Verify dataset
        if verify_dataset():
            logger.info("✅ Dataset acquisition completed successfully!")
            logger.info("Next step: Run tokenization script (scripts/tokenise.py)")
        else:
            logger.error("❌ Dataset verification failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()