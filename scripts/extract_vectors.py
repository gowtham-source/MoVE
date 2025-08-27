#!/usr/bin/env python3
"""
Extract Layer-0 Vectors from TinyLlama for Modular Training Targets

This script extracts hidden states from the first layer of TinyLlama
to serve as training targets for MoVE modular components.
"""

import os
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_layer_vectors():
    """Extract Layer-0 vectors from TinyLlama model."""
    
    # Create output directory
    os.makedirs("data/vecs", exist_ok=True)
    
    # Load tokenizer and model
    logger.info("Loading TinyLlama model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load tokenized dataset
    logger.info("Loading tokenized dataset...")
    try:
        ds = datasets.load_from_disk("data/owt_1pct_tok")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Please run tokenization script first: python scripts/tokenise.py")
        return
    
    # Define collate function for padding
    def collate_fn(batch):
        # Pad sequences to the same length
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        attention_masks = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            pad_len = max_len - seq_len
            
            # Pad input_ids with tokenizer.pad_token_id (or 0 if not available)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded_input_ids = item['input_ids'] + [pad_token_id] * pad_len
            
            # Pad attention_mask with 0s
            padded_attention_mask = item['attention_mask'] + [0] * pad_len
            
            input_ids.append(padded_input_ids)
            attention_masks.append(padded_attention_mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
    
    # Create dataloader with custom collate function
    dataloader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    logger.info("Extracting Layer-0 vectors...")
    model.eval()
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Extracting vectors")):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with hidden states output
            try:
                outputs = model(**batch, output_hidden_states=True)
                
                # Extract Layer-0 hidden states (after embedding + positional encoding)
                layer_0_states = outputs.hidden_states[0]  # [batch_size, seq_len, hidden_dim]
                
                # Save to disk
                output_path = f"data/vecs/l0_step{step:04d}.pt"
                torch.save({
                    'hidden_states': layer_0_states.cpu(),
                    'input_ids': batch['input_ids'].cpu(),
                    'attention_mask': batch['attention_mask'].cpu() if 'attention_mask' in batch else None,
                    'step': step
                }, output_path)
                
                if step % 10 == 0:
                    logger.info(f"Saved step {step}, shape: {layer_0_states.shape}")
                    
            except Exception as e:
                logger.error(f"Error processing step {step}: {e}")
                continue
            
            # Stop after 100 steps as specified
            if step >= 100:
                break
    
    logger.info(f"Vector extraction complete! Saved {step + 1} batches to data/vecs/")
    
    # Save metadata
    metadata = {
        'total_steps': step + 1,
        'batch_size': 4,
        'model_name': 'tiny_llama',
        'layer_extracted': 0,
        'hidden_dim': layer_0_states.shape[-1],
        'seq_len': layer_0_states.shape[1]
    }
    torch.save(metadata, "data/vecs/metadata.pt")
    logger.info(f"Saved metadata: {metadata}")

def verify_extracted_vectors():
    """Verify the extracted vectors."""
    logger.info("Verifying extracted vectors...")
    
    # Check if vectors directory exists
    if not os.path.exists("data/vecs"):
        logger.error("Vectors directory not found. Run extraction first.")
        return
    
    # Load metadata
    try:
        metadata = torch.load("data/vecs/metadata.pt")
        logger.info(f"Metadata: {metadata}")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return
    
    # Check a few vector files
    for step in [0, 10, 50]:
        file_path = f"data/vecs/l0_step{step:04d}.pt"
        if os.path.exists(file_path):
            data = torch.load(file_path)
            logger.info(f"Step {step}: hidden_states shape = {data['hidden_states'].shape}")
        else:
            logger.warning(f"Step {step} file not found: {file_path}")
    
    logger.info("Verification complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Layer-0 vectors from TinyLlama")
    parser.add_argument("--verify", action="store_true", help="Verify extracted vectors")
    args = parser.parse_args()
    
    if args.verify:
        verify_extracted_vectors()
    else:
        extract_layer_vectors()