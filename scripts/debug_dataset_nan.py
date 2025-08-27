#!/usr/bin/env python3
"""
Dataset Debugging Script for NaN Issues

Investigates the ArXiv dataset to identify samples that might cause NaN losses.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np

def analyze_text_sample(text, tokenizer, max_length=512):
    """Analyze a single text sample for potential issues."""
    issues = []
    
    # Check text properties
    if not text or len(text.strip()) == 0:
        issues.append("Empty text")
        return issues, None
    
    if len(text) > 100000:  # Very long text
        issues.append(f"Very long text: {len(text)} chars")
    
    # Check for unusual characters
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    if non_ascii_count > len(text) * 0.5:
        issues.append(f"High non-ASCII ratio: {non_ascii_count}/{len(text)}")
    
    # Tokenize
    try:
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Check tokenization issues
        if torch.isnan(input_ids.float()).any():
            issues.append("NaN in input_ids")
        
        if torch.isinf(input_ids.float()).any():
            issues.append("Inf in input_ids")
        
        # Check for unusual token patterns
        unique_tokens = len(torch.unique(input_ids))
        if unique_tokens < 5:  # Very few unique tokens
            issues.append(f"Few unique tokens: {unique_tokens}")
        
        # Check for extreme token values
        max_token = input_ids.max().item()
        if max_token >= tokenizer.vocab_size:
            issues.append(f"Token out of vocab: {max_token} >= {tokenizer.vocab_size}")
        
        return issues, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_length': len(text),
            'token_count': attention_mask.sum().item(),
            'unique_tokens': unique_tokens
        }
        
    except Exception as e:
        issues.append(f"Tokenization error: {str(e)}")
        return issues, None

def test_loss_calculation(input_ids, vocab_size):
    """Test if input_ids would cause NaN in loss calculation."""
    try:
        # Create dummy logits
        seq_len = input_ids.size(0)
        logits = torch.randn(1, seq_len, vocab_size) * 0.1  # Small random logits
        
        # Prepare labels
        labels = input_ids.clone().unsqueeze(0)
        
        # Calculate loss like in training
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        if torch.isnan(loss) or torch.isinf(loss):
            return f"Loss calculation produces NaN/Inf: {loss.item()}"
        
        return None
        
    except Exception as e:
        return f"Loss calculation error: {str(e)}"

def main():
    print("Debugging ArXiv dataset for NaN issues...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load dataset
    dataset_path = "data/arxiv_dataset_500m"
    try:
        dataset = load_from_disk(dataset_path)
        train_data = dataset['train']
        print(f"Loaded dataset with {len(train_data)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # Analyze samples
    problematic_samples = []
    total_issues = 0
    
    print("\nAnalyzing first 1000 samples...")
    
    for i in range(min(1000, len(train_data))):
        if i % 100 == 0:
            print(f"Processed {i}/1000 samples...")
        
        try:
            item = train_data[i]
            text = item.get('text', item.get('content', ''))
            
            issues, token_info = analyze_text_sample(text, tokenizer)
            
            if issues:
                total_issues += len(issues)
                problematic_samples.append({
                    'index': i,
                    'issues': issues,
                    'text_preview': text[:200] + "..." if len(text) > 200 else text,
                    'token_info': token_info
                })
            
            # Test loss calculation if tokenization succeeded
            if token_info:
                loss_issue = test_loss_calculation(token_info['input_ids'], len(tokenizer))
                if loss_issue:
                    problematic_samples.append({
                        'index': i,
                        'issues': [loss_issue],
                        'text_preview': text[:200] + "..." if len(text) > 200 else text,
                        'token_info': token_info
                    })
                    total_issues += 1
        
        except Exception as e:
            problematic_samples.append({
                'index': i,
                'issues': [f"Processing error: {str(e)}"],
                'text_preview': "Error accessing sample",
                'token_info': None
            })
            total_issues += 1
    
    # Report results
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Total samples analyzed: {min(1000, len(train_data))}")
    print(f"Problematic samples found: {len(problematic_samples)}")
    print(f"Total issues: {total_issues}")
    
    if problematic_samples:
        print(f"\n=== TOP 10 PROBLEMATIC SAMPLES ===")
        for i, sample in enumerate(problematic_samples[:10]):
            print(f"\nSample {sample['index']}:")
            print(f"  Issues: {sample['issues']}")
            print(f"  Text preview: {sample['text_preview']}")
            if sample['token_info']:
                info = sample['token_info']
                print(f"  Token info: {info['token_count']} tokens, {info['unique_tokens']} unique")
    
    # Test with a simple, clean sample
    print(f"\n=== TESTING WITH CLEAN SAMPLE ===")
    clean_text = "This is a simple test sentence for debugging purposes."
    issues, token_info = analyze_text_sample(clean_text, tokenizer)
    print(f"Clean sample issues: {issues}")
    
    if token_info:
        loss_issue = test_loss_calculation(token_info['input_ids'], len(tokenizer))
        print(f"Clean sample loss test: {loss_issue or 'OK'}")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    if len(problematic_samples) > len(train_data) * 0.1:
        print("- High percentage of problematic samples detected")
        print("- Consider dataset preprocessing or filtering")
    
    if any("Empty text" in str(sample['issues']) for sample in problematic_samples):
        print("- Remove empty text samples")
    
    if any("Very long text" in str(sample['issues']) for sample in problematic_samples):
        print("- Consider truncating very long texts before tokenization")
    
    if any("Token out of vocab" in str(sample['issues']) for sample in problematic_samples):
        print("- Tokenizer vocabulary mismatch detected")
        print("- Check tokenizer compatibility with dataset")
    
    print("\nDebugging completed!")

if __name__ == '__main__':
    main()