#!/usr/bin/env python3
"""
ArXiv Dataset Preparation Script for MoVE 1B Model

Processes Common Pile ArXiv datasets:
- common-pile/arxiv_abstracts: ArXiv paper abstracts (CC0 licensed)
- common-pile/arxiv_papers: Full ArXiv papers (openly licensed)

Optimized for 1B parameter MoVE model training on RTX 4090.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

import torch
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class ArXivDatasetProcessor:
    """Processor for ArXiv datasets from Common Pile."""
    
    def __init__(self, tokenizer_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0', max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = max_length
        self.vocab_size = len(self.tokenizer)
        
        print(f"Initialized ArXiv processor with tokenizer: {tokenizer_name}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max sequence length: {max_length}")
    
    def clean_arxiv_text(self, text: str) -> str:
        """Clean ArXiv text content."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove LaTeX commands (basic cleanup)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove mathematical expressions in brackets
        text = re.sub(r'\$[^$]*\$', '', text)
        text = re.sub(r'\\\([^)]*\\\)', '', text)
        text = re.sub(r'\\\[[^]]*\\\]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def process_arxiv_abstract(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Process a single ArXiv abstract example."""
        # Extract text content
        text = example.get('text', '')
        
        # Parse metadata if available
        metadata = {}
        if 'meta' in example and example['meta']:
            try:
                if isinstance(example['meta'], str):
                    metadata = json.loads(example['meta'])
                else:
                    metadata = example['meta']
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        # Clean the abstract text
        cleaned_text = self.clean_arxiv_text(text)
        
        # Format as structured text for training
        if cleaned_text:
            # Add context markers for abstracts
            formatted_text = f"Abstract: {cleaned_text}"
            
            # Add subject information if available
            if 'subjects' in metadata:
                subjects = metadata.get('subjects', '')
                if subjects:
                    formatted_text = f"Subject: {subjects}\n{formatted_text}"
            
            return {'text': formatted_text}
        
        return {'text': ''}
    
    def process_arxiv_paper(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Process a single ArXiv full paper example."""
        # Extract text content
        text = example.get('text', '')
        
        # Parse metadata if available
        metadata = {}
        if 'meta' in example and example['meta']:
            try:
                if isinstance(example['meta'], str):
                    metadata = json.loads(example['meta'])
                else:
                    metadata = example['meta']
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        # Clean the paper text
        cleaned_text = self.clean_arxiv_text(text)
        
        # For full papers, we might want to extract sections
        if cleaned_text:
            # Basic section detection
            sections = self.extract_paper_sections(cleaned_text)
            
            if sections:
                # Format sections for training
                formatted_sections = []
                for section_name, section_content in sections.items():
                    if section_content.strip():
                        formatted_sections.append(f"{section_name}: {section_content.strip()}")
                
                formatted_text = "\n\n".join(formatted_sections)
            else:
                formatted_text = cleaned_text
            
            return {'text': formatted_text}
        
        return {'text': ''}
    
    def extract_paper_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from ArXiv paper text."""
        sections = {}
        
        # Common section patterns
        section_patterns = [
            r'(?i)\b(abstract)\b[:\s]*([^\n]*(?:\n(?!\b(?:introduction|related work|methodology|method|approach|experiments|results|conclusion|references)\b)[^\n]*)*)',
            r'(?i)\b(introduction)\b[:\s]*([^\n]*(?:\n(?!\b(?:related work|methodology|method|approach|experiments|results|conclusion|references)\b)[^\n]*)*)',
            r'(?i)\b(related work|literature review)\b[:\s]*([^\n]*(?:\n(?!\b(?:methodology|method|approach|experiments|results|conclusion|references)\b)[^\n]*)*)',
            r'(?i)\b(methodology|method|approach)\b[:\s]*([^\n]*(?:\n(?!\b(?:experiments|results|conclusion|references)\b)[^\n]*)*)',
            r'(?i)\b(experiments?|evaluation)\b[:\s]*([^\n]*(?:\n(?!\b(?:results|conclusion|references)\b)[^\n]*)*)',
            r'(?i)\b(results?)\b[:\s]*([^\n]*(?:\n(?!\b(?:conclusion|references)\b)[^\n]*)*)',
            r'(?i)\b(conclusion|discussion)\b[:\s]*([^\n]*(?:\n(?!\b(?:references)\b)[^\n]*)*)',
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                section_name = match.group(1).title()
                section_content = match.group(2).strip()
                if section_content and len(section_content) > 50:  # Minimum content length
                    sections[section_name] = section_content[:2000]  # Limit section length
        
        return sections
    
    def tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """Tokenize text examples."""
        # Filter out empty texts
        texts = [text for text in examples['text'] if text and text.strip()]
        
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_attention_mask=True
        )
        
        return tokenized
    
    def group_texts(self, examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
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

def download_arxiv_abstracts(num_samples: Optional[int] = None, streaming: bool = False) -> Dataset:
    """Download ArXiv abstracts from Common Pile."""
    print("Downloading ArXiv abstracts from Common Pile...")
    
    try:
        # Load the dataset
        dataset = load_dataset(
            'common-pile/arxiv_abstracts',
            split='train',
            streaming=streaming,
            trust_remote_code=True
        )
        
        # Limit samples if specified
        if num_samples and not streaming:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"ArXiv abstracts loaded: {len(dataset) if not streaming else 'streaming'} examples")
        return dataset
        
    except Exception as e:
        print(f"Error loading ArXiv abstracts: {e}")
        print("Creating fallback synthetic dataset...")
        return create_synthetic_arxiv_abstracts(num_samples or 1000)

def download_arxiv_papers(num_samples: Optional[int] = None, streaming: bool = False) -> Dataset:
    """Download ArXiv papers from Common Pile."""
    print("Downloading ArXiv papers from Common Pile...")
    
    try:
        # Load the dataset
        dataset = load_dataset(
            'common-pile/arxiv_papers',
            split='train',
            streaming=streaming,
            trust_remote_code=True
        )
        
        # Limit samples if specified
        if num_samples and not streaming:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"ArXiv papers loaded: {len(dataset) if not streaming else 'streaming'} examples")
        return dataset
        
    except Exception as e:
        print(f"Error loading ArXiv papers: {e}")
        print("Creating fallback synthetic dataset...")
        return create_synthetic_arxiv_papers(num_samples or 500)

def create_synthetic_arxiv_abstracts(num_samples: int = 1000) -> Dataset:
    """Create synthetic ArXiv abstracts for fallback."""
    print(f"Creating {num_samples} synthetic ArXiv abstracts...")
    
    # Sample abstract templates
    templates = [
        "We present a novel approach to {topic} using {method}. Our method achieves {result} on {dataset}. The key innovation is {innovation}. Experimental results demonstrate {performance}.",
        "This paper introduces {method} for {problem}. We show that {approach} can significantly improve {metric}. Our contributions include {contribution1} and {contribution2}.",
        "We propose {algorithm} to address {challenge} in {domain}. The method is based on {theory} and achieves {outcome}. Extensive experiments validate our approach.",
        "In this work, we investigate {research_question} through {methodology}. Our findings reveal {discovery} and provide insights into {implications}.",
        "We develop a {framework} for {application} that combines {technique1} with {technique2}. Results show {improvement} over existing methods."
    ]
    
    # Sample terms for templates
    topics = ["machine learning", "natural language processing", "computer vision", "deep learning", "neural networks"]
    methods = ["transformer architecture", "attention mechanism", "convolutional networks", "reinforcement learning", "self-supervised learning"]
    results = ["state-of-the-art performance", "significant improvements", "competitive results", "superior accuracy"]
    datasets = ["benchmark datasets", "standard evaluations", "multiple tasks", "diverse domains"]
    
    abstracts = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        abstract = template.format(
            topic=topics[i % len(topics)],
            method=methods[i % len(methods)],
            result=results[i % len(results)],
            dataset=datasets[i % len(datasets)],
            innovation="novel architecture design",
            performance="superior results",
            problem="complex optimization",
            approach="our proposed method",
            metric="accuracy and efficiency",
            contribution1="theoretical analysis",
            contribution2="empirical validation",
            algorithm="efficient algorithm",
            challenge="scalability issues",
            domain="machine learning",
            theory="information theory",
            outcome="improved performance",
            research_question="fundamental questions",
            methodology="experimental design",
            discovery="important patterns",
            implications="future research",
            framework="unified framework",
            application="practical applications",
            technique1="deep learning",
            technique2="optimization techniques",
            improvement="substantial improvements"
        )
        abstracts.append(f"Abstract: {abstract}")
    
    return Dataset.from_dict({"text": abstracts})

def create_synthetic_arxiv_papers(num_samples: int = 500) -> Dataset:
    """Create synthetic ArXiv papers for fallback."""
    print(f"Creating {num_samples} synthetic ArXiv papers...")
    
    papers = []
    for i in range(num_samples):
        paper = f"""Title: Advanced Methods in Machine Learning Research {i+1}

Abstract: This paper presents novel approaches to machine learning with applications in various domains. We propose innovative algorithms that achieve superior performance on benchmark tasks.

Introduction: Machine learning has become increasingly important in modern applications. This work addresses key challenges in the field through novel methodological contributions.

Methodology: Our approach combines theoretical insights with practical implementations. We develop new algorithms based on established principles while introducing innovative techniques.

Experiments: We conduct extensive experiments on multiple datasets to validate our approach. The results demonstrate significant improvements over existing methods.

Results: Our method achieves state-of-the-art performance across various metrics. The improvements are consistent and statistically significant.

Conclusion: This work contributes to the advancement of machine learning through novel algorithmic developments and comprehensive experimental validation."""
        papers.append(paper)
    
    return Dataset.from_dict({"text": papers})

def prepare_arxiv_dataset(
    output_dir: str = 'data/arxiv_dataset',
    include_abstracts: bool = True,
    include_papers: bool = True,
    abstracts_samples: Optional[int] = 50000,  # Limit for RTX 4090
    papers_samples: Optional[int] = 10000,     # Limit for RTX 4090
    tokenizer_name: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    max_length: int = 2048,
    num_proc: int = 4,
    test_size: float = 0.05,
    streaming: bool = False
) -> Tuple[Dataset, Dict[str, Any]]:
    """Prepare ArXiv dataset for 1B MoVE model training."""
    
    print(f"Preparing ArXiv dataset for 1B MoVE model")
    print(f"Output directory: {output_dir}")
    print(f"Include abstracts: {include_abstracts} (samples: {abstracts_samples})")
    print(f"Include papers: {include_papers} (samples: {papers_samples})")
    print(f"Max length: {max_length}")
    print(f"Test size: {test_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = ArXivDatasetProcessor(tokenizer_name, max_length)
    
    # Download datasets
    datasets_list = []
    
    if include_abstracts:
        abstracts_dataset = download_arxiv_abstracts(abstracts_samples, streaming)
        if abstracts_dataset:
            # Process abstracts
            print("Processing ArXiv abstracts...")
            processed_abstracts = abstracts_dataset.map(
                processor.process_arxiv_abstract,
                num_proc=num_proc,
                desc="Processing abstracts"
            )
            # Filter out empty texts
            processed_abstracts = processed_abstracts.filter(
                lambda x: x['text'] and len(x['text'].strip()) > 10
            )
            datasets_list.append(processed_abstracts)
            print(f"Processed abstracts: {len(processed_abstracts)} examples")
    
    if include_papers:
        papers_dataset = download_arxiv_papers(papers_samples, streaming)
        if papers_dataset:
            # Process papers
            print("Processing ArXiv papers...")
            processed_papers = papers_dataset.map(
                processor.process_arxiv_paper,
                num_proc=num_proc,
                desc="Processing papers"
            )
            # Filter out empty texts
            processed_papers = processed_papers.filter(
                lambda x: x['text'] and len(x['text'].strip()) > 50
            )
            datasets_list.append(processed_papers)
            print(f"Processed papers: {len(processed_papers)} examples")
    
    if not datasets_list:
        raise ValueError("No datasets loaded successfully")
    
    # Combine datasets
    if len(datasets_list) > 1:
        print("Combining ArXiv datasets...")
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
        processor.tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=combined_dataset.column_names,
        desc="Tokenizing"
    )
    
    # Group texts into chunks
    print("Grouping texts into chunks...")
    grouped_dataset = tokenized_dataset.map(
        processor.group_texts,
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
    
    # Calculate statistics
    train_size = len(split_dataset['train'])
    val_size = len(split_dataset['test'])
    total_train_tokens = train_size * max_length
    total_val_tokens = val_size * max_length
    
    # Save metadata
    metadata = {
        'dataset_type': 'arxiv_common_pile',
        'include_abstracts': include_abstracts,
        'include_papers': include_papers,
        'abstracts_samples': abstracts_samples,
        'papers_samples': papers_samples,
        'tokenizer': tokenizer_name,
        'max_length': max_length,
        'vocab_size': processor.vocab_size,
        'train_size': train_size,
        'validation_size': val_size,
        'total_tokens_train': total_train_tokens,
        'total_tokens_validation': total_val_tokens,
        'test_size': test_size,
        'processing_date': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown')
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nArXiv dataset preparation completed!")
    print(f"Train examples: {train_size:,}")
    print(f"Validation examples: {val_size:,}")
    print(f"Total training tokens: {total_train_tokens:,}")
    print(f"Total validation tokens: {total_val_tokens:,}")
    print(f"Estimated memory usage: {(total_train_tokens * 4) / (1024**3):.2f} GB")
    
    return split_dataset, metadata

def estimate_1b_training_requirements(dataset_size: int, tokens_per_example: int = 2048) -> Dict[str, float]:
    """Estimate training requirements for 1B MoVE model on RTX 4090."""
    
    # Model parameters
    model_params = 1e9  # 1B parameters
    bytes_per_param = 4  # FP32
    
    # Memory calculations
    model_memory_gb = (model_params * bytes_per_param) / (1024**3)
    
    # Training memory (model + gradients + optimizer states + activations)
    training_memory_gb = model_memory_gb * 4  # Rough estimate
    
    # Dataset memory
    total_tokens = dataset_size * tokens_per_example
    dataset_memory_gb = (total_tokens * 4) / (1024**3)  # 4 bytes per token
    
    # RTX 4090 specs
    gpu_memory_gb = 16
    available_memory_gb = gpu_memory_gb * 0.9  # Leave some headroom
    
    # Batch size estimation
    memory_per_token = training_memory_gb / tokens_per_example
    max_batch_tokens = int(available_memory_gb / memory_per_token)
    max_batch_size = max(1, max_batch_tokens // tokens_per_example)
    
    # Training time estimation (very rough)
    tokens_per_second = 1000  # Conservative estimate for RTX 4090
    total_training_seconds = total_tokens / tokens_per_second
    training_hours = total_training_seconds / 3600
    
    results = {
        'model_memory_gb': model_memory_gb,
        'training_memory_gb': training_memory_gb,
        'dataset_memory_gb': dataset_memory_gb,
        'total_memory_required_gb': training_memory_gb + dataset_memory_gb,
        'gpu_memory_available_gb': available_memory_gb,
        'memory_fits': (training_memory_gb + dataset_memory_gb) <= available_memory_gb,
        'recommended_batch_size': max_batch_size,
        'estimated_training_hours': training_hours,
        'total_tokens': total_tokens
    }
    
    print("\n=== 1B MoVE Training Requirements (RTX 4090) ===")
    print(f"Model memory: {model_memory_gb:.2f} GB")
    print(f"Training memory: {training_memory_gb:.2f} GB")
    print(f"Dataset memory: {dataset_memory_gb:.2f} GB")
    print(f"Total memory required: {results['total_memory_required_gb']:.2f} GB")
    print(f"GPU memory available: {available_memory_gb:.2f} GB")
    print(f"Memory fits: {'✓' if results['memory_fits'] else '✗'}")
    print(f"Recommended batch size: {max_batch_size}")
    print(f"Estimated training time: {training_hours:.1f} hours")
    print(f"Total tokens: {total_tokens:,}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Prepare ArXiv dataset for MoVE 1B model')
    
    parser.add_argument('--output_dir', type=str, default='data/arxiv_dataset',
                       help='Output directory for processed dataset')
    parser.add_argument('--abstracts', action='store_true', default=True,
                       help='Include ArXiv abstracts')
    parser.add_argument('--papers', action='store_true', default=True,
                       help='Include ArXiv papers')
    parser.add_argument('--abstracts_samples', type=int, default=50000,
                       help='Number of abstract samples (None for all)')
    parser.add_argument('--papers_samples', type=int, default=10000,
                       help='Number of paper samples (None for all)')
    parser.add_argument('--tokenizer', type=str, 
                       default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Tokenizer to use')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--num_proc', type=int, default=4,
                       help='Number of processes for data processing')
    parser.add_argument('--test_size', type=float, default=0.05,
                       help='Fraction of data to use for validation')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming for very large datasets')
    parser.add_argument('--estimate_requirements', action='store_true',
                       help='Estimate training requirements for 1B model')
    
    args = parser.parse_args()
    
    # Prepare dataset
    dataset, metadata = prepare_arxiv_dataset(
        output_dir=args.output_dir,
        include_abstracts=args.abstracts,
        include_papers=args.papers,
        abstracts_samples=args.abstracts_samples,
        papers_samples=args.papers_samples,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        test_size=args.test_size,
        streaming=args.streaming
    )
    
    # Estimate training requirements if requested
    if args.estimate_requirements:
        estimate_1b_training_requirements(
            dataset_size=len(dataset['train']),
            tokens_per_example=args.max_length
        )

if __name__ == '__main__':
    main()