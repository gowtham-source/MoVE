"""LLM Benchmark Evaluation Script

Evaluates MoVE models on standard LLM benchmarks:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense reasoning)
- ARC (AI2 Reasoning Challenge)
- TruthfulQA
- GSM8K (Math reasoning)
- HumanEval (Code generation)
- PIQA (Physical reasoning)
- WinoGrande (Commonsense reasoning)

Optimized for single GPU evaluation with memory management.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from move_large import MoVELarge, create_move_model_large
from move import MoVE, create_move_model

class BenchmarkEvaluator:
    """Evaluator for LLM benchmarks."""
    
    def __init__(self, model, tokenizer, device='cuda', max_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_logprobs(self, text: str, choices: List[str]) -> List[float]:
        """Get log probabilities for multiple choice answers."""
        logprobs = []
        
        for choice in choices:
            # Combine text and choice
            full_text = text + choice
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=False
            ).to(self.device)
            
            # Get text length for choice tokens
            text_inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            text_length = text_inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'])
                logits = outputs.logits
                
                # Get log probabilities for choice tokens
                choice_logits = logits[0, text_length-1:-1]  # Exclude last token
                choice_tokens = inputs['input_ids'][0, text_length:]
                
                # Calculate log probability
                log_probs = F.log_softmax(choice_logits, dim=-1)
                choice_logprob = log_probs.gather(1, choice_tokens.unsqueeze(1)).sum().item()
                
                logprobs.append(choice_logprob)
        
        return logprobs
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text continuation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length - max_new_tokens,
            padding=False
        ).to(self.device)
        
        with torch.no_grad():
            # Simple greedy generation
            input_ids = inputs['input_ids']
            
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append token
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated = self.tokenizer.decode(
            input_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()

def evaluate_mmlu(evaluator: BenchmarkEvaluator, subset: str = 'all', num_samples: int = None) -> Dict[str, float]:
    """Evaluate on MMLU benchmark."""
    print(f"Evaluating MMLU ({subset})...")
    
    # Load MMLU dataset
    if subset == 'all':
        # Load all subjects
        subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
            'formal_logic', 'global_facts', 'high_school_biology',
            'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography',
            'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology',
            'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality',
            'international_law', 'jurisprudence', 'logical_fallacies',
            'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions'
        ]
    else:
        subjects = [subset]
    
    all_correct = 0
    all_total = 0
    subject_scores = {}
    
    for subject in subjects:
        try:
            dataset = load_dataset('cais/mmlu', subject, split='test')
            
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = 0
            
            for example in tqdm(dataset, desc=f"MMLU {subject}"):
                question = example['question']
                choices = [example['choices'][i] for i in range(4)]
                correct_answer = example['answer']
                
                # Format prompt
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"
                
                # Get logprobs for each choice
                choice_labels = ['A', 'B', 'C', 'D']
                logprobs = evaluator.get_logprobs(prompt, choice_labels)
                
                # Predict answer
                predicted = np.argmax(logprobs)
                
                if predicted == correct_answer:
                    correct += 1
                total += 1
                
                # Memory cleanup
                if total % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            accuracy = correct / total if total > 0 else 0
            subject_scores[subject] = accuracy
            all_correct += correct
            all_total += total
            
            print(f"  {subject}: {accuracy:.3f} ({correct}/{total})")
            
        except Exception as e:
            print(f"  Error evaluating {subject}: {e}")
            continue
    
    overall_accuracy = all_correct / all_total if all_total > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'subject_scores': subject_scores,
        'total_correct': all_correct,
        'total_questions': all_total
    }

def evaluate_hellaswag(evaluator: BenchmarkEvaluator, num_samples: int = None) -> Dict[str, float]:
    """Evaluate on HellaSwag benchmark."""
    print("Evaluating HellaSwag...")
    
    dataset = load_dataset('hellaswag', split='validation')
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="HellaSwag"):
        context = example['ctx']
        endings = example['endings']
        correct_answer = int(example['label'])
        
        # Format prompt
        prompt = f"Context: {context}\nWhich ending makes the most sense?\n"
        
        # Get logprobs for each ending
        logprobs = evaluator.get_logprobs(prompt, endings)
        
        # Predict answer
        predicted = np.argmax(logprobs)
        
        if predicted == correct_answer:
            correct += 1
        total += 1
        
        # Memory cleanup
        if total % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def evaluate_arc(evaluator: BenchmarkEvaluator, challenge: bool = True, num_samples: int = None) -> Dict[str, float]:
    """Evaluate on ARC benchmark."""
    subset = 'ARC-Challenge' if challenge else 'ARC-Easy'
    print(f"Evaluating {subset}...")
    
    dataset = load_dataset('ai2_arc', subset, split='test')
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc=subset):
        question = example['question']
        choices = example['choices']['text']
        labels = example['choices']['label']
        correct_answer = example['answerKey']
        
        # Find correct answer index
        try:
            correct_idx = labels.index(correct_answer)
        except ValueError:
            continue  # Skip if answer key not found
        
        # Format prompt
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}. {choice}\n"
        prompt += "Answer:"
        
        # Get logprobs for each choice label
        logprobs = evaluator.get_logprobs(prompt, labels)
        
        # Predict answer
        predicted = np.argmax(logprobs)
        
        if predicted == correct_idx:
            correct += 1
        total += 1
        
        # Memory cleanup
        if total % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def evaluate_gsm8k(evaluator: BenchmarkEvaluator, num_samples: int = None) -> Dict[str, float]:
    """Evaluate on GSM8K math reasoning benchmark."""
    print("Evaluating GSM8K...")
    
    dataset = load_dataset('gsm8k', 'main', split='test')
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="GSM8K"):
        question = example['question']
        answer = example['answer']
        
        # Extract numerical answer
        answer_lines = answer.split('\n')
        numerical_answer = None
        for line in answer_lines:
            if '####' in line:
                try:
                    numerical_answer = float(line.split('####')[1].strip().replace(',', ''))
                    break
                except:
                    continue
        
        if numerical_answer is None:
            continue
        
        # Format prompt
        prompt = f"Question: {question}\nLet's solve this step by step.\nAnswer:"
        
        # Generate response
        generated = evaluator.generate_text(prompt, max_new_tokens=200)
        
        # Extract numerical answer from generated text
        predicted_answer = None
        try:
            # Look for numbers in the generated text
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', generated.replace(',', ''))
            if numbers:
                predicted_answer = float(numbers[-1])  # Take the last number
        except:
            pass
        
        if predicted_answer is not None and abs(predicted_answer - numerical_answer) < 1e-6:
            correct += 1
        total += 1
        
        # Memory cleanup
        if total % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def evaluate_piqa(evaluator: BenchmarkEvaluator, num_samples: int = None) -> Dict[str, float]:
    """Evaluate on PIQA physical reasoning benchmark."""
    print("Evaluating PIQA...")
    
    dataset = load_dataset('piqa', split='validation')
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="PIQA"):
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        correct_answer = int(example['label'])
        
        # Format prompt
        prompt = f"Goal: {goal}\nWhich solution is better?\n"
        
        # Get logprobs for each solution
        solutions = [sol1, sol2]
        logprobs = evaluator.get_logprobs(prompt, solutions)
        
        # Predict answer
        predicted = np.argmax(logprobs)
        
        if predicted == correct_answer:
            correct += 1
        total += 1
        
        # Memory cleanup
        if total % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def run_comprehensive_evaluation(
    model_path: str,
    model_type: str = 'move',
    benchmarks: List[str] = ['mmlu', 'hellaswag', 'arc_challenge', 'arc_easy', 'piqa'],
    num_samples: Optional[int] = None,
    output_file: str = 'benchmark_results.json',
    device: str = 'cuda'
):
    """Run comprehensive benchmark evaluation."""
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    # Load model
    if model_type == 'move':
        model = create_move_model('medium')
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'move_large':
        model = create_move_model_large('1b')
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(model, tokenizer, device)
    
    # Run evaluations
    results = {
        'model_path': model_path,
        'model_type': model_type,
        'num_samples': num_samples,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': {}
    }
    
    for benchmark in benchmarks:
        print(f"\n{'='*50}")
        print(f"Running {benchmark.upper()} evaluation")
        print(f"{'='*50}")
        
        try:
            if benchmark == 'mmlu':
                result = evaluate_mmlu(evaluator, num_samples=num_samples)
            elif benchmark == 'hellaswag':
                result = evaluate_hellaswag(evaluator, num_samples=num_samples)
            elif benchmark == 'arc_challenge':
                result = evaluate_arc(evaluator, challenge=True, num_samples=num_samples)
            elif benchmark == 'arc_easy':
                result = evaluate_arc(evaluator, challenge=False, num_samples=num_samples)
            elif benchmark == 'gsm8k':
                result = evaluate_gsm8k(evaluator, num_samples=num_samples)
            elif benchmark == 'piqa':
                result = evaluate_piqa(evaluator, num_samples=num_samples)
            else:
                print(f"Unknown benchmark: {benchmark}")
                continue
            
            results['benchmarks'][benchmark] = result
            
            # Print summary
            if 'accuracy' in result:
                print(f"\n{benchmark.upper()} Accuracy: {result['accuracy']:.3f}")
            elif 'overall_accuracy' in result:
                print(f"\n{benchmark.upper()} Overall Accuracy: {result['overall_accuracy']:.3f}")
            
        except Exception as e:
            print(f"Error evaluating {benchmark}: {e}")
            results['benchmarks'][benchmark] = {'error': str(e)}
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    for benchmark, result in results['benchmarks'].items():
        if 'error' in result:
            print(f"{benchmark.upper()}: ERROR - {result['error']}")
        elif 'accuracy' in result:
            print(f"{benchmark.upper()}: {result['accuracy']:.3f}")
        elif 'overall_accuracy' in result:
            print(f"{benchmark.upper()}: {result['overall_accuracy']:.3f}")
    
    print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate MoVE model on LLM benchmarks')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['move', 'move_large'],
                       default='move', help='Type of model')
    parser.add_argument('--benchmarks', nargs='+',
                       choices=['mmlu', 'hellaswag', 'arc_challenge', 'arc_easy', 'gsm8k', 'piqa'],
                       default=['mmlu', 'hellaswag', 'arc_challenge'],
                       help='Benchmarks to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--output_file', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_comprehensive_evaluation(
        model_path=args.model_path,
        model_type=args.model_type,
        benchmarks=args.benchmarks,
        num_samples=args.num_samples,
        output_file=args.output_file,
        device=args.device
    )

if __name__ == '__main__':
    main()