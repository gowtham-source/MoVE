#!/usr/bin/env python3
"""
Example script demonstrating MoVE model inference
"""

import os
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from inference_move import MoVEInferenceEngine

def main():
    # Example usage of the MoVE inference engine
    
    # Path to your trained model (adjust as needed)
    model_path = "models/move_llama_transfer/checkpoint-final.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_move_llama_transfer.py")
        return
    
    print("Initializing MoVE Inference Engine...")
    
    # Initialize the inference engine
    engine = MoVEInferenceEngine(
        model_path=model_path,
        tokenizer_name="unsloth/Llama-3.2-1B-bnb-4bit",  # Use same tokenizer as training
        device="auto",
        quantization="none",  # Can use "int8" for faster inference
        max_length=2048,
        use_kv_cache=True
    )
    
    print("Model loaded successfully!")
    print("\n" + "="*60)
    
    # Example 1: Simple text generation
    print("Example 1: Simple Text Generation")
    print("-" * 40)
    
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}")
    
    generated_text = engine.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    
    print(f"Generated: {generated_text}")
    print("\n" + "="*60)
    
    # Example 2: Question answering
    print("Example 2: Question Answering")
    print("-" * 40)
    
    qa_prompt = "Question: What is machine learning?\nAnswer:"
    print(f"Prompt: {qa_prompt}")
    
    qa_response = engine.generate(
        prompt=qa_prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    print(f"Generated: {qa_response}")
    print("\n" + "="*60)
    
    # Example 3: Code generation
    print("Example 3: Code Generation")
    print("-" * 40)
    
    code_prompt = "# Python function to calculate fibonacci numbers\ndef fibonacci(n):"
    print(f"Prompt: {code_prompt}")
    
    code_response = engine.generate(
        prompt=code_prompt,
        max_new_tokens=80,
        temperature=0.3,  # Lower temperature for more deterministic code
        top_p=0.95,
        do_sample=True
    )
    
    print(f"Generated: {code_response}")
    print("\n" + "="*60)
    
    # Example 4: Interactive chat (optional)
    print("Example 4: Interactive Chat")
    print("-" * 40)
    print("Starting interactive chat mode...")
    print("Type 'skip' to skip chat mode, or interact with the model")
    
    user_input = input("Do you want to start chat mode? (y/n/skip): ").strip().lower()
    
    if user_input == 'y' or user_input == 'yes':
        engine.chat("You are a helpful AI assistant trained on the MoVE architecture.")
    else:
        print("Skipping chat mode.")
    
    print("\nInference examples completed!")
    
if __name__ == "__main__":
    main()