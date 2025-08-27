#!/usr/bin/env python3
"""Baseline Model Setup Script for MoVE Project

Downloads TinyLlama baseline model and verifies VRAM usage.
This script implements Step 3 from the experimental roadmap.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if CUDA is available and get GPU info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        logger.info(f"✅ CUDA available")
        logger.info(f"GPU Count: {gpu_count}")
        logger.info(f"Current Device: {current_device}")
        logger.info(f"GPU Name: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        
        return True, gpu_memory
    else:
        logger.warning("⚠️ CUDA not available. Will use CPU.")
        return False, 0

def download_tinyllama_model(model_dir: str = "models/tiny_llama"):
    """Download TinyLlama model using huggingface-cli.
    
    Args:
        model_dir: Directory to save the model
    """
    try:
        logger.info("Downloading TinyLlama model...")
        
        # Create models directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Download using huggingface-cli
        model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        cmd = [
            "huggingface-cli", "download",
            model_name,
            "--local-dir", model_dir,
            "--local-dir-use-symlinks", "False"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("✅ TinyLlama model downloaded successfully!")
        logger.info(f"Model saved to: {model_dir}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading model: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def verify_model_loading(model_dir: str = "models/tiny_llama"):
    """Verify model loading and VRAM usage.
    
    Args:
        model_dir: Directory containing the model
    """
    try:
        logger.info("Verifying model loading and VRAM usage...")
        
        # Check if CUDA is available
        cuda_available, gpu_memory = check_gpu_availability()
        
        if cuda_available:
            # Clear GPU cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Initial GPU memory usage: {initial_memory:.2f} GB")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model with appropriate settings
        logger.info("Loading model...")
        if cuda_available:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float32
            )
        
        logger.info("✅ Model loaded successfully!")
        
        # Check VRAM usage
        if cuda_available:
            current_memory = torch.cuda.memory_allocated() / 1e9
            memory_used = current_memory - initial_memory
            logger.info(f"Current GPU memory usage: {current_memory:.2f} GB")
            logger.info(f"Model memory usage: {memory_used:.2f} GB")
            
            # Check if within target (< 12 GB)
            if memory_used < 12.0:
                logger.info(f"✅ VRAM usage ({memory_used:.2f} GB) is within target (< 12 GB)")
            else:
                logger.warning(f"⚠️ VRAM usage ({memory_used:.2f} GB) exceeds target (12 GB)")
        
        # Test inference
        logger.info("Testing inference...")
        test_input = "The future of AI is"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        if cuda_available:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation: {generated_text}")
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        
        return True, memory_used if cuda_available else 0
        
    except Exception as e:
        logger.error(f"Error verifying model: {str(e)}")
        return False, 0

def main():
    """Main function to download and verify the baseline model."""
    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        logger.info("Starting MoVE baseline model setup...")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check GPU availability first
        check_gpu_availability()
        
        # Download model
        if download_tinyllama_model():
            logger.info("Model download completed successfully!")
        else:
            logger.error("Model download failed!")
            sys.exit(1)
        
        # Verify model loading
        success, vram_usage = verify_model_loading()
        
        if success:
            logger.info("✅ Baseline model setup completed successfully!")
            logger.info("Next step: Implement modular components")
            
            if vram_usage > 0:
                logger.info(f"Final VRAM usage: {vram_usage:.2f} GB")
        else:
            logger.error("❌ Model verification failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()