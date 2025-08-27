# MoVE Model Inference Guide

This guide explains how to perform inference with trained MoVE models using the optimized inference engine.

## Overview

The MoVE inference system provides:
- **Optimized inference** with KV-cache for faster generation
- **Quantization support** (INT8/INT4) for memory efficiency
- **Multiple sampling strategies** (temperature, top-p, top-k)
- **Interactive chat mode** for conversational AI
- **Batch processing** capabilities
- **Memory optimization** for large models

## Quick Start

### 1. Basic Text Generation

```bash
python scripts/inference_move.py \
    --model_path models/move_llama_transfer/checkpoint-final.pt \
    --tokenizer unsloth/Llama-3.2-1B-bnb-4bit \
    --prompt "The future of AI is" \
    --max_new_tokens 100 \
    --temperature 0.8
```

### 2. Interactive Chat Mode

```bash
python scripts/inference_move.py \
    --model_path models/move_llama_transfer/checkpoint-final.pt \
    --tokenizer unsloth/Llama-3.2-1B-bnb-4bit \
    --chat
```

### 3. Run Example Script

```bash
python scripts/example_inference.py
```

## Command Line Arguments

### Required Arguments
- `--model_path`: Path to the trained MoVE model checkpoint

### Model Configuration
- `--tokenizer`: Tokenizer to use (default: `meta-llama/Llama-2-7b-hf`)
- `--device`: Device to use (`auto`, `cuda`, `cpu`)
- `--quantization`: Quantization method (`none`, `int8`, `int4`)
- `--max_length`: Maximum sequence length (default: 2048)

### Generation Parameters
- `--prompt`: Text prompt for generation
- `--max_new_tokens`: Maximum number of new tokens (default: 100)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top_p`: Top-p (nucleus) sampling threshold (default: 0.9)
- `--top_k`: Top-k sampling threshold (default: 50)

### Mode Selection
- `--chat`: Start interactive chat mode
- `--no_kv_cache`: Disable KV-cache (slower but uses less memory)

## Python API Usage

### Basic Usage

```python
from scripts.inference_move import MoVEInferenceEngine

# Initialize the engine
engine = MoVEInferenceEngine(
    model_path="models/move_llama_transfer/checkpoint-final.pt",
    tokenizer_name="unsloth/Llama-3.2-1B-bnb-4bit",
    device="auto",
    quantization="none",
    use_kv_cache=True
)

# Generate text
result = engine.generate(
    prompt="Explain quantum computing in simple terms:",
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9
)

print(result)
```

### Advanced Configuration

```python
# With quantization for memory efficiency
engine = MoVEInferenceEngine(
    model_path="models/move_llama_transfer/checkpoint-final.pt",
    tokenizer_name="unsloth/Llama-3.2-1B-bnb-4bit",
    device="cuda",
    quantization="int8",  # Reduces memory usage
    max_length=4096,
    use_kv_cache=True
)

# Generate with custom parameters
result = engine.generate(
    prompt="Write a Python function to sort a list:",
    max_new_tokens=200,
    temperature=0.3,  # Lower for more deterministic output
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)
```

## Model Loading Formats

The inference engine supports multiple model formats:

### 1. PyTorch Checkpoint (.pt/.pth)
```python
# Single file checkpoint
model_path = "models/move_model.pt"
```

### 2. Hugging Face Format
```python
# Directory with config.json and pytorch_model.bin
model_path = "models/move_huggingface/"
```

### 3. Custom Checkpoint Format
```python
# Checkpoint with config and state_dict
checkpoint = {
    'config': model_config,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'model.pt')
```

## Performance Optimization

### Memory Optimization

1. **Use Quantization**:
   ```bash
   --quantization int8  # Reduces memory by ~50%
   ```

2. **Adjust Sequence Length**:
   ```bash
   --max_length 1024  # Reduce for lower memory usage
   ```

3. **Disable KV-Cache** (if memory is very limited):
   ```bash
   --no_kv_cache
   ```

### Speed Optimization

1. **Enable KV-Cache** (default):
   - Caches key-value pairs for faster generation
   - Significantly speeds up multi-token generation

2. **Use GPU**:
   ```bash
   --device cuda
   ```

3. **Optimize Batch Size**:
   ```python
   # For batch processing
   engine.generate_batch(prompts, batch_size=4)
   ```

## Sampling Strategies

### Temperature Sampling
- **Low (0.1-0.3)**: More deterministic, good for factual content
- **Medium (0.7-0.8)**: Balanced creativity and coherence
- **High (1.0+)**: More creative but potentially less coherent

### Top-p (Nucleus) Sampling
- **0.9**: Good balance (default)
- **0.95**: More diverse
- **0.8**: More focused

### Top-k Sampling
- **50**: Good default
- **20**: More focused
- **100**: More diverse

## Example Use Cases

### 1. Text Completion
```python
prompt = "The benefits of renewable energy include"
result = engine.generate(prompt, max_new_tokens=100, temperature=0.7)
```

### 2. Question Answering
```python
prompt = "Q: What is the capital of France?\nA:"
result = engine.generate(prompt, max_new_tokens=50, temperature=0.3)
```

### 3. Code Generation
```python
prompt = "# Function to calculate factorial\ndef factorial(n):"
result = engine.generate(prompt, max_new_tokens=100, temperature=0.2)
```

### 4. Creative Writing
```python
prompt = "Once upon a time in a magical forest,"
result = engine.generate(prompt, max_new_tokens=200, temperature=0.9)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Use quantization
   --quantization int8
   
   # Reduce sequence length
   --max_length 1024
   
   # Disable KV-cache
   --no_kv_cache
   ```

2. **Model Not Found**:
   - Check the model path is correct
   - Ensure the model was saved properly during training
   - Verify file permissions

3. **Tokenizer Issues**:
   - Use the same tokenizer as training
   - Check tokenizer compatibility
   - Ensure internet connection for downloading tokenizer

4. **Slow Generation**:
   - Enable KV-cache (default)
   - Use GPU if available
   - Consider quantization
   - Reduce max_new_tokens

### Performance Monitoring

```python
import time

start_time = time.time()
result = engine.generate(prompt, max_new_tokens=100)
end_time = time.time()

print(f"Generation time: {end_time - start_time:.2f}s")
print(f"Tokens per second: {100 / (end_time - start_time):.2f}")
```

## Advanced Features

### Custom Generation Loop

```python
# Manual token-by-token generation
input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = engine.model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == engine.tokenizer.eos_token_id:
            break
```

### Batch Processing

```python
prompts = [
    "The weather today is",
    "Machine learning is",
    "Python programming"
]

results = []
for prompt in prompts:
    result = engine.generate(prompt, max_new_tokens=50)
    results.append(result)
```

## Integration Examples

### Web API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
engine = MoVEInferenceEngine(model_path="path/to/model.pt")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    
    result = engine.generate(prompt, max_new_tokens=max_tokens)
    return jsonify({'generated_text': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Gradio Interface

```python
import gradio as gr

engine = MoVEInferenceEngine(model_path="path/to/model.pt")

def generate_response(prompt, max_tokens, temperature):
    return engine.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(1, 500, value=100, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text")
)

iface.launch()
```

This comprehensive inference system provides everything needed to deploy and use trained MoVE models effectively in production environments.