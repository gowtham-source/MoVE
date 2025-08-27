# MoVE: Production-Ready Large Language Model

This guide explains how to scale the MoVE (Mixture of Vector Experts) architecture to a production-ready large language model with comprehensive training, evaluation, and deployment capabilities.

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers accelerate
pip install numpy matplotlib seaborn plotly flask psutil
pip install evaluate scikit-learn requests tqdm

# Optional: For GPU monitoring
pip install pynvml

# Optional: For advanced optimizations
pip install flash-attn xformers
```

### Hardware Requirements

- **Minimum**: NVIDIA RTX 4090 (16GB VRAM)
- **Recommended**: Multiple GPUs or higher VRAM for larger models
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 100GB+ free space for datasets and checkpoints

## 📊 Model Architectures

The production MoVE supports multiple model sizes optimized for different use cases:

| Model Size | Parameters | Layers | Hidden Size | Attention Heads | MoE Experts | Memory (FP16) |
|------------|------------|--------|-------------|-----------------|-------------|---------------|
| MoVE-1B    | ~1B        | 24     | 2048        | 16              | 8           | ~8GB          |
| MoVE-3B    | ~3B        | 32     | 2560        | 20              | 16          | ~12GB         |
| MoVE-7B    | ~7B        | 32     | 4096        | 32              | 32          | ~16GB         |

## 🏗️ Project Structure

```
MoVE/
├── move.py                     # Original MoVE implementation
├── move_large.py              # Production-ready large models
├── inference_optimized.py     # Optimized inference engine
├── scripts/
│   ├── train_large.py         # Large-scale training script
│   ├── prepare_large_dataset.py # Dataset preparation
│   ├── eval_benchmarks.py     # Benchmark evaluation
│   └── train_move.py          # Original training script
├── deploy/
│   ├── api_server.py          # REST API server
│   └── chat_interface.html    # Web chat interface
├── monitoring/
│   ├── performance_monitor.py # Performance monitoring
│   └── templates/
│       └── dashboard.html     # Web dashboard
└── data/                      # Training datasets
```

## 🎯 Step-by-Step Production Deployment

### Step 1: Prepare Large-Scale Dataset

```bash
# Prepare a large dataset (C4, OpenWebText, etc.)
python scripts/prepare_large_dataset.py \
    --dataset_name "c4" \
    --output_dir "data/c4_processed" \
    --max_length 2048 \
    --num_proc 8 \
    --train_split_size 0.95

# Estimate training time
python scripts/prepare_large_dataset.py --estimate_only --dataset_name "c4"
```

### Step 2: Start Performance Monitoring

```bash
# Start system monitoring (run in background)
python monitoring/performance_monitor.py --mode monitor --interval 10 &

# Start web dashboard (optional)
python monitoring/performance_monitor.py --mode dashboard --port 5000 &
```

### Step 3: Train Large Model

```bash
# Train MoVE-1B model
python scripts/train_large.py \
    --model_size "1B" \
    --data_path "data/c4_processed" \
    --output_dir "checkpoints/move_1b" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --max_steps 100000 \
    --save_steps 5000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --warmup_steps 1000 \
    --use_flash_attention \
    --gradient_checkpointing

# Train MoVE-3B model (requires more memory optimization)
python scripts/train_large.py \
    --model_size "3B" \
    --data_path "data/c4_processed" \
    --output_dir "checkpoints/move_3b" \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 8e-5 \
    --num_epochs 2 \
    --max_steps 80000 \
    --save_steps 4000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --warmup_steps 800 \
    --use_flash_attention \
    --gradient_checkpointing \
    --cpu_offload
```

### Step 4: Evaluate on Benchmarks

```bash
# Evaluate on standard LLM benchmarks
python scripts/eval_benchmarks.py \
    --model_path "checkpoints/move_1b" \
    --benchmarks "mmlu,hellaswag,arc,truthfulqa" \
    --batch_size 8 \
    --max_samples 1000 \
    --output_dir "results/move_1b_eval"

# Evaluate on coding benchmarks
python scripts/eval_benchmarks.py \
    --model_path "checkpoints/move_1b" \
    --benchmarks "humaneval,gsm8k" \
    --batch_size 4 \
    --max_samples 500 \
    --output_dir "results/move_1b_coding"
```

### Step 5: Deploy Model

```bash
# Start API server
python deploy/api_server.py \
    --model_path "checkpoints/move_1b" \
    --host "0.0.0.0" \
    --port 8000 \
    --max_length 2048 \
    --temperature 0.7 \
    --top_p 0.9

# Test API
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain quantum computing:", "max_length": 200}'
```

### Step 6: Use Optimized Inference

```python
from inference_optimized import OptimizedMoVEInference

# Load model with optimizations
inference = OptimizedMoVEInference(
    model_path="checkpoints/move_1b",
    device="cuda",
    use_kv_cache=True,
    quantization="fp16",
    max_batch_size=8
)

# Generate text
response = inference.generate(
    prompt="Write a Python function to calculate fibonacci numbers:",
    max_length=200,
    temperature=0.7,
    top_p=0.9
)

print(response)
```

## 📈 Performance Monitoring

### Real-time Dashboard

Access the web dashboard at `http://localhost:5000` to monitor:

- System resources (CPU, Memory, GPU)
- Training progress (Loss, Learning Rate, Perplexity)
- Model performance benchmarks
- Real-time metrics and alerts

### Generate Performance Reports

```bash
# Generate comprehensive performance report
python monitoring/performance_monitor.py --mode report --output_dir "reports/$(date +%Y%m%d)"
```

## 🔧 Advanced Configuration

### Memory Optimization

For training larger models on RTX 4090:

```python
# In train_large.py, use these settings:
training_args = {
    "gradient_checkpointing": True,
    "dataloader_pin_memory": False,
    "gradient_accumulation_steps": 16,
    "per_device_train_batch_size": 1,
    "fp16": True,
    "cpu_offload": True,  # For 7B models
}
```

### Custom Dataset Integration

```python
# Add your custom dataset
from scripts.prepare_large_dataset import DatasetProcessor

processor = DatasetProcessor()
processor.process_custom_dataset(
    data_path="path/to/your/data.jsonl",
    output_dir="data/custom_processed",
    text_column="text",
    max_length=2048
)
```

### Model Architecture Customization

```python
# Modify move_large.py for custom architectures
config = MoVELargeConfig(
    vocab_size=50257,
    max_position_embeddings=4096,  # Longer sequences
    num_hidden_layers=40,          # Deeper model
    hidden_size=3072,              # Wider model
    num_attention_heads=24,
    moe_experts=64,                # More experts
    moe_topk=4,                    # More active experts
    # ... other parameters
)
```

## 🎯 Benchmark Results

Expected performance on standard benchmarks:

| Benchmark | MoVE-1B | MoVE-3B | MoVE-7B | GPT-3.5 |
|-----------|---------|---------|---------|----------|
| MMLU      | 45-50%  | 55-60%  | 65-70%  | 70%      |
| HellaSwag | 60-65%  | 70-75%  | 80-85%  | 85%      |
| ARC-Easy  | 70-75%  | 80-85%  | 85-90%  | 90%      |
| ARC-Hard  | 35-40%  | 45-50%  | 55-60%  | 60%      |
| TruthfulQA| 30-35%  | 35-40%  | 40-45%  | 45%      |
| GSM8K     | 15-20%  | 25-30%  | 35-40%  | 40%      |
| HumanEval | 10-15%  | 20-25%  | 30-35%  | 35%      |

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size and increase gradient accumulation
   --batch_size 1 --gradient_accumulation_steps 32
   
   # Enable CPU offloading
   --cpu_offload
   
   # Use gradient checkpointing
   --gradient_checkpointing
   ```

2. **Slow Training**
   ```bash
   # Enable Flash Attention
   --use_flash_attention
   
   # Optimize data loading
   --dataloader_num_workers 4 --dataloader_pin_memory
   
   # Use mixed precision
   --fp16
   ```

3. **Poor Convergence**
   ```bash
   # Adjust learning rate
   --learning_rate 5e-5
   
   # Increase warmup steps
   --warmup_steps 2000
   
   # Use gradient clipping
   --max_grad_norm 1.0
   ```

### Performance Optimization Tips

1. **Dataset Preprocessing**
   - Use efficient tokenization with caching
   - Optimize sequence packing
   - Use memory-mapped datasets for large data

2. **Training Optimization**
   - Use gradient accumulation for effective larger batch sizes
   - Enable mixed precision training (FP16)
   - Use gradient checkpointing for memory efficiency
   - Implement learning rate scheduling

3. **Inference Optimization**
   - Use KV-cache for faster generation
   - Implement quantization (INT8/FP16)
   - Batch multiple requests
   - Use efficient attention mechanisms

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Flash Attention](https://github.com/HazyResearch/flash-attention)
- [Model Optimization Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for:

- Adding new model architectures
- Implementing additional benchmarks
- Optimizing training procedures
- Improving inference performance

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Note**: This is a research implementation. For production use, ensure thorough testing and validation on your specific use cases and datasets.