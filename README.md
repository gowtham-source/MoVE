# Modular Vector Engine (MoVE)

🎯 **Core Hypothesis**: A high-performing large language model can be approximated or even surpassed by a modular system of smaller, specialized models that collaboratively compute the same vector transformations used in Llama-style transformers, but with greater efficiency, interpretability, and adaptability.

## 🔍 Novel Approach Overview

Instead of stacking transformer blocks, we deconstruct the transformer into its functional primitives and rebuild them as independent ML models, each optimized for a specific sub-task:

| Transformer Component | Replaced With          | Model Type                                            |
| --------------------- | ---------------------- | ----------------------------------------------------- |
| Token Embedding       | Sparse Embedding Model | Lightweight encoder (e.g., Sentence-T5, fastText++)   |
| Positional Encoding   | Positional Predictor   | Small MLP or RBF network                              |
| Attention Mechanism   | Attention Approximator | Graph Neural Network or Low-rank Attention Net        |
| Feedforward Layer     | Expert FFN Ensemble    | Mixture-of-Experts (MoE) with domain-specialized FFNs |
| Output Projection     | Output Decoder         | Task-specific decoder (e.g., classifier, generator)   |

## 📁 Project Structure

```
MoVE/
├── data/                    # Dataset storage
│   ├── owt_1pct/           # OpenWebText 1% slice
│   └── owt_1pct_tok/       # Tokenized dataset
├── scripts/                 # Utility scripts
│   ├── tokenise.py         # Tokenization script
│   └── download_data.py    # Dataset acquisition
├── models/                  # Model storage
│   └── tiny_llama/         # TinyLlama baseline
├── move/                    # Core MoVE modules
│   ├── components/         # Modular components
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation framework
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset**:
   ```bash
   python scripts/download_data.py
   ```

3. **Tokenize data**:
   ```bash
   python scripts/tokenise.py
   ```

4. **Download baseline model**:
   ```bash
   huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --local-dir models/tiny_llama
   ```

## 🧪 Experimental Roadmap

### Phase 1: Deconstruction & Baseline
**Goal**: Replicate Llama-7B's vector flow using modular components.

**Steps**:
1. Train a token-to-vector encoder that mimics Llama's embedding layer
2. Build a positional vector generator that aligns with Llama's RoPE
3. Replace attention with a graph-based attention approximator
4. Replace FFN with a MoE system trained on diverse corpora
5. Validate that the final vector output matches Llama's layer-wise activations (cosine similarity > 0.95)

## 📊 Performance Targets

- **Memory Efficiency**: < 12GB VRAM on RTX 4090
- **Vector Similarity**: > 0.95 cosine similarity with Llama activations
- **Inference Speed**: Competitive with TinyLlama baseline
- **Modularity**: Independent training and deployment of components

## 🔗 References

- [LLaMA Explained](https://pub.towardsai.net/llama-explained-a70e71e706e9)
- [Building LLaMA 4 from Scratch](https://www.dailydoseofds.com/building-llama-4-from-scratch-with-python/)

## 📝 License

MIT License - See LICENSE file for details.