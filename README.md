# MLX Fine-Tuning

A production-ready framework for fine-tuning Large Language Models on Apple Silicon using MLX and LoRA (Low-Rank Adaptation).

## Overview

This repository provides tools and examples for efficient LLM fine-tuning on Apple Silicon devices using:

- **MLX Framework**: Apple's machine learning framework optimized for Apple Silicon
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning technique
- **Production-grade evaluation**: LLM-as-a-Judge benchmarking system

## Features

- Parameter-efficient fine-tuning with LoRA
- Support for models from 0.5B to 7B+ parameters
- Production-ready evaluation framework
- Memory-optimized training with gradient checkpointing
- Comprehensive documentation and examples

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install mlx-lm numpy scipy scikit-learn tqdm
```

### Basic Usage

```bash
# Download a model
mlx_lm.convert --hf-path mlx-community/Qwen2.5-0.5B-Instruct-4bit -q

# Fine-tune with LoRA
python -m mlx_lm.lora \
  --model mlx_model \
  --train \
  --data data/ \
  --adapter-path adapters/my-model \
  --iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-5

# Generate with fine-tuned model
mlx_lm.generate --model mlx_model --adapter-path adapters/my-model \
  --prompt "Your prompt here"
```

## Understanding LoRA

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that dramatically reduces the computational and memory requirements for adapting large language models to specific tasks.

### How LoRA Works

Instead of fine-tuning all model parameters, LoRA injects trainable low-rank matrices into the model's architecture. Specifically, for a pre-trained weight matrix W, LoRA represents the weight update as:

```
W' = W + BA
```

Where:
- `W` is the frozen pre-trained weight matrix (dimensions d × k)
- `B` and `A` are trainable low-rank matrices (dimensions d × r and r × k)
- `r` is the rank, typically r << min(d, k)

### Key Benefits

1. **Memory Efficiency**: Only trains ~0.1-3% of parameters
   - 7B model: ~100M trainable params vs 7B frozen params
   - Reduces memory footprint by 3-10x

2. **Training Speed**: Faster training due to fewer parameters
   - Reduced gradient computation
   - Less memory movement
   - Faster convergence on small datasets

3. **Storage**: Adapters are tiny (10-100MB vs multi-GB models)
   - Easy to version and share
   - Multiple task-specific adapters for one base model

4. **Modularity**: Swap adapters without reloading base model
   - One base model + multiple task adapters
   - No catastrophic forgetting

### LoRA Parameters

**Rank (r)**:
- Controls adapter capacity
- Typical values: 8, 16, 32, 64
- Higher rank = more capacity but more parameters
- Rule of thumb: Start with r=16

**Alpha (α)**:
- Scaling factor for LoRA updates
- Typical values: 16, 32, 64
- Common pattern: α = 2r
- Controls how much LoRA influences the model

**Target Modules**:
- Which layers to apply LoRA
- Common: Query, Key, Value, Dense layers
- More modules = more parameters but better adaptation

### Example Configuration

```python
# Low-resource setting (16GB RAM)
rank = 8
alpha = 16
target_modules = ["query", "value"]  # ~0.5% parameters

# Balanced setting (32GB RAM)
rank = 16
alpha = 32
target_modules = ["query", "key", "value", "dense"]  # ~1-2% parameters

# High-capacity setting (64GB+ RAM)
rank = 64
alpha = 128
target_modules = ["query", "key", "value", "dense", "mlp"]  # ~3-5% parameters
```

### When to Use LoRA

**Best For**:
- Task-specific adaptation (classification, QA, summarization)
- Limited compute resources
- Multiple task variants from one base model
- Quick experimentation

**Not Ideal For**:
- Teaching completely new knowledge (use full fine-tuning)
- Dramatically changing model behavior
- When you have unlimited compute

## Documentation

For comprehensive guides and best practices, see:

- **[MLX Fine-Tuning Research Guide](docs/MLX_FINETUNING_RESEARCH_GUIDE.md)** - Complete research guide covering training configurations, loss curves, optimization strategies, and production deployment
- **[Embedding LoRA Guide](docs/EMBEDDING_LORA_GUIDE.md)** - Specialized guide for fine-tuning embedding models with LoRA for semantic search and retrieval tasks
- **[Walmart-Amazon Training Guide](docs/WALMART_AMAZON_TRAINING_GUIDE.md)** - Real-world product matching use case with complete training pipeline

## Project Structure

```
mlx-finetuning/
├── README.md                           # This file
├── CONTRIBUTING.md                     # Contributing guidelines
├── requirements.txt                    # Python dependencies
├── docs/
│   ├── MLX_FINETUNING_RESEARCH_GUIDE.md   # Research guide & best practices
│   ├── EMBEDDING_LORA_GUIDE.md            # Embedding model fine-tuning
│   └── WALMART_AMAZON_TRAINING_GUIDE.md   # Product matching use case
├── scripts/
│   ├── run_finetune.py                 # Main fine-tuning script
│   ├── download_model.py               # Download MLX models
│   ├── download_judgelm.py             # Download JudgeLM dataset
│   ├── evaluate_judge_model.py         # Evaluate judge models
│   ├── evaluate_model_detailed.py      # Detailed evaluation
│   ├── llm_judge_benchmark.py          # LLM-as-a-Judge benchmark
│   ├── train_embedding_lora.py         # Embedding-specific training
│   ├── upload_to_hf.py                 # Upload to HuggingFace
│   ├── compare_models.py               # Model comparison
│   └── test_judge.py                   # Test judge outputs
├── examples/
│   └── quick-start.sh                  # Quick start example
└── data/
    └── README.md                       # Data format documentation
```

## Training Guide

### Data Format

Training data should be in JSONL format with conversational structure:

```json
{"messages": [
  {"role": "user", "content": "Question or prompt"},
  {"role": "assistant", "content": "Expected response"}
]}
```

### Hyperparameter Guidelines

**Learning Rate**:
- Small models (0.5-1B): 1e-5 to 5e-5
- Medium models (3-7B): 1e-5 to 2e-5
- Large models (13B+): 5e-6 to 1e-5

**Batch Size**:
- Limited by memory
- 0.5B models: 16-64
- 3-7B models: 4-16
- Adjust based on available RAM

**Iterations**:
- Small datasets (<1K): 500-1000 iterations
- Medium datasets (1-10K): 1000-3000 iterations
- Large datasets (10K+): 3000-10000 iterations

**Sequence Length**:
- Shorter = faster training, less memory
- Balance task requirements vs resources
- Typical: 512 (fast), 1024 (balanced), 2048 (long context)

### Memory Optimization

**Gradient Checkpointing**:
```bash
--grad-checkpoint  # Trades compute for memory
```

**Reduce Batch Size**:
```bash
--batch-size 4  # Start small, increase if memory allows
```

**Limit Layers**:
```bash
--num-layers 8  # Only fine-tune last N layers
```

**Shorter Sequences**:
```bash
--max-seq-length 512  # Reduce if hitting OOM
```

## Evaluation

### Statistical Metrics

Basic accuracy and loss metrics are computed during training and validation.

### LLM-as-a-Judge

For evaluating generative quality, we provide an LLM-as-a-Judge framework:

```bash
python scripts/llm_benchmark.py \
  --judge-model your-model \
  --judge-adapter your-adapters \
  --evaluator-model reference-model \
  --test-data data/test.jsonl
```

This evaluates:
- Response quality and coherence
- Task-specific performance
- Reasoning ability
- Overall model utility

## Examples

### Fine-tuning a Judge Model

```bash
# Download model
mlx_lm.convert --hf-path mlx-community/Qwen2.5-0.5B-Instruct-4bit -q

# Train
python -m mlx_lm.lora \
  --model mlx_model \
  --train \
  --data data/judgelm \
  --adapter-path adapters/judge \
  --iters 2000 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --num-layers 8

# Evaluate
python scripts/llm_benchmark.py \
  --judge-model mlx_model \
  --judge-adapter adapters/judge \
  --evaluator-model mlx_model \
  --max-samples 100
```

### Multi-Task Adaptation

```bash
# Train task-specific adapters
python -m mlx_lm.lora --data task1/ --adapter-path adapters/task1 --train
python -m mlx_lm.lora --data task2/ --adapter-path adapters/task2 --train

# Use different adapters with same base model
mlx_lm.generate --model base --adapter-path adapters/task1 --prompt "..."
mlx_lm.generate --model base --adapter-path adapters/task2 --prompt "..."
```

## Supported Models

Any model from the MLX Community on Hugging Face is supported:

- **Qwen2.5**: 0.5B, 1.5B, 3B, 7B
- **Gemma**: 2B, 7B
- **Phi**: 2B, 3B
- **SmolLM**: 135M, 360M, 1.7B
- **Llama**: Various sizes

Find models at: https://huggingface.co/mlx-community

## Performance Tips

### Training Speed

1. **Reduce sequence length**: Shorter sequences train faster
2. **Increase batch size**: Better GPU utilization (if memory allows)
3. **Use fewer layers**: `--num-layers 4` for quick experiments
4. **Disable progress bars**: `--verbose False` for batch jobs

### Memory Optimization

1. **Enable gradient checkpointing**: `--grad-checkpoint`
2. **Reduce batch size**: Start with 4, increase gradually
3. **Limit trainable layers**: `--num-layers 8`
4. **Use 4-bit quantized models**: Significantly reduces memory

### Quality Improvements

1. **More training iterations**: Allow model to converge
2. **Higher rank**: `--lora-rank 32` for more capacity
3. **Learning rate tuning**: Try 5e-6, 1e-5, 2e-5, 5e-5
4. **More training data**: Quality and quantity matter

## Troubleshooting

### Out of Memory (OOM)

```bash
# Try these in order:
1. --grad-checkpoint
2. --batch-size 4
3. --num-layers 4
4. --max-seq-length 512
5. Use smaller base model
```

### Poor Performance

```bash
# Diagnose:
1. Check loss curve (should decrease steadily)
2. Evaluate on validation set frequently
3. Try different learning rates
4. Ensure data quality and format
5. Increase training iterations
```

### Slow Training

```bash
# Speed up:
1. Reduce --max-seq-length
2. Increase --batch-size (if memory allows)
3. Use --num-layers for quick experiments
4. Remove --grad-checkpoint if memory allows
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows style guidelines
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{mlx_finetuning,
  title = {MLX Fine-Tuning: Production-Ready LoRA Training for Apple Silicon},
  year = {2025},
  url = {https://github.com/rachittshah/mlx-finetuning}
}
```

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2203.06904)

## Acknowledgments

Built with:
- MLX by Apple
- MLX-LM by the MLX community
- LoRA technique by Microsoft Research
