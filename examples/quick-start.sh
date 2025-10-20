#!/bin/bash
# Quick Start Example for MLX Fine-Tuning

set -e

echo "MLX Fine-Tuning Quick Start"
echo "=============================="
echo ""

# Step 1: Setup
echo "[1/5] Setting up environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -q mlx-lm numpy scipy scikit-learn tqdm

# Step 2: Download Model
echo "[2/5] Downloading Qwen2.5-0.5B-Instruct-4bit..."
mlx_lm.convert --hf-path mlx-community/Qwen2.5-0.5B-Instruct-4bit -q

# Step 3: Prepare Sample Data
echo "[3/5] Creating sample training data..."
mkdir -p data
cat > data/train.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "Explain gravity in simple terms."}, {"role": "assistant", "content": "Gravity is a force that pulls objects toward each other. On Earth, it's what keeps us on the ground and makes things fall down."}]}
{"messages": [{"role": "user", "content": "What is 15 x 12?"}, {"role": "assistant", "content": "15 x 12 = 180"}]}
EOF

cat > data/valid.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "What is the largest ocean?"}, {"role": "assistant", "content": "The Pacific Ocean is the largest ocean on Earth."}]}
EOF

# Step 4: Fine-tune
echo "[4/5] Fine-tuning model (this will take a few minutes)..."
python -m mlx_lm.lora \
  --model mlx_model \
  --train \
  --data data/ \
  --adapter-path adapters/quick-start \
  --iters 100 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --steps-per-report 20

# Step 5: Test
echo "[5/5] Testing fine-tuned model..."
echo ""
echo "Generating response to: 'What is machine learning?'"
echo ""
mlx_lm.generate \
  --model mlx_model \
  --adapter-path adapters/quick-start \
  --prompt "What is machine learning?" \
  --max-tokens 100

echo ""
echo "=============================="
echo "Quick start complete!"
echo "Your fine-tuned adapters are in: adapters/quick-start/"
echo ""
echo "To use your model:"
echo "  mlx_lm.generate --model mlx_model --adapter-path adapters/quick-start --prompt 'Your prompt'"
