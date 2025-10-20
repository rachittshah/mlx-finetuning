#!/usr/bin/env python3
"""
Train embedding model with LoRA on Walmart-Amazon product matching

Uses sentence-transformers + PEFT for LoRA fine-tuning
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "walmart_amazon_synthetic"

def load_triplets(filename):
    """Load triplet data from JSON"""
    with open(filename) as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append(InputExample(texts=[
            item["anchor"],
            item["positive"],
            item["negative"]
        ]))

    return examples

def train():
    print("=" * 80)
    print("EMBEDDING LoRA FINE-TUNING - WALMART-AMAZON PRODUCT MATCHING")
    print("=" * 80)

    # Load base model
    print("\n* Loading base model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Apply LoRA to the transformer
    print("* Applying LoRA adapters...")
    base_model = model[0].auto_model

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,                    # LoRA rank
        lora_alpha=16,          # Scaling
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"],  # Apply to attention + FFN
        bias="none"
    )

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    # Replace model
    model[0].auto_model = lora_model

    # Load data
    print(f"\n* Loading training data from {DATA_DIR}...")
    train_examples = load_triplets(DATA_DIR / "train.json")
    val_examples = load_triplets(DATA_DIR / "val.json")

    print(f"✓ Train examples: {len(train_examples)}")
    print(f"✓ Val examples: {len(val_examples)}")

    # Create dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    # Define loss (Triplet loss for product matching)
    train_loss = losses.TripletLoss(model)

    # Train
    print("\n* Starting training...")
    print(f"  - Epochs: 10")
    print(f"  - Batch size: 8")
    print(f"  - Loss: Triplet Loss")
    print(f"  - LoRA rank: 8")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=10,
        output_path=str(PROJECT_ROOT / "artifacts" / "embedding_lora" / "walmart-amazon"),
        show_progress_bar=True,
        save_best_model=True
    )

    print("\n* Training complete!")
    print(f"* Model saved to: artifacts/embedding_lora/walmart-amazon")

def test():
    """Test the trained model"""
    print("=" * 80)
    print("TESTING EMBEDDING MODEL")
    print("=" * 80)

    model_path = PROJECT_ROOT / "artifacts" / "embedding_lora" / "walmart-amazon"

    print(f"\n* Loading model from {model_path}...")
    model = SentenceTransformer(str(model_path))

    # Test cases
    test_cases = [
        {
            "walmart": "Samsung 55-Inch 4K TV UN55CU7000",
            "amazon_match": "Samsung 55 Inch Crystal 4K UHD TV UN55CU7000",
            "amazon_no_match": "LG 55-Inch OLED TV"
        },
        {
            "walmart": "Apple AirPods Pro 2nd Gen",
            "amazon_match": "Apple AirPods Pro (2nd Generation)",
            "amazon_no_match": "Sony WH-1000XM5 Headphones"
        }
    ]

    print("\n* Test Results:")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        walmart_emb = model.encode(case["walmart"])
        match_emb = model.encode(case["amazon_match"])
        no_match_emb = model.encode(case["amazon_no_match"])

        # Compute similarities
        match_score = torch.nn.functional.cosine_similarity(
            torch.tensor(walmart_emb).unsqueeze(0),
            torch.tensor(match_emb).unsqueeze(0)
        ).item()

        no_match_score = torch.nn.functional.cosine_similarity(
            torch.tensor(walmart_emb).unsqueeze(0),
            torch.tensor(no_match_emb).unsqueeze(0)
        ).item()

        print(f"\nTest {i}:")
        print(f"  Walmart: {case['walmart']}")
        print(f"  Match:     {case['amazon_match']} → similarity: {match_score:.3f}")
        print(f"  No Match:  {case['amazon_no_match']} → similarity: {no_match_score:.3f}")
        print(f"  {'* CORRECT' if match_score > no_match_score else '* WRONG'}")

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test()
    else:
        train()
        test()
