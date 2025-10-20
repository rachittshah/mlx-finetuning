#!/usr/bin/env python3
"""
Download JudgeLM-100K dataset for training LLM-as-a-Judge models

This dataset contains 100K high-quality judge samples with GPT-4-generated judgements.
Perfect for fine-tuning models to evaluate and compare LLM responses.

Usage:
    python download_judgelm.py --samples 10000
    python download_judgelm.py --split train  # Full training set (100k)
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
import sys

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"


def format_judgelm_to_chat(example):
    """
    Convert JudgeLM examples to chat format for fine-tuning

    JudgeLM format includes:
    - question_body: The user question
    - answer1_body/answer2_body: Two candidate answers
    - text: GPT-4 generated reasoning/explanation
    - score: Which answer is better
    """
    question = example.get("question_body", "")
    answer1 = example.get("answer1_body", "")
    answer2 = example.get("answer2_body", "")
    explanation = example.get("text", "")

    # Parse score to determine choice (score is [score1, score2])
    scores = example.get("score", [0, 0])
    if scores[0] > scores[1]:
        choice = "A"
    elif scores[1] > scores[0]:
        choice = "B"
    else:
        choice = "tie"

    # Create instruction for judging
    judge_instruction = f"""Compare the following two answers to the question and determine which is better.

Question: {question}

Answer A: {answer1}

Answer B: {answer2}

Provide a detailed explanation of which answer is better and why, then conclude with your choice (A, B, or tie)."""

    # Create expected response (GPT-4 judgement)
    judge_response = f"""{explanation}

**Verdict:** {choice}"""

    return {
        "messages": [
            {"role": "user", "content": judge_instruction},
            {"role": "assistant", "content": judge_response}
        ]
    }


def download_judgelm(samples: int = None, split: str = "train", train_val_split: float = 0.95):
    """
    Download and format JudgeLM dataset

    Args:
        samples: Limit to N samples (None = all)
        split: Dataset split ('train' or 'val')
        train_val_split: Ratio for train/validation split
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = "judgelm.jsonl"
    train_file = DATA_DIR / output_file
    val_file = DATA_DIR / output_file.replace(".jsonl", "_val.jsonl")

    print(f"\n* Downloading JudgeLM-100K dataset from BAAI")
    print(f"* Split: {split}")
    if samples:
        print(f"📏 Limiting to: {samples} samples")

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("BAAI/JudgeLM-100K", split=split, streaming=samples is not None)

        # Convert to list and limit samples if needed
        if samples:
            print(f"⏳ Processing {samples} samples...")
            examples = []
            for i, example in enumerate(dataset):
                if i >= samples:
                    break
                examples.append(example)
        else:
            print(f"⏳ Processing all samples (this may take a while)...")
            examples = list(dataset)

        print(f"✓ Loaded {len(examples)} examples")

        # Convert to chat format
        print("🔄 Converting to chat format for fine-tuning...")
        formatted_examples = []
        for example in examples:
            try:
                formatted = format_judgelm_to_chat(example)
                formatted_examples.append(formatted)
            except Exception as e:
                print(f"⚠*  Skipping example due to formatting error: {e}")
                continue

        print(f"✓ Formatted {len(formatted_examples)} examples")

        # Split into train/val
        split_idx = int(len(formatted_examples) * train_val_split)
        train_examples = formatted_examples[:split_idx]
        val_examples = formatted_examples[split_idx:]

        print(f"\n* Writing files...")
        print(f"  Train: {len(train_examples)} examples → {train_file}")
        print(f"  Val:   {len(val_examples)} examples → {val_file}")

        # Write train file
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        # Write validation file
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"\n* Successfully downloaded JudgeLM dataset!")
        print(f"\n📚 About this dataset:")
        print(f"   - 100K+ judge samples with GPT-4 judgements")
        print(f"   - Trains models to evaluate and compare responses")
        print(f"   - Agreement exceeding 90% (surpasses human-to-human)")
        print(f"   - Published at ICLR 2025 (Spotlight paper)")
        print(f"\n* Use these files in the GUI:")
        print(f"   Training data: {train_file}")
        print(f"   Validation data: {val_file}")

    except Exception as e:
        print(f"\n* Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download JudgeLM-100K dataset for LLM-as-a-Judge training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 10,000 samples for quick testing
  python download_judgelm.py --samples 10000

  # Download full training set (100K samples)
  python download_judgelm.py

  # Download validation set
  python download_judgelm.py --split val

About JudgeLM:
  JudgeLM is an ICLR 2025 Spotlight paper that provides high-quality
  judge training data. Models trained on this can evaluate LLM outputs
  with >90% agreement, surpassing human-to-human agreement.

  Dataset: https://huggingface.co/datasets/BAAI/JudgeLM-100K
  Paper: https://arxiv.org/abs/2310.17631
        """
    )

    parser.add_argument(
        "--samples",
        type=int,
        help="Limit to N samples (downloads all 100K if not specified)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to download (default: train)"
    )

    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.95,
        help="Train/validation split ratio (default: 0.95)"
    )

    args = parser.parse_args()

    download_judgelm(args.samples, args.split, args.train_val_split)


if __name__ == "__main__":
    main()
