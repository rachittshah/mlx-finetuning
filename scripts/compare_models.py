#!/usr/bin/env python3
"""
Compare base model vs fine-tuned judge model
"""

from mlx_lm import load, generate

def compare_models(adapter_path="artifacts/lora_adapters/qwen2.5-judge-fast"):
    """Compare base vs fine-tuned model"""

    test_prompt = """Compare the following two answers to the question and determine which is better.

Question: What is the capital of France?

Answer A: The capital of France is Paris, a beautiful city known for its culture, art, and the Eiffel Tower.

Answer B: Paris.

Provide a detailed explanation of which answer is better and why, then conclude with your choice (A, B, or tie)."""

    print("=" * 80)
    print("LOADING BASE MODEL (No Fine-tuning)")
    print("=" * 80)
    base_model, base_tokenizer = load("artifacts/base_model/qwen2.5-7b-4bit")

    print("\n🤖 BASE MODEL RESPONSE:")
    print("-" * 80)
    base_response = generate(
        base_model,
        base_tokenizer,
        prompt=test_prompt,
        max_tokens=300,
        temp=0.7,
        verbose=False
    )
    print(base_response)
    print("-" * 80)

    print("\n" + "=" * 80)
    print("LOADING FINE-TUNED JUDGE MODEL")
    print("=" * 80)
    judge_model, judge_tokenizer = load(
        "artifacts/base_model/qwen2.5-7b-4bit",
        adapter_path=adapter_path
    )

    print("\n* FINE-TUNED JUDGE RESPONSE:")
    print("-" * 80)
    judge_response = generate(
        judge_model,
        judge_tokenizer,
        prompt=test_prompt,
        max_tokens=300,
        temp=0.7,
        verbose=False
    )
    print(judge_response)
    print("-" * 80)

    print("\n* Comparison complete!")
    print("\nNOTE: The fine-tuned model should:")
    print("  1. Provide structured reasoning")
    print("  2. Compare both answers systematically")
    print("  3. End with a clear verdict (A, B, or tie)")

if __name__ == "__main__":
    compare_models()
