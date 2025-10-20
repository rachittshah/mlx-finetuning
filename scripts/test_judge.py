#!/usr/bin/env python3
"""
Test your fine-tuned judge model with MLX
"""

from mlx_lm import load, generate
import sys

def test_judge(adapter_path="artifacts/lora_adapters/qwen2.5-judge-fast"):
    """Test the judge model with sample prompts"""

    print("🔄 Loading model and adapter...")
    model, tokenizer = load(
        "artifacts/base_model/qwen2.5-7b-4bit",
        adapter_path=adapter_path
    )
    print("* Model loaded!\n")

    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "answer_a": "The capital of France is Paris, a beautiful city known for its culture, art, and the Eiffel Tower.",
            "answer_b": "Paris."
        },
        {
            "question": "How do I make a perfect scrambled eggs?",
            "answer_a": "Beat eggs with milk, cook in butter over medium heat, stir constantly until just set. Season with salt and pepper.",
            "answer_b": "Put eggs in pan and cook them."
        },
        {
            "question": "Explain quantum computing in simple terms.",
            "answer_a": "Quantum computers use qubits that can be 0 and 1 simultaneously, enabling parallel computation.",
            "answer_b": "Quantum computers use quantum mechanics principles like superposition and entanglement to process information differently than classical computers. While classical bits are either 0 or 1, qubits can exist in multiple states simultaneously, allowing quantum computers to solve certain problems exponentially faster."
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")

        prompt = f"""Compare the following two answers to the question and determine which is better.

Question: {case['question']}

Answer A: {case['answer_a']}

Answer B: {case['answer_b']}

Provide a detailed explanation of which answer is better and why, then conclude with your choice (A, B, or tie)."""

        print(f"\nQuestion: {case['question']}")
        print(f"\nAnswer A: {case['answer_a']}")
        print(f"\nAnswer B: {case['answer_b']}")
        print("\n🤖 JUDGE'S VERDICT:")
        print("-" * 80)

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=400,
            temp=0.7,
            verbose=False
        )
        print(response)
        print("-" * 80)

    print("\n* All tests completed!")

if __name__ == "__main__":
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/lora_adapters/qwen2.5-judge-fast"
    test_judge(adapter_path)
