#!/usr/bin/env python3
"""
LLM-as-a-Judge Benchmark for JudgeLM Fine-Tuned Models

Uses a stronger reference model to evaluate the quality of judge outputs on:
- Reasoning quality
- Rating accuracy
- Verdict justification
- Explanation coherence
- Overall judge performance
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from mlx_lm import load, generate
from tqdm import tqdm
import argparse
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent

EVALUATION_RUBRIC = """
# Judge Evaluation Rubric

You are evaluating a judge model's response to a comparison task. Rate the judge on these criteria:

## 1. Rating Accuracy (1-5)
- Are the numerical ratings (1-10) reasonable for each answer?
- Do the ratings reflect the quality differences described?
- Are ratings justified by the explanation?

## 2. Verdict Correctness (1-5)
- Is the final verdict (A/B/tie) consistent with the ratings?
- Does the verdict match the quality analysis?
- Is the winner choice reasonable?

## 3. Reasoning Quality (1-5)
- Does the explanation identify specific strengths/weaknesses?
- Are comparisons concrete and detailed?
- Is the analysis fair and balanced?

## 4. Explanation Coherence (1-5)
- Is the explanation well-structured and clear?
- Does it flow logically from analysis to verdict?
- Are grammar and formatting appropriate?

## 5. Overall Judge Quality (1-5)
- Would this judgment be helpful to a user?
- Does it demonstrate good judgment ability?
- Is it comparable to human-level evaluation?

## Scoring Guidelines
5 - Excellent: Professional quality, insightful analysis
4 - Good: Solid evaluation with minor issues
3 - Adequate: Acceptable but with notable gaps
2 - Poor: Significant problems or errors
1 - Very Poor: Fails basic requirements

Provide scores and brief justifications for each criterion, then an overall assessment.
"""


@dataclass
class JudgeEvaluation:
    rating_accuracy: int
    verdict_correctness: int
    reasoning_quality: int
    explanation_coherence: int
    overall_quality: int
    justification: str
    overall_score: float


def create_evaluation_prompt(question: str, answer_a: str, answer_b: str, judge_response: str) -> str:
    """Create prompt for evaluating a judge's response."""
    prompt = f"""{EVALUATION_RUBRIC}

# Task to Evaluate

-Original Question:-
{question}

-Answer A:-
{answer_a[:500]}...

-Answer B:-
{answer_b[:500]}...

---

# Judge's Response to Evaluate:

{judge_response}

---

# Your Evaluation

Provide your evaluation in this format:

-Rating Accuracy:- [1-5]
[Brief justification]

-Verdict Correctness:- [1-5]
[Brief justification]

-Reasoning Quality:- [1-5]
[Brief justification]

-Explanation Coherence:- [1-5]
[Brief justification]

-Overall Judge Quality:- [1-5]
[Brief justification]

-Summary:-
[Overall assessment of this judge's performance]
"""
    return prompt


def parse_evaluation(response: str) -> JudgeEvaluation:
    """Parse evaluation response into structured format."""
    # Extract scores
    rating_accuracy = extract_score(response, "Rating Accuracy")
    verdict_correctness = extract_score(response, "Verdict Correctness")
    reasoning_quality = extract_score(response, "Reasoning Quality")
    explanation_coherence = extract_score(response, "Explanation Coherence")
    overall_quality = extract_score(response, "Overall Judge Quality")

    # Calculate overall score
    overall_score = (
        rating_accuracy + verdict_correctness + reasoning_quality +
        explanation_coherence + overall_quality
    ) / 5.0

    # Extract summary
    summary_match = re.search(r'\*\*Summary:\*\*\s*(.+?)(?=\n\n|\Z)', response, re.DOTALL)
    justification = summary_match.group(1).strip() if summary_match else "No summary provided"

    return JudgeEvaluation(
        rating_accuracy=rating_accuracy,
        verdict_correctness=verdict_correctness,
        reasoning_quality=reasoning_quality,
        explanation_coherence=explanation_coherence,
        overall_quality=overall_quality,
        justification=justification,
        overall_score=overall_score
    )


def extract_score(text: str, criterion: str) -> int:
    """Extract score for a specific criterion."""
    # Look for pattern like "-Criterion:- [1-5]" or "-Criterion:- 3"
    pattern = rf'\*\*{criterion}:\*\*\s*\[?(\d)\]?'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return 3  # Default to middle score if not found


def extract_question_and_answers(prompt: str) -> Tuple[str, str, str]:
    """Extract question and answers from comparison prompt."""
    # Extract question
    q_match = re.search(r'Question:\s*\n(.+?)\n\nAnswer A:', prompt, re.DOTALL)
    question = q_match.group(1).strip() if q_match else "Unknown question"

    # Extract Answer A
    a_match = re.search(r'Answer A:\s*(.+?)\n\nAnswer B:', prompt, re.DOTALL)
    answer_a = a_match.group(1).strip() if a_match else "Unknown answer"

    # Extract Answer B
    b_match = re.search(r'Answer B:\s*(.+?)(?=\n\nProvide|$)', prompt, re.DOTALL)
    answer_b = b_match.group(1).strip() if b_match else "Unknown answer"

    return question, answer_a, answer_b


def load_test_data(data_path: Path, max_samples: int = None) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def run_benchmark(
    judge_model,
    judge_tokenizer,
    evaluator_model,
    evaluator_tokenizer,
    test_data: List[Dict],
    max_tokens: int = 200,
) -> Dict:
    """Run LLM-as-a-Judge benchmark."""

    evaluations = []

    print(f"\n* Running LLM-as-a-Judge benchmark on {len(test_data)} samples...")
    print("  Step 1: Generate judgments from fine-tuned model")
    print("  Step 2: Evaluate judgments with reference model")

    for sample in tqdm(test_data, desc="Benchmarking"):
        messages = sample['messages']
        prompt = messages[0]['content']

        # Extract question and answers
        question, answer_a, answer_b = extract_question_and_answers(prompt)

        # Step 1: Get judge's response
        try:
            judge_response = generate(
                judge_model,
                judge_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        except Exception as e:
            print(f"\n* Error generating judge response: {e}")
            continue

        # Step 2: Evaluate judge's response
        eval_prompt = create_evaluation_prompt(question, answer_a, answer_b, judge_response)

        try:
            eval_response = generate(
                evaluator_model,
                evaluator_tokenizer,
                prompt=eval_prompt,
                max_tokens=400,
                verbose=False
            )

            evaluation = parse_evaluation(eval_response)

            evaluations.append({
                'prompt': prompt[:200] + "...",
                'judge_response': judge_response,
                'evaluation': {
                    'rating_accuracy': evaluation.rating_accuracy,
                    'verdict_correctness': evaluation.verdict_correctness,
                    'reasoning_quality': evaluation.reasoning_quality,
                    'explanation_coherence': evaluation.explanation_coherence,
                    'overall_quality': evaluation.overall_quality,
                    'overall_score': evaluation.overall_score,
                    'justification': evaluation.justification
                },
                'evaluator_response': eval_response
            })

        except Exception as e:
            print(f"\n* Error evaluating response: {e}")
            continue

    # Calculate aggregate metrics
    if not evaluations:
        return {'error': 'No successful evaluations'}

    metrics = calculate_aggregate_metrics(evaluations)

    return {
        'metrics': metrics,
        'evaluations': evaluations,
        'total_samples': len(evaluations)
    }


def calculate_aggregate_metrics(evaluations: List[Dict]) -> Dict:
    """Calculate aggregate metrics from evaluations."""
    rating_accuracy = []
    verdict_correctness = []
    reasoning_quality = []
    explanation_coherence = []
    overall_quality = []
    overall_scores = []

    for eval_data in evaluations:
        evaluation = eval_data['evaluation']
        rating_accuracy.append(evaluation['rating_accuracy'])
        verdict_correctness.append(evaluation['verdict_correctness'])
        reasoning_quality.append(evaluation['reasoning_quality'])
        explanation_coherence.append(evaluation['explanation_coherence'])
        overall_quality.append(evaluation['overall_quality'])
        overall_scores.append(evaluation['overall_score'])

    def avg(lst): return sum(lst) / len(lst) if lst else 0
    def pct_good(lst): return sum(1 for x in lst if x >= 4) / len(lst) * 100 if lst else 0

    return {
        'averages': {
            'rating_accuracy': avg(rating_accuracy),
            'verdict_correctness': avg(verdict_correctness),
            'reasoning_quality': avg(reasoning_quality),
            'explanation_coherence': avg(explanation_coherence),
            'overall_quality': avg(overall_quality),
            'overall_score': avg(overall_scores)
        },
        'percent_good_or_better': {
            'rating_accuracy': pct_good(rating_accuracy),
            'verdict_correctness': pct_good(verdict_correctness),
            'reasoning_quality': pct_good(reasoning_quality),
            'explanation_coherence': pct_good(explanation_coherence),
            'overall_quality': pct_good(overall_quality)
        },
        'distribution': {
            'excellent_5': sum(1 for x in overall_quality if x == 5),
            'good_4': sum(1 for x in overall_quality if x == 4),
            'adequate_3': sum(1 for x in overall_quality if x == 3),
            'poor_2': sum(1 for x in overall_quality if x == 2),
            'very_poor_1': sum(1 for x in overall_quality if x == 1)
        }
    }


def print_results(results: Dict):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("* LLM-AS-A-JUDGE BENCHMARK RESULTS")
    print("=" * 80)

    metrics = results['metrics']
    averages = metrics['averages']
    pct_good = metrics['percent_good_or_better']
    dist = metrics['distribution']

    print(f"\n* AVERAGE SCORES (out of 5):")
    print(f"  Rating Accuracy:      {averages['rating_accuracy']:.2f} ({pct_good['rating_accuracy']:.0f}% good+)")
    print(f"  Verdict Correctness:  {averages['verdict_correctness']:.2f} ({pct_good['verdict_correctness']:.0f}% good+)")
    print(f"  Reasoning Quality:    {averages['reasoning_quality']:.2f} ({pct_good['reasoning_quality']:.0f}% good+)")
    print(f"  Explanation Coherence:{averages['explanation_coherence']:.2f} ({pct_good['explanation_coherence']:.0f}% good+)")
    print(f"  Overall Quality:      {averages['overall_quality']:.2f} ({pct_good['overall_quality']:.0f}% good+)")

    print(f"\n* OVERALL SCORE: {averages['overall_score']:.2f} / 5.0")

    print(f"\n* QUALITY DISTRIBUTION:")
    total = sum(dist.values())
    print(f"  ---- Excellent (5): {dist['excellent_5']:3d} ({dist['excellent_5']/total*100:.0f}%)")
    print(f"  -   Good (4):      {dist['good_4']:3d} ({dist['good_4']/total*100:.0f}%)")
    print(f"  -     Adequate (3):  {dist['adequate_3']:3d} ({dist['adequate_3']/total*100:.0f}%)")
    print(f"  -       Poor (2):      {dist['poor_2']:3d} ({dist['poor_2']/total*100:.0f}%)")
    print(f"  *         Very Poor (1): {dist['very_poor_1']:3d} ({dist['very_poor_1']/total*100:.0f}%)")

    # Benchmark grade
    overall = averages['overall_score']
    if overall >= 4.5:
        grade = "* EXCELLENT - Production Ready"
    elif overall >= 4.0:
        grade = "* GOOD - Strong Performance"
    elif overall >= 3.5:
        grade = "* ACCEPTABLE - Usable"
    elif overall >= 3.0:
        grade = "* FAIR - Needs Improvement"
    else:
        grade = "* POOR - Not Recommended"

    print(f"\n* BENCHMARK GRADE: {grade}")
    print(f"   Total Samples: {results['total_samples']}")

    print("\n" + "=" * 80)


def save_results(results: Dict, output_path: Path):
    """Save benchmark results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n* Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Benchmark")
    parser.add_argument('--judge-model', type=str, required=True, help="Path to judge model")
    parser.add_argument('--judge-adapter', type=str, required=True, help="Path to judge adapter")
    parser.add_argument('--evaluator-model', type=str, required=True, help="Path to evaluator model (stronger reference)")
    parser.add_argument('--evaluator-adapter', type=str, default=None, help="Path to evaluator adapter (optional)")
    parser.add_argument('--test-data', type=str, default='data/judgelm_val.jsonl', help="Path to test data")
    parser.add_argument('--max-samples', type=int, default=None, help="Max samples to evaluate")
    parser.add_argument('--max-tokens', type=int, default=200, help="Max tokens for judge")
    parser.add_argument('--output', type=str, default=None, help="Output path")

    args = parser.parse_args()

    print("=" * 80)
    print("* LLM-AS-A-JUDGE BENCHMARK")
    print("=" * 80)

    # Load judge model
    print(f"\n* Loading judge model: {args.judge_model}")
    print(f"* Loading judge adapter: {args.judge_adapter}")
    judge_model, judge_tokenizer = load(args.judge_model, adapter_path=args.judge_adapter)
    print("* Judge model loaded")

    # Load evaluator model
    print(f"\n* Loading evaluator model: {args.evaluator_model}")
    if args.evaluator_adapter:
        print(f"* Loading evaluator adapter: {args.evaluator_adapter}")
        evaluator_model, evaluator_tokenizer = load(args.evaluator_model, adapter_path=args.evaluator_adapter)
    else:
        evaluator_model, evaluator_tokenizer = load(args.evaluator_model)
    print("* Evaluator model loaded")

    # Load test data
    test_data_path = PROJECT_ROOT / args.test_data
    print(f"\n* Loading test data: {test_data_path}")
    test_data = load_test_data(test_data_path, args.max_samples)
    print(f"* Loaded {len(test_data)} test samples")

    # Run benchmark
    results = run_benchmark(
        judge_model,
        judge_tokenizer,
        evaluator_model,
        evaluator_tokenizer,
        test_data,
        max_tokens=args.max_tokens
    )

    if 'error' in results:
        print(f"\n* Benchmark failed: {results['error']}")
        return

    # Print results
    print_results(results)

    # Save results
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        judge_name = Path(args.judge_adapter).name
        output_path = PROJECT_ROOT / "artifacts" / "benchmarks" / f"{judge_name}_llm_bench.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)

    print("\n* Benchmark complete!")


if __name__ == "__main__":
    main()
