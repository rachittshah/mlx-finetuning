#!/usr/bin/env python3
"""
Production-Grade Evaluation Suite for JudgeLM Fine-Tuned Models

Evaluates model performance on:
- Rating accuracy (MAE, RMSE)
- Correlation metrics (Pearson, Spearman)
- Winner prediction accuracy
- Response quality metrics
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from mlx_lm import load, generate
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).parent.parent


def extract_ratings_and_verdict(response: str) -> Tuple[int, int, str]:
    """
    Extract ratings and verdict from model response.

    Expected format:
    "rating_A rating_B\n... explanation ...\n**Verdict:** A/B/tie"

    Returns: (rating_A, rating_B, verdict)
    """
    # Extract first two numbers as ratings
    numbers = re.findall(r'\b([0-9]|10)\b', response)
    rating_a = int(numbers[0]) if len(numbers) >= 1 else 5
    rating_b = int(numbers[1]) if len(numbers) >= 2 else 5

    # Extract verdict (A, B, or tie)
    verdict_match = re.search(r'\*\*Verdict:\*\*\s*([ABab]|tie)', response, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
        if verdict == 'TIE':
            verdict = 'tie'
    else:
        # Fallback: determine from ratings
        if rating_a > rating_b:
            verdict = 'A'
        elif rating_b > rating_a:
            verdict = 'B'
        else:
            verdict = 'tie'

    return rating_a, rating_b, verdict


def parse_ground_truth(response: str) -> Tuple[int, int, str]:
    """Extract ground truth ratings and verdict from dataset."""
    return extract_ratings_and_verdict(response)


def load_test_data(data_path: Path, max_samples: int = None) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    max_tokens: int = 200,
    temperature: float = 0.1,
) -> Dict:
    """
    Evaluate model on test data.

    Returns metrics dictionary.
    """
    predictions = {
        'rating_a': [],
        'rating_b': [],
        'verdict': []
    }

    ground_truth = {
        'rating_a': [],
        'rating_b': [],
        'verdict': []
    }

    print(f"\n🔮 Running inference on {len(test_data)} test samples...")

    for sample in tqdm(test_data, desc="Evaluating"):
        # Extract prompt and ground truth
        messages = sample['messages']
        prompt = messages[0]['content']  # User message
        gt_response = messages[1]['content']  # Assistant response

        # Get ground truth
        gt_rating_a, gt_rating_b, gt_verdict = parse_ground_truth(gt_response)
        ground_truth['rating_a'].append(gt_rating_a)
        ground_truth['rating_b'].append(gt_rating_b)
        ground_truth['verdict'].append(gt_verdict)

        # Generate prediction
        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )

            # Extract predictions
            pred_rating_a, pred_rating_b, pred_verdict = extract_ratings_and_verdict(response)
            predictions['rating_a'].append(pred_rating_a)
            predictions['rating_b'].append(pred_rating_b)
            predictions['verdict'].append(pred_verdict)

        except Exception as e:
            print(f"\n* Error generating response: {e}")
            # Use default values
            predictions['rating_a'].append(5)
            predictions['rating_b'].append(5)
            predictions['verdict'].append('tie')

    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)

    return metrics, predictions, ground_truth


def calculate_metrics(ground_truth: Dict, predictions: Dict) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    """
    gt_a = np.array(ground_truth['rating_a'])
    gt_b = np.array(ground_truth['rating_b'])
    pred_a = np.array(predictions['rating_a'])
    pred_b = np.array(predictions['rating_b'])

    # Rating metrics
    mae_a = mean_absolute_error(gt_a, pred_a)
    mae_b = mean_absolute_error(gt_b, pred_b)
    mae_overall = (mae_a + mae_b) / 2

    rmse_a = np.sqrt(mean_squared_error(gt_a, pred_a))
    rmse_b = np.sqrt(mean_squared_error(gt_b, pred_b))
    rmse_overall = (rmse_a + rmse_b) / 2

    # Correlation metrics
    pearson_a, pearson_a_pval = stats.pearsonr(gt_a, pred_a)
    pearson_b, pearson_b_pval = stats.pearsonr(gt_b, pred_b)

    spearman_a, spearman_a_pval = stats.spearmanr(gt_a, pred_a)
    spearman_b, spearman_b_pval = stats.spearmanr(gt_b, pred_b)

    # Winner prediction accuracy
    gt_verdicts = ground_truth['verdict']
    pred_verdicts = predictions['verdict']
    verdict_accuracy = accuracy_score(gt_verdicts, pred_verdicts)

    # Confusion matrix for verdicts
    unique_labels = sorted(list(set(gt_verdicts + pred_verdicts)))
    cm = confusion_matrix(gt_verdicts, pred_verdicts, labels=unique_labels)

    # Per-class metrics
    verdict_counts = {
        'A': {'correct': 0, 'total': 0},
        'B': {'correct': 0, 'total': 0},
        'tie': {'correct': 0, 'total': 0}
    }

    for gt, pred in zip(gt_verdicts, pred_verdicts):
        verdict_counts[gt]['total'] += 1
        if gt == pred:
            verdict_counts[gt]['correct'] += 1

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for verdict in verdict_counts:
        total = verdict_counts[verdict]['total']
        if total > 0:
            per_class_accuracy[verdict] = verdict_counts[verdict]['correct'] / total
        else:
            per_class_accuracy[verdict] = 0.0

    # Rating distribution analysis
    rating_diff_gt = gt_a - gt_b
    rating_diff_pred = pred_a - pred_b
    diff_correlation, _ = stats.pearsonr(rating_diff_gt, rating_diff_pred)

    metrics = {
        'rating_metrics': {
            'mae': {
                'answer_a': float(mae_a),
                'answer_b': float(mae_b),
                'overall': float(mae_overall)
            },
            'rmse': {
                'answer_a': float(rmse_a),
                'answer_b': float(rmse_b),
                'overall': float(rmse_overall)
            },
            'pearson_correlation': {
                'answer_a': float(pearson_a),
                'answer_b': float(pearson_b),
                'p_value_a': float(pearson_a_pval),
                'p_value_b': float(pearson_b_pval)
            },
            'spearman_correlation': {
                'answer_a': float(spearman_a),
                'answer_b': float(spearman_b),
                'p_value_a': float(spearman_a_pval),
                'p_value_b': float(spearman_b_pval)
            }
        },
        'verdict_metrics': {
            'accuracy': float(verdict_accuracy),
            'per_class_accuracy': per_class_accuracy,
            'confusion_matrix': cm.tolist(),
            'labels': unique_labels,
            'counts': verdict_counts
        },
        'rating_difference': {
            'correlation': float(diff_correlation),
            'description': 'Correlation between ground truth and predicted rating differences'
        },
        'summary': {
            'total_samples': len(gt_a),
            'mae': float(mae_overall),
            'verdict_accuracy': float(verdict_accuracy),
            'rating_correlation': float((pearson_a + pearson_b) / 2)
        }
    }

    return metrics


def print_metrics(metrics: Dict):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 80)
    print("* PRODUCTION EVALUATION RESULTS")
    print("=" * 80)

    summary = metrics['summary']
    print(f"\n* SUMMARY METRICS:")
    print(f"  - Total Samples: {summary['total_samples']}")
    print(f"  - Mean Absolute Error (MAE): {summary['mae']:.3f}")
    print(f"  - Winner Prediction Accuracy: {summary['verdict_accuracy']:.1%}")
    print(f"  - Average Rating Correlation: {summary['rating_correlation']:.3f}")

    rating_metrics = metrics['rating_metrics']
    print(f"\n* RATING ACCURACY:")
    print(f"  MAE (Answer A): {rating_metrics['mae']['answer_a']:.3f}")
    print(f"  MAE (Answer B): {rating_metrics['mae']['answer_b']:.3f}")
    print(f"  RMSE (Answer A): {rating_metrics['rmse']['answer_a']:.3f}")
    print(f"  RMSE (Answer B): {rating_metrics['rmse']['answer_b']:.3f}")

    print(f"\n🔗 CORRELATION METRICS:")
    print(f"  Pearson (Answer A): {rating_metrics['pearson_correlation']['answer_a']:.3f}")
    print(f"  Pearson (Answer B): {rating_metrics['pearson_correlation']['answer_b']:.3f}")
    print(f"  Spearman (Answer A): {rating_metrics['spearman_correlation']['answer_a']:.3f}")
    print(f"  Spearman (Answer B): {rating_metrics['spearman_correlation']['answer_b']:.3f}")

    verdict_metrics = metrics['verdict_metrics']
    print(f"\n* WINNER PREDICTION:")
    print(f"  Overall Accuracy: {verdict_metrics['accuracy']:.1%}")
    print(f"  Per-Class Accuracy:")
    for verdict, acc in verdict_metrics['per_class_accuracy'].items():
        count = verdict_metrics['counts'][verdict]['total']
        print(f"    - {verdict}: {acc:.1%} ({count} samples)")

    print(f"\n🔢 CONFUSION MATRIX:")
    cm = verdict_metrics['confusion_matrix']
    labels = verdict_metrics['labels']
    print(f"  Predicted →")
    print(f"  Actual ↓   {' '.join(f'{l:>6}' for l in labels)}")
    for i, label in enumerate(labels):
        row = '  '.join(f'{cm[i][j]:>6}' for j in range(len(labels)))
        print(f"  {label:>6}     {row}")

    print("\n" + "=" * 80)


def save_results(metrics: Dict, predictions: Dict, ground_truth: Dict, output_path: Path):
    """Save evaluation results to JSON file."""
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n* Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JudgeLM fine-tuned model")
    parser.add_argument('--model', type=str, required=True, help="Path to base model")
    parser.add_argument('--adapter-path', type=str, required=True, help="Path to adapter weights")
    parser.add_argument('--test-data', type=str, default='data/judgelm_val.jsonl', help="Path to test data")
    parser.add_argument('--max-samples', type=int, default=None, help="Max number of samples to evaluate")
    parser.add_argument('--max-tokens', type=int, default=200, help="Max tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.1, help="Sampling temperature")
    parser.add_argument('--output', type=str, default=None, help="Output path for results")

    args = parser.parse_args()

    print("=" * 80)
    print("* PRODUCTION-GRADE JUDGELM EVALUATION")
    print("=" * 80)

    # Load model
    print(f"\n* Loading model: {args.model}")
    print(f"* Loading adapters: {args.adapter_path}")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    print("✓ Model loaded successfully")

    # Load test data
    test_data_path = PROJECT_ROOT / args.test_data
    print(f"\n* Loading test data: {test_data_path}")
    test_data = load_test_data(test_data_path, args.max_samples)
    print(f"✓ Loaded {len(test_data)} test samples")

    # Run evaluation
    metrics, predictions, ground_truth = evaluate_model(
        model,
        tokenizer,
        test_data,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Print results
    print_metrics(metrics)

    # Save results
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        adapter_name = Path(args.adapter_path).name
        output_path = PROJECT_ROOT / "artifacts" / "evaluations" / f"{adapter_name}_eval.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(metrics, predictions, ground_truth, output_path)

    # Benchmark assessment
    print("\n* BENCHMARK ASSESSMENT:")
    mae = metrics['summary']['mae']
    verdict_acc = metrics['summary']['verdict_accuracy']
    correlation = metrics['summary']['rating_correlation']

    if mae < 1.0 and verdict_acc > 0.7 and correlation > 0.6:
        grade = "* EXCELLENT"
    elif mae < 1.5 and verdict_acc > 0.6 and correlation > 0.4:
        grade = "* GOOD"
    elif mae < 2.0 and verdict_acc > 0.5 and correlation > 0.3:
        grade = "* ACCEPTABLE"
    else:
        grade = "* NEEDS IMPROVEMENT"

    print(f"  Overall Grade: {grade}")
    print(f"  - Rating Error: {'*' if mae < 1.5 else '*'} {mae:.3f} MAE")
    print(f"  - Winner Accuracy: {'*' if verdict_acc > 0.6 else '*'} {verdict_acc:.1%}")
    print(f"  - Rating Correlation: {'*' if correlation > 0.4 else '*'} {correlation:.3f}")

    print("\n* Evaluation complete!")


if __name__ == "__main__":
    main()
