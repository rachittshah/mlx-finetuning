#!/usr/bin/env python3
"""
Detailed evaluation of the Walmart-Amazon embedding model with proper metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "walmart_amazon_real"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "embedding_lora" / "walmart-amazon-real"

def create_product_text(row):
    """Create a rich text representation of a product"""
    parts = []
    if pd.notna(row.get('title')):
        parts.append(row['title'])
    if pd.notna(row.get('brand')):
        parts.append(f"Brand: {row['brand']}")
    if pd.notna(row.get('category')):
        parts.append(f"Category: {row['category']}")
    if pd.notna(row.get('modelno')):
        parts.append(f"Model: {row['modelno']}")
    if pd.notna(row.get('price')):
        parts.append(f"${row['price']:.2f}")
    return " | ".join(parts)

def load_test_data():
    """Load test data"""
    tableA = pd.read_csv(DATA_DIR / "tableA.csv")
    tableB = pd.read_csv(DATA_DIR / "tableB.csv")
    test_pairs = pd.read_csv(DATA_DIR / "test.csv")

    # Create text representations
    tableA['text'] = tableA.apply(create_product_text, axis=1)
    tableB['text'] = tableB.apply(create_product_text, axis=1)

    walmart_products = tableA.set_index('id')['text'].to_dict()
    amazon_products = tableB.set_index('id')['text'].to_dict()

    sentences1 = []
    sentences2 = []
    labels = []

    for _, row in test_pairs.iterrows():
        walmart_text = walmart_products.get(row['ltable_id'])
        amazon_text = amazon_products.get(row['rtable_id'])

        if walmart_text and amazon_text:
            sentences1.append(walmart_text)
            sentences2.append(amazon_text)
            labels.append(row['label'])

    return sentences1, sentences2, labels

def main():
    print("=" * 80)
    print("DETAILED EVALUATION - WALMART-AMAZON EMBEDDING MODEL")
    print("=" * 80)

    # Load model
    print(f"\n* Loading model from {MODEL_PATH}...")
    model = SentenceTransformer(str(MODEL_PATH))

    # Load test data
    print(f"\n* Loading test data...")
    sentences1, sentences2, labels = load_test_data()
    labels = np.array(labels)

    print(f"✓ Loaded {len(labels)} test pairs")
    print(f"  - Positive (matches): {labels.sum()}")
    print(f"  - Negative (non-matches): {len(labels) - labels.sum()}")
    print(f"  - Class balance: {labels.sum() / len(labels):.1%} positive")

    # Compute embeddings
    print(f"\n🔮 Computing embeddings...")
    emb1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
    emb2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True, batch_size=32)

    print(f"\n* Embedding statistics:")
    print(f"  - Embedding 1 shape: {emb1.shape}")
    print(f"  - Embedding 2 shape: {emb2.shape}")
    print(f"  - Embedding 1 mean: {emb1.mean().item():.6f}")
    print(f"  - Embedding 1 std: {emb1.std().item():.6f}")
    print(f"  - Embedding 2 mean: {emb2.mean().item():.6f}")
    print(f"  - Embedding 2 std: {emb2.std().item():.6f}")
    print(f"  - Contains NaN (emb1): {torch.isnan(emb1).any().item()}")
    print(f"  - Contains NaN (emb2): {torch.isnan(emb2).any().item()}")
    print(f"  - Contains Inf (emb1): {torch.isinf(emb1).any().item()}")
    print(f"  - Contains Inf (emb2): {torch.isinf(emb2).any().item()}")

    # Compute similarities
    print(f"\n🔢 Computing cosine similarities...")
    similarities = torch.nn.functional.cosine_similarity(emb1, emb2).cpu().numpy()

    print(f"\n* Similarity statistics:")
    print(f"  - Min: {np.nanmin(similarities):.6f}")
    print(f"  - Max: {np.nanmax(similarities):.6f}")
    print(f"  - Mean: {np.nanmean(similarities):.6f}")
    print(f"  - Std: {np.nanstd(similarities):.6f}")
    print(f"  - Contains NaN: {np.isnan(similarities).sum()} / {len(similarities)}")
    print(f"  - Positive samples mean: {np.nanmean(similarities[labels == 1]):.6f}")
    print(f"  - Negative samples mean: {np.nanmean(similarities[labels == 0]):.6f}")

    # If all NaN, try manual computation
    if np.isnan(similarities).all():
        print(f"\n⚠*  All similarities are NaN! Trying manual computation...")

        # Try computing a few manually
        for i in range(min(5, len(emb1))):
            e1 = emb1[i].cpu().numpy()
            e2 = emb2[i].cpu().numpy()

            # Check for NaN in individual embeddings
            print(f"\n  Sample {i}:")
            print(f"    - Emb1 NaN: {np.isnan(e1).any()}, norm: {np.linalg.norm(e1):.6f}")
            print(f"    - Emb2 NaN: {np.isnan(e2).any()}, norm: {np.linalg.norm(e2):.6f}")

            # Manual cosine similarity
            dot = np.dot(e1, e2)
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            manual_sim = dot / (norm1 * norm2 + 1e-8)
            print(f"    - Manual cosine: {manual_sim:.6f}")
            print(f"    - PyTorch cosine: {similarities[i]}")

    # Handle NaN by replacing with -1 (worst similarity)
    similarities_clean = np.nan_to_num(similarities, nan=-1.0)

    # Find optimal threshold using different methods
    print(f"\n* Finding optimal threshold...")

    best_threshold = 0.5
    best_f1 = 0
    threshold_results = []

    for threshold in np.arange(0.0, 1.0, 0.05):
        predictions = (similarities_clean > threshold).astype(int)

        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        threshold_results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n* Threshold analysis (top 10 by F1):")
    threshold_df = pd.DataFrame(threshold_results).sort_values('f1', ascending=False).head(10)
    print(threshold_df.to_string(index=False))

    # Evaluate at best threshold
    print(f"\n* Best threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")

    predictions = (similarities_clean > best_threshold).astype(int)

    print(f"\n* Classification Metrics:")
    print(f"  - Accuracy:  {accuracy_score(labels, predictions):.3f}")
    print(f"  - Precision: {precision_score(labels, predictions, zero_division=0):.3f}")
    print(f"  - Recall:    {recall_score(labels, predictions, zero_division=0):.3f}")
    print(f"  - F1 Score:  {f1_score(labels, predictions, zero_division=0):.3f}")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              No Match  Match")
    print(f"  Actual No: {cm[0][0]:8d}  {cm[0][1]:5d}")
    print(f"  Actual Yes: {cm[1][0]:7d}  {cm[1][1]:5d}")

    # Per-class metrics
    print(f"\n* Classification Report:")
    print(classification_report(labels, predictions,
                                target_names=['No Match', 'Match'],
                                zero_division=0))

    # ROC AUC if similarities are valid
    if not np.isnan(similarities).all():
        try:
            roc_auc = roc_auc_score(labels, similarities_clean)
            avg_prec = average_precision_score(labels, similarities_clean)
            print(f"\n* Ranking Metrics:")
            print(f"  - ROC AUC: {roc_auc:.3f}")
            print(f"  - Average Precision: {avg_prec:.3f}")
        except:
            print(f"\n⚠*  Could not compute ROC AUC (insufficient data)")

    # Save results
    results = {
        'test_samples': len(labels),
        'positive_samples': int(labels.sum()),
        'negative_samples': int(len(labels) - labels.sum()),
        'best_threshold': float(best_threshold),
        'metrics': {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
        },
        'confusion_matrix': cm.tolist(),
        'similarity_stats': {
            'min': float(np.nanmin(similarities)),
            'max': float(np.nanmax(similarities)),
            'mean': float(np.nanmean(similarities)),
            'std': float(np.nanstd(similarities)),
            'nan_count': int(np.isnan(similarities).sum())
        }
    }

    results_file = MODEL_PATH / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n* Results saved to: {results_file}")

    # Show some example predictions
    print(f"\n* Sample Predictions (first 10):")
    for i in range(min(10, len(labels))):
        actual = "MATCH" if labels[i] == 1 else "NO MATCH"
        pred = "MATCH" if predictions[i] == 1 else "NO MATCH"
        sim = similarities[i]
        correct = "*" if predictions[i] == labels[i] else "*"

        print(f"\n  {correct} Sim: {sim:.3f} | Actual: {actual:8s} | Predicted: {pred:8s}")
        print(f"    Walmart: {sentences1[i][:70]}...")
        print(f"    Amazon:  {sentences2[i][:70]}...")

if __name__ == "__main__":
    main()
