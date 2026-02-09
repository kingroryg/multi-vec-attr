"""
Identification Metrics

Metrics for evaluating watermark detection accuracy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from collections import defaultdict


def compute_accuracy(
    predictions: List[Optional[int]],
    ground_truth: List[int],
) -> Dict[str, float]:
    """
    Compute identification accuracy metrics.

    Args:
        predictions: Predicted vendor IDs (None for undetected)
        ground_truth: True vendor IDs

    Returns:
        Dict with accuracy metrics
    """
    n = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    detected = sum(1 for p in predictions if p is not None)

    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "num_samples": n,
        "num_correct": correct,
        "detection_rate": detected / n if n > 0 else 0.0,
        "num_detected": detected,
    }


def compute_topk_accuracy(
    correlations: np.ndarray,
    ground_truth: List[int],
    k: int = 3,
) -> float:
    """
    Compute top-k identification accuracy.

    Args:
        correlations: Array of shape (num_samples, num_vendors)
        ground_truth: True vendor IDs

    Returns:
        Top-k accuracy
    """
    correct = 0
    for corrs, gt in zip(correlations, ground_truth):
        topk = np.argsort(corrs)[-k:]
        if gt in topk:
            correct += 1
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0.0


def compute_confusion_matrix(
    predictions: List[int],
    ground_truth: List[int],
    num_vendors: int,
) -> np.ndarray:
    """
    Compute confusion matrix for vendor identification.

    Args:
        predictions: Predicted vendor IDs
        ground_truth: True vendor IDs
        num_vendors: Total number of vendors

    Returns:
        Confusion matrix of shape (num_vendors, num_vendors)
    """
    labels = list(range(num_vendors))
    return confusion_matrix(ground_truth, predictions, labels=labels)


def compute_per_vendor_metrics(
    predictions: List[int],
    ground_truth: List[int],
    num_vendors: int,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-vendor precision, recall, and F1.

    Args:
        predictions: Predicted vendor IDs
        ground_truth: True vendor IDs
        num_vendors: Total number of vendors

    Returns:
        Dict mapping vendor ID to metrics
    """
    labels = list(range(num_vendors))
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, zero_division=0
    )

    per_vendor = {}
    for vid in range(num_vendors):
        per_vendor[vid] = {
            "precision": float(precision[vid]),
            "recall": float(recall[vid]),
            "f1": float(f1[vid]),
            "support": int(support[vid]),
        }

    return per_vendor


def compute_roc_curve(
    correlations: np.ndarray,
    ground_truth: List[int],
    target_vendor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve for detecting a specific vendor.

    Args:
        correlations: Array of shape (num_samples, num_vendors)
        ground_truth: True vendor IDs
        target_vendor: Vendor ID to compute ROC for

    Returns:
        fpr, tpr, thresholds
    """
    # Binary labels: 1 if sample is from target vendor, 0 otherwise
    y_true = [1 if g == target_vendor else 0 for g in ground_truth]

    # Scores: correlation with target vendor
    y_score = correlations[:, target_vendor]

    return roc_curve(y_true, y_score)


def compute_auc(
    correlations: np.ndarray,
    ground_truth: List[int],
    target_vendor: Optional[int] = None,
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        correlations: Array of shape (num_samples, num_vendors)
        ground_truth: True vendor IDs
        target_vendor: If provided, compute AUC for this vendor only.
                       Otherwise, compute macro-average AUC.

    Returns:
        AUC score
    """
    if target_vendor is not None:
        fpr, tpr, _ = compute_roc_curve(correlations, ground_truth, target_vendor)
        return auc(fpr, tpr)

    # Macro-average AUC
    vendors = sorted(set(ground_truth))
    aucs = []
    for v in vendors:
        try:
            fpr, tpr, _ = compute_roc_curve(correlations, ground_truth, v)
            aucs.append(auc(fpr, tpr))
        except ValueError:
            # Skip if not enough samples for this vendor
            continue
    return np.mean(aucs) if aucs else 0.0


def compute_detection_threshold(
    correlations: np.ndarray,
    ground_truth: List[int],
    target_fpr: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """
    Find detection threshold for target false positive rate.

    Args:
        correlations: Array of shape (num_samples, num_vendors)
        ground_truth: True vendor IDs
        target_fpr: Target false positive rate

    Returns:
        threshold, metrics_at_threshold
    """
    # Collect max correlations and whether prediction was correct
    max_corrs = np.max(correlations, axis=1)
    predictions = np.argmax(correlations, axis=1)
    correct = [p == g for p, g in zip(predictions, ground_truth)]

    # Sort by correlation (descending)
    sorted_indices = np.argsort(max_corrs)[::-1]
    sorted_corrs = max_corrs[sorted_indices]
    sorted_correct = [correct[i] for i in sorted_indices]

    # Find threshold at target FPR
    n = len(sorted_corrs)
    target_fp = int(n * target_fpr)

    # Count false positives at each threshold
    cumsum_incorrect = np.cumsum([not c for c in sorted_correct])

    # Find threshold where FP count equals target
    for i, (corr, fp_count) in enumerate(zip(sorted_corrs, cumsum_incorrect)):
        if fp_count >= target_fp:
            threshold = corr
            break
    else:
        threshold = sorted_corrs[-1]

    # Compute metrics at this threshold
    detected = max_corrs >= threshold
    tp = sum(1 for d, c in zip(detected, correct) if d and c)
    fp = sum(1 for d, c in zip(detected, correct) if d and not c)
    fn = sum(1 for d, c in zip(detected, correct) if not d and c)

    metrics = {
        "threshold": float(threshold),
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "false_positive_rate": fp / n if n > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
    }

    return threshold, metrics
