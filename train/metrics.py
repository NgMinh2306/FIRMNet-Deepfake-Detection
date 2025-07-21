# train/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_curve
)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import get_logger

logger = get_logger(__name__)


def calculate_eer(y_true, y_score):
    """
    Calculate Equal Error Rate (EER) for binary classification.

    Args:
        y_true (array-like): Ground-truth binary labels (0 or 1).
        y_score (array-like): Predicted probabilities or scores for class 1.

    Returns:
        eer (float): Equal Error Rate.
        threshold (float): Threshold at which EER occurs.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    threshold = thresholds[idx]
    return eer, threshold


def compute_metrics(y_true, y_prob, extra_metrics=None):
    """
    Compute common classification metrics for binary classification.

    Args:
        y_true (array-like): Ground-truth binary labels.
        y_prob (array-like): Predicted class probabilities (shape: [N, 2]).
        extra_metrics (list[str]): Optional metrics to compute, e.g. ['F1 score', 'EER'].

    Returns:
        metrics_dict (dict): Dictionary with keys: accuracy, auc, + optional metrics.
    """
    if isinstance(extra_metrics, str):
        extra_metrics = [extra_metrics]
    elif extra_metrics is None:
        extra_metrics = []

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob[:, 1])
    }

    if 'AP' in extra_metrics:
        metrics['AP'] = average_precision_score(y_true, y_prob[:, 1])
    if 'EER' in extra_metrics:
        metrics['EER'] = calculate_eer(y_true, y_prob[:, 1])[0]
    if 'F1 score' in extra_metrics:
        metrics['F1 score'] = f1_score(y_true, y_pred, average='macro')
    if 'precision' in extra_metrics:
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    if 'recall' in extra_metrics:
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')

    logger.info(f"Computed metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test classification metrics")
    parser.add_argument("--add", type=str, nargs='*', default=['AP', 'F1 score', 'precision', 'recall', 'EER'],
                        help="Extra metrics to compute")
    args = parser.parse_args()

    logger.info("Running CLI test for metrics.py")

    # Dummy test data
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_prob = np.array([
        [0.9, 0.1],
        [0.3, 0.7],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.4, 0.6]
    ])

    results = compute_metrics(y_true, y_prob, extra_metrics=args.add)
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
