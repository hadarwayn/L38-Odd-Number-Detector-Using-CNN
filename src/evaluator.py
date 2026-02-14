"""
Evaluation Module for CNN Odd Number Detector.

This module runs the trained model on the full test set and computes
classification metrics:
- Accuracy: overall percentage of correct predictions
- Precision: of all images predicted "odd", how many actually were?
- Recall: of all actually-odd images, how many did we catch?
- F1-Score: harmonic mean of precision and recall (single quality number)
- Confusion Matrix: 2×2 grid showing TP, TN, FP, FN counts

WHY these metrics?
Accuracy alone can be misleading if classes are imbalanced. Precision
and recall tell us about different kinds of mistakes. F1 balances both.
The confusion matrix shows the full picture at a glance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

CLASSIFICATION_THRESHOLD = 0.5


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate trained model on the test set and compute all metrics.

    Args:
        model: Trained OddNumberCNN.
        test_loader: DataLoader for the test split.
        device: CPU or CUDA device.

    Returns:
        Tuple of (metrics_dict, all_labels, all_predictions).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)

            all_labels.extend(labels.numpy().astype(int))
            all_predictions.extend(preds)
            all_probabilities.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        },
        "total_test_samples": len(y_true),
    }

    logger.info("Test Accuracy:  %.2f%%", metrics["accuracy"] * 100)
    logger.info("Precision:      %.4f", metrics["precision"])
    logger.info("Recall:         %.4f", metrics["recall"])
    logger.info("F1-Score:       %.4f", metrics["f1_score"])
    logger.info("Confusion: TP=%d TN=%d FP=%d FN=%d", tp, tn, fp, fn)

    return metrics, y_true, y_pred


def save_results(
    metrics: Dict,
    history: Dict,
    output_path: str | Path,
) -> None:
    """Save training history and evaluation metrics to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy/list types to plain Python for JSON serialization
    result = {
        "training": {
            "epochs": len(history.get("train_loss", [])),
            "final_loss": history["train_loss"][-1] if history.get(
                "train_loss") else None,
            "final_accuracy": history["test_accuracy"][-1] if history.get(
                "test_accuracy") else None,
            "training_time_seconds": history.get("training_time_seconds"),
            "loss_per_epoch": history.get("train_loss", []),
            "accuracy_per_epoch": history.get("test_accuracy", []),
        },
        "evaluation": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Results saved to %s", output_path)
