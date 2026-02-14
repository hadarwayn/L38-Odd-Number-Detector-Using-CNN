"""
Visualization Module — Training & Prediction Plots.

Generates training-related plots: loss curve, accuracy curve,
and a prediction grid showing model output on test images.
All plots are saved at 300 DPI to results/graphs/.
"""

import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

GRAPH_DIR_DEFAULT = "results/graphs"
DPI = 300
LABEL_NAMES = {0: "Even Only", 1: "Has Odd"}


def plot_loss_curve(
    losses: List[float], output_dir: str | Path = GRAPH_DIR_DEFAULT,
) -> None:
    """Plot training loss vs. epoch — shows the network learning."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, "b-o", linewidth=2, markersize=5)
    ax.set_title("Training Loss Over Epochs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(True, alpha=0.3, linestyle="--")

    path = output_dir / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_accuracy_curve(
    accuracies: List[float], output_dir: str | Path = GRAPH_DIR_DEFAULT,
) -> None:
    """Plot test accuracy vs. epoch — shows prediction quality improving."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(accuracies) + 1)
    pct = [a * 100 for a in accuracies]
    ax.plot(epochs, pct, "g-o", linewidth=2, markersize=5)
    ax.set_title("Test Accuracy Over Epochs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle="--")

    path = output_dir / "accuracy_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_prediction_grid(
    model: nn.Module, test_loader: DataLoader,
    device: torch.device, output_dir: str | Path = GRAPH_DIR_DEFAULT,
) -> None:
    """4x4 grid of test images with predicted vs actual labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    images, labels = next(iter(test_loader))
    images_dev = images[:16].to(device)
    with torch.no_grad():
        logits = model(images_dev).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels[:16].numpy().astype(int)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Prediction Grid (Green=Correct, Red=Wrong)",
                 fontsize=14, fontweight="bold")
    for idx, ax in enumerate(axes.flat):
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        actual = LABEL_NAMES[labels_np[idx]]
        predicted = LABEL_NAMES[preds[idx]]
        correct = preds[idx] == labels_np[idx]
        color = "green" if correct else "red"
        ax.set_title(f"P:{predicted}\nA:{actual}",
                     fontsize=8, color=color)
        ax.axis("off")

    path = output_dir / "prediction_grid.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", path)
