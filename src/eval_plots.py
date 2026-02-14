"""
Visualization Module — Evaluation Plots.

Generates evaluation-related plots: confusion matrix heatmap
and sample images from each class.
All plots are saved at 300 DPI to results/graphs/.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

GRAPH_DIR_DEFAULT = "results/graphs"
DPI = 300
LABEL_NAMES = {0: "Even Only", 1: "Has Odd"}


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    output_dir: str | Path = GRAPH_DIR_DEFAULT,
) -> None:
    """Heatmap of the 2x2 confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Even Only", "Has Odd"])
    ax.set_yticklabels(["Even Only", "Has Odd"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    fig.colorbar(im)

    path = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_sample_images(
    test_loader: DataLoader,
    output_dir: str | Path = GRAPH_DIR_DEFAULT,
) -> None:
    """Show 5 example images from each class."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_imgs, all_labels = [], []
    for imgs, lbls in test_loader:
        all_imgs.append(imgs)
        all_labels.append(lbls)
    all_imgs = torch.cat(all_imgs)
    all_labels = torch.cat(all_labels).numpy().astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Sample Images by Class", fontsize=14, fontweight="bold")

    for cls, row_axes in enumerate(axes):
        indices = np.where(all_labels == cls)[0][:5]
        for col, idx in enumerate(indices):
            row_axes[col].imshow(all_imgs[idx].squeeze().numpy(), cmap="gray")
            row_axes[col].set_title(LABEL_NAMES[cls], fontsize=9)
            row_axes[col].axis("off")

    path = output_dir / "sample_images.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", path)
