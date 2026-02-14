#!/usr/bin/env python3
"""
L38 — CNN Odd Number Detector
Main Entry Point

Runs the complete pipeline:
  1. Load & preprocess dataset from input/
  2. Build the CNN model
  3. Train for N epochs
  4. Evaluate on test set
  5. Generate all visualizations
  6. Save results

Usage:
    python main.py
    python main.py --epochs 20 --batch-size 64
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset
from src.model import TransferCNN
from src.trainer import train_model
from src.evaluator import evaluate_model, save_results
from src.visualizer import (
    plot_loss_curve,
    plot_accuracy_curve,
    plot_prediction_grid,
)
from src.eval_plots import plot_confusion_matrix, plot_sample_images

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
MODEL_PATH = RESULTS_DIR / "model.pth"
LOG_PATH = RESULTS_DIR / "training_log.json"


def setup_logging() -> None:
    """Configure console + file logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """Orchestrate the full CNN training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="L38 CNN Odd Number Detector")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  L38 — CNN Odd Number Detector")
    print(f"  Device: {device}")
    print("=" * 60)

    # Step 1 — Load dataset
    print("\n[1/5] Loading dataset...")
    train_loader, test_loader, summary = load_dataset(
        INPUT_DIR, batch_size=args.batch_size)
    print(f"  Total: {summary['total_images']} images "
          f"(odd={summary['odd_count']}, even={summary['even_count']})")
    print(f"  Train: {summary['train_count']}  |  Test: {summary['test_count']}")

    # Step 2 — Build model
    print("\n[2/5] Building CNN model...")
    model = TransferCNN()
    model.print_summary()
    print(f"  Parameters: {model.count_parameters():,}")

    # Step 3 — Train
    print(f"\n[3/5] Training for {args.epochs} epochs...")
    model, history = train_model(
        model, train_loader, test_loader,
        num_epochs=args.epochs, learning_rate=args.lr, device=device)

    # Step 4 — Evaluate
    print("\n[4/5] Evaluating on test set...")
    metrics, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    # Save model + results JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    save_results(metrics, history, LOG_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # Step 5 — Visualizations
    print("\n[5/5] Generating visualizations...")
    plot_loss_curve(history["train_loss"], GRAPHS_DIR)
    plot_accuracy_curve(history["test_accuracy"], GRAPHS_DIR)
    plot_prediction_grid(model, test_loader, device, GRAPHS_DIR)
    plot_confusion_matrix(y_true, y_pred, GRAPHS_DIR)
    plot_sample_images(test_loader, GRAPHS_DIR)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Final accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Results:  {RESULTS_DIR}")
    print(f"  Graphs:   {GRAPHS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
