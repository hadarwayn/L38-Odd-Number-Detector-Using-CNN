"""
Data Loading Module for CNN Odd Number Detector.

Pipeline: scan flat input/ -> assign labels from filenames ->
load & preprocess images -> stratified train/test split ->
wrap in PyTorch DataLoaders for batch training.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

IMAGE_SIZE = 64
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _scan_images(input_dir: Path) -> Tuple[list, list]:
    """Scan flat input/ directory and assign labels from filename prefixes."""
    paths, labels = [], []
    for filepath in sorted(input_dir.iterdir()):
        if filepath.suffix.lower() not in VALID_EXTENSIONS:
            continue
        name = filepath.name.lower()
        if name.startswith("include-odd-numbers"):
            labels.append(1)
            paths.append(filepath)
        elif name.startswith("include-even-numbers"):
            labels.append(0)
            paths.append(filepath)
        else:
            logger.warning("Skipping unknown file: %s", filepath.name)
    return paths, labels


def _load_and_preprocess(paths: list) -> np.ndarray:
    """Load images, convert to grayscale, resize to 64x64, normalize."""
    images = []
    for path in paths:
        try:
            img = Image.open(path).convert("L")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            pixel_array = np.array(img, dtype=np.float32) / 255.0
            images.append(pixel_array)
        except Exception as exc:
            logger.warning("Skipping corrupt image %s: %s", path.name, exc)
    return np.expand_dims(np.array(images), axis=1)


def load_dataset(
    input_dir: str | Path,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, dict]:
    """Complete data loading pipeline: scan -> load -> split -> DataLoaders."""
    input_dir = Path(input_dir)
    logger.info("Scanning images in %s", input_dir)

    paths, labels = _scan_images(input_dir)
    logger.info("Found %d images (%d odd, %d even)",
                len(paths), sum(labels), len(labels) - sum(labels))

    images = _load_and_preprocess(paths)
    labels_array = np.array(labels, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_array,
        test_size=test_size, random_state=random_seed,
        stratify=labels_array,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    summary = {
        "total_images": len(paths),
        "odd_count": int(sum(labels)),
        "even_count": int(len(labels) - sum(labels)),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "image_shape": f"(1, {IMAGE_SIZE}, {IMAGE_SIZE})",
        "batch_size": batch_size,
    }
    logger.info("Dataset ready — train: %d, test: %d",
                summary["train_count"], summary["test_count"])
    return train_loader, test_loader, summary
