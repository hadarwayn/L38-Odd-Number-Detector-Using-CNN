# PRD — L38: Synthetic Odd Number Detector CNN

**Version:** 2.0  
**Date:** 2026-02-14  
**Project Code:** L38  
**Course:** AI Developer Expert — Dr. Yoram Segal  
**Lesson Reference:** Lesson 38 — Convolutional Neural Networks (CNNs)

---

## 1. Executive Summary

This project builds a complete **Convolutional Neural Network (CNN) image classification pipeline** that demonstrates every concept taught in Lesson 38 — from raw pixel input to final prediction output.

**The Task:** Given a grayscale image containing printed numbers, the CNN must decide:
- **Class 1 (Positive):** The image contains at least one **positive odd digit** (1, 3, 5, 7, 9)
- **Class 0 (Negative):** The image contains **only even digits** (0, 2, 4, 6, 8)

**Why This Project?** Instead of using abstract academic datasets, we make CNN concepts tangible. Students can literally *see* what the network is learning — the shapes of odd digits vs. even digits. Every CNN building block from Lesson 38 (convolution kernels, feature maps, ReLU activation, max pooling, fully connected layers, and loss optimization) is implemented, trained, and visualized.

---

## 2. Objectives

| # | Objective | Success Metric |
|---|-----------|---------------|
| 1 | Build a complete CNN classification pipeline from data loading to prediction | All pipeline stages execute without errors |
| 2 | Train the CNN to distinguish odd-containing vs. even-only images | Accuracy > 90% on test set |
| 3 | Visualize training progress and model predictions | Loss curve + prediction grid generated |
| 4 | Demonstrate every CNN concept from Lesson 38 | README explains each layer with analogies |
| 5 | Follow AI Developer Expert project standards | ≤150 lines/file, modular code, type hints |
| 6 | Create an educational portfolio piece | Passes the "15-year-old test" for clarity |

---

## 3. Target Users & Applications

### 3.1 Primary Users

**AI Developer Expert students** (layman level) who have just completed Lesson 38 on CNNs and want to see theory in action.

### 3.2 Use Cases

1. **Learning Reinforcement:** A student runs the project to watch a CNN train in real-time, seeing the loss decrease epoch by epoch — making the abstract "gradient descent" concept concrete.

2. **Concept Visualization:** A student examines the prediction grid to understand *why* the CNN classified certain images correctly or incorrectly — building intuition about what features the network learned.

3. **Parameter Experimentation:** A student modifies the learning rate, number of epochs, or batch size to observe how these hyperparameters affect training speed and final accuracy — developing practical tuning skills.

4. **Portfolio Demonstration:** A student shows this project to a potential employer or in a course presentation as evidence of understanding deep learning fundamentals.

---

## 4. Dataset Specification

### 4.1 Dataset Source & Composition

The dataset was curated from **three sources** to ensure diversity and robustness:

| Source | Description | Contribution |
|--------|-------------|-------------|
| **Gemini (AI-Generated)** | Synthetic images created by Gemini nanobanana | ~1/3 of dataset |
| **MNIST** | Classic handwritten digit dataset (educational, free) | ~1/3 of dataset |
| **SVHN** | Street View House Numbers dataset (educational, free) | ~1/3 of dataset |

### 4.2 Dataset Size & Balance

| Class | Label | Filename Prefix | Count |
|-------|-------|----------------|-------|
| **Positive (Has Odd)** | 1 | `include-odd-numbers-*` | 1,207 images |
| **Negative (Even Only)** | 0 | `include-even-numbers-*` | 1,200 images |
| **Total** | — | — | **2,407 images** |

### 4.3 Image Specifications

- **Format:** PNG and JPEG (grayscale)
- **Resolution:** Variable (resized to 64×64 during preprocessing)
- **Color Space:** Converted to grayscale (single channel)
- **Pixel Values:** 0–255 (raw), normalized to 0.0–1.0 for training
- **Storage Location:** All images in a **flat** `input/` directory (no subdirectories)

### 4.4 Labeling Strategy

**The CNN does not know in advance which images are odd or even.** Labels are derived from filename prefixes *only during data loading* for supervised training:
- Filename starts with `include-odd-numbers` → label **1** (has odd digit)
- Filename starts with `include-even-numbers` → label **0** (even only)

The model itself learns to classify purely from pixel data. At evaluation time, predictions are compared against ground-truth labels to measure accuracy.

### 4.5 Train/Test Split

| Split | Percentage | Approximate Count |
|-------|-----------|-------------------|
| Training | 80% | ~1,926 images |
| Testing | 20% | ~481 images |

The split is performed with stratification to maintain class balance in both sets. A fixed random seed (42) ensures reproducibility.

---

## 5. Functional Requirements

### 5.1 Data Loading Pipeline

| Requirement | Description |
|-------------|-------------|
| **FR-1.1** | Load all images from flat `input/` directory |
| **FR-1.2** | Assign labels based on filename prefix: `include-odd-numbers` → 1, `include-even-numbers` → 0 |
| **FR-1.3** | Resize images to 64×64 if needed (handle inconsistent sizes gracefully) |
| **FR-1.4** | Normalize pixel values from [0, 255] to [0.0, 1.0] |
| **FR-1.5** | Convert to PyTorch tensors with shape (N, 1, 64, 64) |
| **FR-1.6** | Create stratified train/test split (80/20) |
| **FR-1.7** | Wrap data in PyTorch DataLoader with configurable batch size |
| **FR-1.8** | Print dataset summary: total images, class distribution, split sizes |

### 5.2 CNN Architecture

The architecture directly implements the Lesson 38 pipeline:

```
Input Image (1 × 64 × 64)
    │
    ▼
┌─────────────────────────────┐
│  Conv2D Layer 1             │  ← "The Flashlight" scanning for edges
│  (1 → 16 filters, 3×3)     │     and simple patterns
│  + ReLU Activation          │  ← "The Bouncer" removing negatives
│  + MaxPool2D (2×2)          │  ← "The Summarizer" shrinking the map
└─────────────────────────────┘
    │  Output: 16 × 32 × 32
    ▼
┌─────────────────────────────┐
│  Conv2D Layer 2             │  ← Scanning for more complex shapes
│  (16 → 32 filters, 3×3)    │     (curves, corners of digits)
│  + ReLU Activation          │
│  + MaxPool2D (2×2)          │
└─────────────────────────────┘
    │  Output: 32 × 16 × 16
    ▼
┌─────────────────────────────┐
│  Flatten                    │  ← "The Bridge" converting 2D → 1D
│  (32 × 16 × 16 = 8,192)    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Fully Connected Layer 1    │  ← "The Brain" making decisions
│  (8192 → 64 neurons)       │
│  + ReLU Activation          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Fully Connected Layer 2    │  ← Final decision: odd or not?
│  (64 → 1 neuron)           │
│  + Sigmoid Activation       │  ← Outputs probability 0.0 to 1.0
└─────────────────────────────┘
```

**Lesson 38 Concept Mapping:**

| CNN Concept (Lesson 38) | Implementation | Analogy |
|------------------------|----------------|---------|
| Convolution & Kernels | `nn.Conv2d` layers | A flashlight scanning a dark room patch-by-patch |
| Feature Maps | Output of Conv layers | A "highlight map" showing where patterns were found |
| ReLU Activation | `F.relu()` | A bouncer who only lets positive signals through |
| Max Pooling | `nn.MaxPool2d` | Reading a summary instead of the whole book |
| Flattening | `x.view(-1, ...)` | Unrolling a grid into a single line of numbers |
| Fully Connected | `nn.Linear` | The "brain" that votes on what the features mean |
| Sigmoid/Output | `torch.sigmoid()` | A confidence meter: 0% = "all even", 100% = "has odd" |

### 5.3 Training Pipeline

| Requirement | Description |
|-------------|-------------|
| **FR-3.1** | Loss function: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) |
| **FR-3.2** | Optimizer: Adam (learning_rate = 0.001) |
| **FR-3.3** | Batch size: 32 |
| **FR-3.4** | Training epochs: 15 (configurable) |
| **FR-3.5** | Track training loss per epoch |
| **FR-3.6** | Track validation accuracy per epoch |
| **FR-3.7** | Print progress: epoch number, loss, accuracy after each epoch |
| **FR-3.8** | Save trained model to `results/model.pth` |

### 5.4 Evaluation & Metrics

| Requirement | Description |
|-------------|-------------|
| **FR-4.1** | Calculate final test accuracy |
| **FR-4.2** | Generate confusion matrix (TP, TN, FP, FN) |
| **FR-4.3** | Calculate precision, recall, and F1-score |
| **FR-4.4** | Report total training time |

### 5.5 Visualization Outputs

| Requirement | Description | Output File |
|-------------|-------------|-------------|
| **FR-5.1** | Training loss curve (loss vs. epoch) | `results/graphs/loss_curve.png` |
| **FR-5.2** | Accuracy curve (accuracy vs. epoch) | `results/graphs/accuracy_curve.png` |
| **FR-5.3** | Prediction grid: 4×4 grid of test images with predicted vs. actual labels | `results/graphs/prediction_grid.png` |
| **FR-5.4** | Confusion matrix heatmap | `results/graphs/confusion_matrix.png` |
| **FR-5.5** | Sample images display (examples from each class) | `results/graphs/sample_images.png` |

---

## 6. Technical Requirements

### 6.1 Environment

| Requirement | Specification |
|-------------|--------------|
| **Python** | 3.10+ |
| **Virtual Environment** | UV (mandatory) |
| **Operating Systems** | Windows (WSL), Linux, macOS |
| **GPU** | Optional (CPU training is sufficient for this dataset size) |

### 6.2 Core Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| `torch` | CNN model, training loop, tensors | ≥2.0 |
| `torchvision` | Image transforms and dataset utilities | ≥0.15 |
| `numpy` | Array operations, data manipulation | ≥1.24 |
| `matplotlib` | All visualizations and graphs | ≥3.7 |
| `Pillow` | Image loading and preprocessing | ≥10.0 |
| `scikit-learn` | Train/test split, confusion matrix, metrics | ≥1.3 |

### 6.3 Code Standards

| Standard | Requirement |
|----------|------------|
| **Max file length** | ≤150 lines per .py file |
| **Type hints** | All functions must have type annotations |
| **Docstrings** | Every function must explain WHAT, WHY, and HOW |
| **Comments** | Pass the "15-year-old test" — plain English explanations |
| **Vectorization** | Use NumPy/PyTorch operations, not raw Python loops for data |
| **Imports** | Relative imports within `src/` package |
| **Paths** | All paths relative using `pathlib.Path` |
| **Error handling** | Graceful errors with informative messages |

### 6.4 Performance Requirements

| Metric | Target |
|--------|--------|
| Training time (CPU) | < 5 minutes for 15 epochs |
| Model accuracy | > 90% on test set |
| Memory usage | < 2 GB RAM |
| Total images processed | All 2,407 images |

---

## 7. Project Structure

```
L38-cnn-odd-number-detector/
│
├── README.md                    # Project showcase + CNN learning guide
├── main.py                      # Single entry point — runs everything
├── requirements.txt             # All dependencies with exact versions
├── .gitignore                   # Secrets, cache, venv protection
│
├── venv/                        # Virtual environment indicator
│   └── .gitkeep                 # Setup instructions
│
├── input/                       # Dataset — flat directory, 2,407 images
│   ├── include-odd-numbers-*.png/jpg    # 1,207 images with odd digits
│   ├── include-even-numbers-*.png/jpg   # 1,200 images with even digits only
│   └── ...
│
├── src/                         # All source code
│   ├── __init__.py              # Package marker
│   ├── data_loader.py           # Load, normalize, split dataset
│   ├── model.py                 # CNN architecture definition
│   ├── trainer.py               # Training loop + validation
│   ├── evaluator.py             # Test evaluation + metrics
│   └── visualizer.py            # All plots and visualizations
│
├── docs/                        # Documentation
│   ├── PRD.md                   # This document
│   └── tasks.json               # Implementation task breakdown
│
├── results/                     # All outputs
│   ├── graphs/                  # Generated visualizations
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   ├── prediction_grid.png
│   │   ├── confusion_matrix.png
│   │   └── sample_images.png
│   ├── model.pth                # Saved trained model weights
│   └── training_log.json        # Epoch-by-epoch metrics
│
└── logs/                        # Ring buffer logging
    ├── config/
    │   └── log_config.json
    └── .gitkeep
```

---

## 8. Success Criteria

### 8.1 Functional Success

- [ ] All 2,407 images loaded successfully from `input/` directories
- [ ] CNN trains without errors for configured number of epochs
- [ ] Loss decreases consistently over training epochs
- [ ] Test accuracy exceeds 90%
- [ ] All 5 visualization outputs generated and saved

### 8.2 Educational Success

- [ ] README explains every CNN layer with real-world analogies
- [ ] README maps each code component to Lesson 38 concepts
- [ ] A 15-year-old can understand the README explanations
- [ ] Results section shows visual evidence of learning

### 8.3 Code Quality Success

- [ ] All .py files ≤ 150 lines
- [ ] Every function has type hints and docstrings
- [ ] Project runs with single command: `python main.py`
- [ ] All paths are relative (works on any computer)
- [ ] Follows complete PROJECT_GUIDELINES.md standards

---

## 9. CNN Concepts Demonstrated (Lesson 38 Alignment)

This section maps every Lesson 38 concept to its implementation in this project:

| Lesson 38 Topic | Where Demonstrated | What Students See |
|------------------|--------------------|-------------------|
| Images as pixel grids | `data_loader.py` normalization | Raw 0-255 values → 0.0-1.0 floats |
| Convolution operation | `model.py` Conv2d layers | Kernel "flashlight" scanning the image |
| Feature maps | Model architecture | 16 → 32 feature maps extracted |
| ReLU activation | `model.py` forward pass | Negative values silenced to zero |
| Max pooling (downsampling) | `model.py` MaxPool2d | Image dimensions halved: 64→32→16 |
| Flattening | `model.py` view operation | 2D grid → 1D vector for classification |
| Fully connected layers | `model.py` Linear layers | "Brain" making the final decision |
| Loss function (BCE) | `trainer.py` | Error measurement driving learning |
| Backpropagation | `trainer.py` loss.backward() | Weights updated to reduce error |
| Epochs & training loop | `trainer.py` | Repeated learning cycles |
| Overfitting awareness | Train vs. test accuracy comparison | Why we split data |

---

## 10. Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Accuracy below 90% | Medium | Low | Increase epochs; adjust learning rate; add dropout |
| Overfitting (high train, low test accuracy) | Medium | Medium | Monitor train vs. test gap; add dropout layer if needed |
| Images not loading (format issues) | High | Low | Robust error handling with skip-and-log for corrupt files |
| Slow training on CPU | Low | Medium | Batch size tuning; 15 epochs sufficient for this dataset |
| Class imbalance (1207 vs 1200) | Low | Low | Nearly balanced; stratified split handles minor difference |

---

## 11. Future Enhancements (Out of Scope for v1.0)

- **Feature Map Visualization:** Show what each Conv layer "sees" for a given input
- **Saliency Maps / Grad-CAM:** Highlight which image regions influenced the prediction
- **Confusion Matrix Deep-Dive:** Display the actual misclassified images
- **Data Augmentation:** Add rotation, flip, noise to improve generalization
- **Multi-class Extension:** Classify which specific digit is present (0-9)
- **Model Comparison:** Compare CNN accuracy vs. a simple fully-connected network

---

## 12. Learning Objectives

After completing this project, the student will be able to:

1. **Explain** what a CNN does and why it's better than a fully-connected network for images
2. **Identify** the role of each CNN layer: convolution, ReLU, pooling, flatten, fully connected
3. **Interpret** a training loss curve and understand what "convergence" means
4. **Run** a complete deep learning pipeline from data loading to prediction
5. **Modify** hyperparameters (learning rate, epochs, batch size) and predict their effects
6. **Read** PyTorch code and map it to the theoretical CNN architecture from Lesson 38

---

*End of PRD — Version 2.0*
