# xbd-siamese-explainable-damage_assessment
Building damage assessment from satellite imagery using Siamese Networks and Explainable AI (LIME). Trained on xBD dataset for disaster response applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Data Preparation](#1-data-preparation)
  - [Training](#2-training)
  - [Inference](#3-inference)
  - [XAI Analysis](#4-xai-analysis)
- [Results](#results)

## Overview

This project implements a Deep Learning system for automatic building damage assessment from natural disasters using the **xBD** (xView2 Building Damage Dataset). The system:

1. **Classifies** buildings into 4 damage categories: `no-damage`, `minor-damage`, `major-damage`, `destroyed`
2. **Explains** model decisions through Explainable AI techniques (LIME)
3. **Supports** multiple CNN and Transformer architectures as backbones

## Key Features

### Siamese Architecture
- **Dual input**: Pre-disaster and post-disaster images (128×128 pixels)
- **Shared backbones**: Shared weights for feature extraction
- **Embedding fusion**: Concatenation of embeddings to capture changes

### Supported Backbones
**CNN (torchvision - ImageNet-1K):**
- EfficientNet-B0, EfficientNet-B3
- ResNet50
- ConvNeXt-Tiny, ConvNeXt-Small

**Transformers (timm - ImageNet-22K):**
- Swin Transformer-Tiny, Swin Transformer-Small

### Advanced Training Strategies
- **Stratified Group K-Fold Cross-Validation** (K=5)
- **Gradual Unfreezing**: Feature extraction → Fine-tuning
- **Mixed Precision Training** (AMP)
- **Data Augmentation sincronizzato** per coppie siamese
- **Multiple Loss Functions**: Cross-Entropy, Weighted CE, Focal Loss

### Explainable AI (XAI)
- **LIME** (Local Interpretable Model-agnostic Explanations)
  - Single-sample and batch analysis
  - Validation metrics: Insertion/Deletion AUC
  - Model comparison
- **Analysis by disaster type** (6 types: flooding, wind, earthquake, tsunami, volcanic eruption, wildfire)
- **Visualizations** with superpixel heatmaps

## Project Architecture

```
xbd/phyton/
│
├── 01_training/              
│   ├── data_prep/            
│   └── addestramento.py      
│
├── 02_inference/             
│   ├── data_prep_test/      
│   ├── inference.py          # Ensemble
│   ├── inference_single_fold.py  
│   └── evaluate_model.py     # Performance evaluation
│
├── 04_explainability/        
│   └── test/
│       └── LIME/
│           ├── explain_lime_test.py           # Single example analysis
│           ├── explain_lime_all_classes_test.py 
│           ├── evaluate_lime_test.py          # Single validation
│           ├── evaluate_lime_test_batch.py    
│           └── analyze_lime_results.py        # Results analysis
│
└── core/                     
    ├── config.py             # Configurations (Training/Inference/XAI)
    ├── models.py             # Model architectures
    ├── dataset.py            # Data loading and transforms
    ├── losses.py             # Loss functions
    ├── utils.py              # General utilities
    ├── inference_utils.py    
    ├── xai_shared.py         
    └── xai_metrics.py        
```

## Installation

### Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/xbd-siamese-damage-assessment.git
cd xbd-siamese-damage-assessment/xbd/phyton
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU installation (optional)
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Dataset

### xBD (xView2 Building Damage Dataset)

**Source**: [xView2 Challenge]([https://xview2.org/](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset)

**Composition**:
- **19 catastrophic events**
- **6 disaster types**: flooding, wind, earthquake, tsunami, volcanic eruption, wildfire
- **850,736 building annotations**
- **22,000+ satellite images** (1024×1024 pixels)

**Split**:
- **Train Set**: 18,336 images, 632,228 buildings
- **Test Set**: 1,866 images, 109,724 buildings
- **Hold Set**: 1,866 images, 109,784 buildings (not used)

**Damage Classes**:
| Class | Code | Description |
|--------|--------|-------------|
| No Damage | 0 | Intact building, no visible damage |
| Minor Damage | 1 | Partial damage (cracks, missing tiles, surrounding water) |
| Major Damage | 2 | Partial collapse, invasion of water/mud/lava |
| Destroyed | 3 | Burned down, completely collapsed or submerged |


## Usage

### 1. Data Preparation

#### a) Create base catalog (Train)
```bash
cd 01_training/data_prep
python 01_data_preprocessing.py
```
**Output**: `xbd_train_val_catalog.csv`

#### b) Enrich with metadata and damage counts
```bash
python 02_prep_validation_set_conteggio_danni.py
```
**Output**: `xbd_train_val_catalog_enriched.csv`

#### c) Stratify for Cross-Validation
```bash
python 03_prep_validation_set_stratificazione.py
```
**Output**: `xbd_catalog_with_folds.csv`

**Stratification Strategy**:
- **Grouped**: All patches from the same scene in the same fold (prevents data leakage)
- **Stratified**: Uniform distribution of classes and disaster types across folds

#### d) Create 128×128 patches
```bash
python 04_patching.py
```
**Output**: 
- `patched_dataset_unified/images/` (PNG images)
- `patched_dataset_unified/labels.csv` (patch metadata)

#### e) Convert to HDF5 format
```bash
python 05_to_hdf5.py
```
**Output**: `patched_dataset_unified/xbd_dataset.hdf5`
---

#### Test Set Preparation (analogous to train)
```bash
cd 02_inference/data_prep_test
python 01_data_preprocessing_test.py
python 02_add_disaster_type.py
python 03_patching_test.py
python 04_preprocess_to_hdf5_test.py
```

### 2. Training

#### Best Configuration from my tests (trade-off between performance and time ) 
Edit `core/config.py` (`TrainingConfig` class):

```python
# BACKBONE
self.backbone_name = 'convnext_small'  # or 'resnet50', 'efficientnet_b0', etc.
self.use_timm = False  # True for ImageNet-22K, False for ImageNet-1K

# HYPERPARAMETERS
self.batch_size = 96
self.num_epochs = 12
self.learning_rate = 1e-4
self.dropout_rate = 0.5

# TRAINING STRATEGY
self.use_gradual_unfreezing = False  

# LOSS
self.loss_function = 'cross_entropy'  

# DATA AUGMENTATION
self.enable_data_augmentation = True
self.augment_horizontal_flip_prob = 0.5
self.augment_vertical_flip_prob = 0.5
self.augment_rotation_prob = 0.5
self.augment_color_jitter_prob = 0.8
```

#### Start Training
```bash
cd 01_training
python addestramento.py
```

**Output** (in `results/<experiment_name>/`):
```
results/exp_YYYYMMDD_HHMMSS/
├── fold_0/
│   ├── models/best_model_fold_0.pth
│   ├── plots/
│   │   ├── confusion_matrix_fold_0.png
│   │   ├── loss_curves_fold_0.png
│   │   └── disaster_matrices_fold_0.png
│   └── reports/
│       └── disaster_reports_fold_0.csv
├── fold_1/ ... fold_4/
└── training_summary.txt  # Aggregate metrics
```

### 3. Inference

#### a) Ensemble (recommended)
Uses all 5 fold models for more robust predictions.

```bash
cd 02_inference
# Edit InferenceConfig in core/config.py
python inference.py
```

**Output**: `results/<experiment>/predictions/predictions_<experiment>.csv`

#### b) Single Fold
Uses the model from a specific fold.

```bash
python inference_single_fold.py
# Edit FOLD_NUMBER in the script
```

**Output**: `results/<experiment>/fold_X/predictions/predictions_fold_X_test.csv`

#### c) Performance Evaluation
```bash
python evaluate_model.py
```

**Edit configuration in the script**:
```python
EXPERIMENT_NAME = 'exp_8_convenext_small'
EVALUATION_TYPE = 'ensemble'  # or 'single_fold'
FOLD_NUMBER = 1  # if single_fold
```

**Output** (in `results/<experiment>/evaluations/`):
- `classification_report_<type>.txt` - Per-class metrics
- `confusion_matrix_<type>.png` - Global confusion matrix
- `confusion_matrix_normalized_<type>.png` - Normalized by recall
- `disaster_matrices_<type>.png` - Separate matrices per disaster type
- `event_matrices_<type>.png` - Matrices per specific event

### 4. XAI Analysis

#### a) Single Example Explanation
Visualizes which superpixels the model considers important for a specific prediction.

```bash
cd 04_explainability/test/LIME
python explain_lime_test.py
```

**Configuration** (`ExplainLIMETestConfig` class):
```python
EXAMPLE_INDEX = 32837  # Index in test set
FOLD_TO_EXPLAIN = 1
EXPERIMENT_TO_EXPLAIN = "exp_8_convenext_small"
BACKBONE_NAME = 'convnext_small'

TARGET_CLASS_INDEX = None  # None = predicted class, 0-3 = specific class

# LIME PARAMETERS
NUM_SAMPLES = 2000  # Perturbations
NUM_FEATURES = 100  # Top superpixels
USE_STRATIFICATION = True  # Stratified sampling
```

**Output**: `xai_results/test_single/<scene_id>_lime_analysis.png`

#### b) Multi-Class Explanation
Shows what LIME "sees" for all 4 classes on the same example.

```bash
python explain_lime_all_classes_test.py
```

**Output**: Figure with 4 subplots (one heatmap per class)

#### c) LIME Quality Validation (Single)
Calculates Insertion/Deletion AUC metrics for one example.

```bash
python evaluate_lime_test.py
```

**Metrics**:
- **Insertion AUC**: How quickly confidence grows when adding important superpixels
- **Deletion AUC**: How quickly confidence drops when removing important superpixels

**Output**: Figure with Insertion/Deletion curves for all 4 classes

#### d) Batch Validation (400 samples/class)
Evaluates LIME on a stratified sample of the test set.

```bash
python evaluate_lime_test_batch.py
```

**Configuration**:
```python
SAMPLES_PER_CLASS = 400  # Stratified by disaster type
ONLY_CORRECT_PREDICTIONS = True 
NUM_SAMPLES = 500  # LIME perturbations (reduced vs single)
```

**Output** (in `xai_results/test_stratified_400perclass_<model>/`):
- `individual_results.csv` - AUC for each example
- `aggregate_summary.json` - Aggregate metrics
- `averaged_curves.png` - Interpolated average curves

#### e) LIME Results Analysis
Filters and aggregates batch results in various ways.

```bash
python analyze_lime_results.py --csv <path_to_individual_results.csv> --analysis all
```

**`--analysis` Options**:
- `all`: All analyses
- `overall`: Aggregate metrics (all classes)
- `damage_only`: Only damage classes (excludes no-damage)
- `by_disaster`: Aggregated by disaster type
- `by_class`: Aggregated by class
- `by_class_and_disaster`: Class × disaster matrix
- `visualizations`: Charts (boxplot, heatmap)

**Output** (in `xai_results/<batch_dir>/analysis/`):
- JSON with aggregate metrics
- Comparative charts


## Results 
(only tables)

### Model Performance (Test Set - esamble)

| Model | Macro F1 | F1 No-Damage | F1 Minor Damage | F1 Major Damage | Destroyed | Parameters |
|---------|----------|------------|--------|-----------|------------|----------|
| ConvNeXt-Small | **0.808** | 0.964 | 0.670 | 0.740 | 0.859 | 50M |
| ConvNeXt-Tiny | 0.804 | 0.964 | 0.658 | 0.733 | 0.864| 28M |
| ResNet50 | 0.799 | 0.962 | 0.656 | 0.726 | 0.853 | 25M |
| EfficientNet-B3 | 0.792 | 0.961 | 0.641 | 0.725 | 0.840 | 12M |
| EfficientNet-B0 | 0.793 | 0.961 | 0.647 |0.729| 0.836 | 5M |


### LIME - Explanation Quality (ConvNeXt-Small)

**Insertion/Deletion AUC Metrics** (400 samples/class, only damage classes):

| Model | Samples | Auc Deletion | Std del. | AUC Insertion | std Ins|
|----------------|-----|-----|---------|--------|-------|
| ConvNeXt-Small | 867 | 0.3524 | 0.2327 | 0.8372| 0.1595|
| ConvNeXt-Tiny | 841 | 0.3774 | 0.242 | 0.8097| 0.1736|
| ResNet50 | 857 | 0.3308 | 0.228| 0.7403|0.2069|

**Note**: No-Damage is excluded because removing portions of an intact building doesn't decrease confidence (the building remains intact).
