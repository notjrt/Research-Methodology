# Hybrid ViT-XGBoost for Retinal OCT Classification

Official codebase for the MLDM Research Methodology project (Master Semester 1, UJM).

This repository implements a lightweight hybrid framework combining a frozen Vision Transformer (`vit_base_patch16_224`) feature extractor with PCA, latent-space SMOTE, and XGBoost classification for 4-class retinal disease detection on the Mendeley OCT2017 dataset (CNV, DME, Drusen, Normal).

## Authors
- Jayden Tan
- Yuhsaku Nakai
- Huong Dang Cao
- Pham Tuan Hiep

---

## Method Overview

```text
Raw OCT Image 
   │
   ▼
[1] CLAHE Contrast Enhancement & Resize (224x224)
   │
   ▼
[2] Frozen ViT Feature Extraction (768-D)
   │
   ▼
[3] 5-Fold Stratified Cross-Validation Partition (Leak-Free)
   │
   ▼
[4] PCA Dimensionality Reduction (150-D, fitted on fold train)
   │
   ▼
[5] Latent SMOTE Balancing (Applied strictly on fold train)
   │
   ▼
[6] Tuned XGBoost Classification & 1,000-Image Benchmark Evaluation
```

---

## Project Structure

```text
Research-Methodology/
├── configs/
│   └── config.yaml             # Experiment hyperparameters & dataset paths
├── src/
│   ├── data/
│   │   └── clahe_preprocessor.py # CLAHE enhancement & tensor normalization
│   ├── models/
│   │   ├── vit_extractor.py    # Pretrained ViT feature extractor
│   │   ├── dim_reduction.py    # PCA projection module
│   │   ├── balancer.py         # Latent-space SMOTE balancer
│   │   └── classifiers.py      # XGBoost and baseline models
│   ├── utils/
│   │   ├── seed.py             # Reproducibility utilities
│   │   └── metrics.py          # Multiclass evaluation metrics
│   └── pipeline/
│       ├── step1_extract_features.py # Feature extraction pipeline
│       └── step2_train_evaluate.py   # 5-fold training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Environment Setup

Requires **Python 3.10+**.

### Windows

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
.\.venv\Scripts\Activate.ps1
# (or in CMD: .\.venv\Scripts\activate.bat)

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users**: To install PyTorch with CUDA acceleration, run:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### macOS (Apple Silicon & Intel)

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision
pip install -r requirements.txt
```

---

## Quick Start

### 1. Extract Features (Run once, GPU/Colab recommended)
Extracts 768-D representation vectors from raw scans and saves them to `.npy`:
```bash
python -m src.pipeline.step1_extract_features
```

### 2. Train & Evaluate (Fast on CPU, ~3-5 mins)
Executes 5-Fold Stratified Cross-Validation with PCA, latent-space SMOTE, and tuned XGBoost:
```bash
python -m src.pipeline.step2_train_evaluate
```
