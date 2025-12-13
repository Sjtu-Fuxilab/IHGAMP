# IHGAMP: Integrative Histopathology-Genomic Analysis for Molecular Phenotyping

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## Overview

IHGAMP is a computational framework for predicting molecular phenotypes (e.g., HRD status, MMR deficiency) directly from H&E-stained whole slide images (WSIs) using foundation model embeddings and interpretable machine learning.

### Key Features

- **Foundation Model Integration**: Leverages state-of-the-art pathology foundation models (UNI, OpenSlideFM) for robust feature extraction
- **Multi-Cancer Generalization**: Validated across 30+ cancer types from TCGA (n=20,000 slides)
- **External Validation**: Independently validated on CPTAC, PTRC-HGSOC, and other cohorts
- **Calibrated Predictions**: Implements Platt scaling and isotonic regression for reliable probability estimates
- **Clinical Operating Points**: Provides threshold selection for NPV≥0.95 and PPV≥0.60 clinical scenarios

## Repository Structure

```
IHGAMP/
├── notebooks/
│   ├── 01_preprocessing.ipynb      # WSI tiling and quality control
│   ├── 02_feature_extraction.ipynb # Foundation model embedding
│   ├── 03_model_training.ipynb     # HRD/MMR prediction models
│   ├── 04_evaluation.ipynb         # Metrics, calibration, figures
│   └── 05_external_validation.ipynb# CPTAC and other cohorts
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Tiling and normalization
│   ├── embeddings.py               # Feature extraction utilities
│   ├── models.py                   # Model definitions
│   ├── evaluation.py               # Metrics and calibration
│   └── utils.py                    # Common utilities
├── configs/
│   └── default.yaml                # Configuration template
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Sjtu-Fuxilab/IHGAMP.git
cd IHGAMP

# Create conda environment
conda create -n ihgamp python=3.10
conda activate ihgamp

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Configuration

```python
from src.utils import load_config

config = load_config("configs/default.yaml")
config.paths.wsi_root = "/path/to/your/slides"
config.paths.output_dir = "/path/to/outputs"
```

### 2. Feature Extraction

```python
from src.embeddings import extract_patient_embeddings

embeddings = extract_patient_embeddings(
    wsi_dir=config.paths.wsi_root,
    model_name="UNI",  # or "OpenSlideFM"
    output_path="embeddings.parquet"
)
```

### 3. Model Training & Evaluation

```python
from src.models import HRDPredictor
from src.evaluation import evaluate_with_calibration

model = HRDPredictor(n_components=128, alpha=1.0)
model.fit(X_train, y_train)

results = evaluate_with_calibration(
    model, X_test, y_test,
    calibration_methods=["platt", "isotonic"]
)
```

## Model Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   WSI Input     │────▶│ Foundation Model │────▶│ Patient-level   │
│  (H&E Slide)    │     │   (UNI/OSFM)     │     │   Embedding     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   StandardScaler │◀─────────────┘
                        │   + PCA (128-d)  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Ridge Regressor │
                        │   (α = 1.0)      │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ Platt Calibration│
                        │ (Logistic Reg)   │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  P(HRD-high)     │
                        │   [0, 1]         │
                        └──────────────────┘
```

## Results Summary

### Internal Validation (TCGA, n=1,847)

| Split | AUC | 95% CI | AP | Brier |
|-------|-----|--------|-----|-------|
| Validation | 0.72 | 0.68-0.76 | 0.58 | 0.19 |
| Test | 0.70 | 0.66-0.74 | 0.55 | 0.20 |

### External Validation

| Cohort | n | AUC | 95% CI | Task |
|--------|---|-----|--------|------|
| CPTAC-LUAD | 106 | 0.72 | 0.58-0.85 | HRD |
| CPTAC-LUSC | 108 | 0.53 | 0.41-0.64 | HRD |
| PTRC-HGSOC | 158 | 0.67 | 0.59-0.76 | Platinum Response |
| OBR | 21 | 0.85 | 0.63-0.99 | Bevacizumab Response |

## Clinical Operating Points

| Rule | Threshold | Sensitivity | Specificity | PPV | NPV |
|------|-----------|-------------|-------------|-----|-----|
| Youden-J | 0.42 | 0.65 | 0.70 | 0.48 | 0.82 |
| NPV ≥ 0.95 | 0.18 | 0.92 | 0.35 | 0.32 | 0.95 |
| PPV ≥ 0.60 | 0.58 | 0.38 | 0.88 | 0.60 | 0.75 |

## Citation

```bibtex
@article{ihgamp2025,
  title={IHGAMP: Integrative Histopathology-Genomic Analysis for Molecular Phenotyping},
  author={Zafar, Sanwal Ahmad and Qin, Wei and others},
  journal={In preparation},
  year={2025},
  institution={Shanghai Jiao Tong University}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TCGA Research Network for providing the genomic and imaging data
- CPTAC for external validation cohorts
- Developers of UNI and OpenSlideFM foundation models

## Contact

- **Sanwal Ahmad Zafar** - PhD Candidate, Shanghai Jiao Tong University
- **Wei Qin** - Associate Professor, Shanghai Jiao Tong University
- **Lab**: [FuxiLab](https://github.com/Sjtu-Fuxilab)

---
<p align="center">
  <i>Developed at the Department of Industrial Engineering and Management, Shanghai Jiao Tong University</i>
</p>
