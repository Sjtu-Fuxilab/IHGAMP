"""
Utility functions for IHGAMP.
"""

from __future__ import annotations
import os
import yaml
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PathConfig:
    wsi_root: str = ""
    output_dir: str = "./outputs"
    embeddings_dir: str = "./embeddings"
    models_dir: str = "./models"
    results_dir: str = "./results"


@dataclass  
class PreprocessConfig:
    patch_size: int = 256
    stride: int = 256
    tissue_threshold: float = 0.60
    patch_cap: int = 3000
    level: int = 0
    target_mpp: float = 0.5


@dataclass
class ModelConfig:
    pca_components: int = 128
    ridge_alpha: float = 1.0
    calibration: str = "platt"
    seed: int = 42


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    config = Config()
    if 'paths' in data:
        config.paths = PathConfig(**data['paths'])
    if 'preprocessing' in data:
        config.preprocessing = PreprocessConfig(**data['preprocessing'])
    if 'model' in data:
        config.model = ModelConfig(**data['model'])

    return config


# ============================================================================
# File I/O
# ============================================================================

def read_any_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read tabular data from various formats."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(path)
    elif suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    elif suffix in ['.parquet', '.pq']:
        return pd.read_parquet(path)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif suffix == '.feather':
        return pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_results(
    data: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "results"
) -> Dict[str, Path]:
    """Save results to multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON summary
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    paths['json'] = json_path

    return paths


# ============================================================================
# Data Utilities  
# ============================================================================

def extract_patient_id(slide_name: str, dataset: str = "TCGA") -> str:
    """Extract patient ID from slide filename."""
    if dataset.upper() == "TCGA":
        # TCGA format: TCGA-XX-XXXX-01A-...
        parts = slide_name.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
    return slide_name.split('.')[0]


def get_tissue_mask(
    image: np.ndarray,
    threshold: float = 0.60
) -> np.ndarray:
    """Generate tissue mask from thumbnail image."""
    import cv2

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Otsu's thresholding
    _, mask = cv2.threshold(
        gray, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask > 0


def compute_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """Compute hash of file for integrity checking."""
    hash_func = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Simple logger for experiment tracking."""

    def __init__(self, log_dir: Union[str, Path], name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.entries = []

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().isoformat()
        entry = {"timestamp": timestamp, "level": level, "message": message}
        self.entries.append(entry)
        print(f"[{level}] {message}")

    def save(self):
        path = self.log_dir / f"{self.name}.json"
        with open(path, 'w') as f:
            json.dump(self.entries, f, indent=2)
        return path
