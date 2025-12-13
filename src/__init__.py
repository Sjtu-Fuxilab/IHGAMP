"""
IHGAMP: Integrative Histopathology-Genomic Analysis for Molecular Phenotyping
=============================================================================

A computational framework for predicting molecular phenotypes from H&E slides.
"""

__version__ = "0.1.0"
__author__ = "Sanwal Ahmad Zafar"

from .utils import load_config
from .preprocessing import TileExtractor
from .embeddings import EmbeddingExtractor
from .models import HRDPredictor
from .evaluation import evaluate_with_calibration

__all__ = [
    "load_config",
    "TileExtractor", 
    "EmbeddingExtractor",
    "HRDPredictor",
    "evaluate_with_calibration",
]
