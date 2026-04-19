# IHGAMP

**Pan-cancer HRD prediction from routine H&E whole-slide images using foundation models.**

## Pipeline

```
WSI → Tissue Detection → Tiling (256×256, stride 512) → OpenCLIP ViT-B/16
    → 512d embeddings → Patient mean pooling → PCA (384) → Ridge (α=30)
    → Platt calibration → HRD probability
```

## Results

| Cohort | Endpoint | AUROC |
|--------|----------|-------|
| TCGA (OpenCLIP) | Genomic HRD | 0.766 |
| TCGA (OpenSlideFM) | Genomic HRD | 0.812 |
| CPTAC-LUAD (frozen) | Genomic HRD | 0.671 |
| PTRC-HGSOC | Platinum resistance | 0.673 |
| TCGA-UCEC (off-target) | MMR deficiency | 0.445 |

## Setup

```bash
pip install -r requirements.txt
```

Requires: Python ≥3.10, CUDA GPU, [OpenSlide](https://openslide.org/)

## Usage

Edit `CONFIG` class paths in `ihgamp.py`, then run sections sequentially.

## Data

- **TCGA/CPTAC:** [GDC Portal](https://portal.gdc.cancer.gov/)
- **PTRC-HGSOC:** [Bergstrom et al., JCO 2024](https://doi.org/10.1200/JCO.23.02641)
- **SurGen:** [Loeffler et al., BMC Biology 2024](https://doi.org/10.1186/s12915-024-02022-9)
- **scarHRD:** [GitHub](https://github.com/sztup/scarHRD)
- HRD labels for TCGA are derived from the TCGA DDR Data Resources (Knijnenburg et al., Cell Reports 2018), specifically the HRD_Score column in DDRscores.tsv, which computes the same LOH + TAI + LST scar signatures as the scarHRD R package but from ASCAT-derived copy-number calls rather than Sequenza. The ≥33 threshold corresponds to the top 20th percentile of training-set HRD_Score values.

## License

MIT
