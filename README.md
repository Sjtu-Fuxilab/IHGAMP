# IHGAMP

**Pan-cancer prediction of homologous recombination deficiency (HRD) from routine H&E whole-slide images using foundation models.**

This repository contains the analysis code, trained-model artifacts, and
reproducibility scripts for the IHGAMP framework. The pipeline takes
unannotated H&E-stained histopathology slides as input and produces a
calibrated probability of HRD-positive status, validated across 31 TCGA
cancer types and seven independent external cohorts (CPTAC-LUAD/LUSC/HNSCC/UCEC,
PTRC-HGSOC, SurGen-CRC, OBR).

## Citation

S. A. Zafar, L. Chengliang, A. A. Khan, M. S. Faisal, and W. Qin,
"IHGAMP: Pan-Cancer HRD Prediction From Routine H&E Whole-Slide Images
Using Foundation Models," *APMIS* 134, no. 4 (2026): e70211,
https://doi.org/10.1111/apm.70211.

```bibtex
@article{zafar2026ihgamp,
  author    = {Zafar, Sanwal Ahmad and Chengliang, Liu and Khan, Areeba Ali and Faisal, Muhammad Salman and Qin, Wei},
  title     = {{IHGAMP}: {Pan-Cancer} {HRD} Prediction From Routine {H\&E} Whole-Slide Images Using Foundation Models},
  journal   = {APMIS},
  volume    = {134},
  number    = {4},
  pages     = {e70211},
  year      = {2026},
  doi       = {10.1111/apm.70211},
  url       = {https://doi.org/10.1111/apm.70211},
  publisher = {Wiley}
}
```

## Overview

| Component | Specification |
|---|---|
| Cohort | 19,996 WSIs / 10,797 patients / 31 TCGA cancer types; 8,109 with scarHRD |
| Input tile | 256 × 256 px at 20× (level 0; ~0.5 μm/px), stride 512, ≥30% tissue, max 2,000 tiles/slide |
| Total tiles | 27,608,061 |
| Backbone (primary) | OpenCLIP ViT-B/16, LAION-2B (`laion2b_s34b_b88k`), 512-d |
| Backbone (additional) | OpenSlideFM ViT-L/14 dual-scale (1,536-d), UNI ViT-L/16 (1,536-d) |
| Aggregation | Patient-level mean pooling |
| Model | StandardScaler → PCA(384) → Ridge(α=30.0) → Platt scaling |
| HRD label (TCGA) | scarHRD ≥ 33 (top 20th percentile of training distribution) |
| HRD label (CPTAC) | scarHRD ≥ 42 (clinical cutoff; Loeffler et al. 2024) |
| Splits | 80/10/10 stratified by cancer type, seed = 42 |
| Hardware | Single NVIDIA RTX 4090 (24 GB VRAM); ~143.7 hours total |
| Bootstrap reps | 200 internal, 2,000 external (Supp Table S5) |

**Headline numbers** (reproduced by the notebooks below):

- TCGA test AUROC = 0.766 (95% CI 0.727–0.803), val AUROC = 0.775 (NB07)
- TCGA test AUROC with OpenSlideFM = 0.812 (NB12)
- CPTAC-LUAD frozen-scorer AUROC = 0.671 (95% CI 0.487–0.825); within-cohort OSFM AUROC = 0.723 (NB14, NB15)
- PTRC-HGSOC platinum resistance AUROC = 0.673 (NB16)
- SurGen-CRC MMR off-target AUROC = 0.674; TCGA-UCEC MMR AUROC = 0.445 (NB17)
- TP53 partial r = 0.112 controlling for scarHRD; BRCA1/2 prediction AUC = 0.683 (NB18)
- TSS-level L2 norm CV = 2.1% across 710 tissue source sites (NB10)

## Repository structure

```
notebooks/
├── NB01_Setup_Environment.ipynb            # CUDA, OpenSlide, disk preflight
├── NB02_Tiling.ipynb                       # tile coordinate extraction
├── NB03_Feature_Extraction.ipynb           # OpenCLIP ViT-B/16 patient embeddings
├── NB04_Cohort_Labels.ipynb                # IFNG6 / ANGIO / HRR signatures + 80/10/10 splits
├── NB05_scarHRD_Labels.ipynb               # scarHRD (LOH+TAI+LST) genomic labels
├── NB06_Feature_QC.ipynb                   # patient-level sanitization
├── NB07_Internal_Training.ipynb            # PCA(384) → Ridge(30) → Platt; Figure 3A-D
├── NB08_LOCO.ipynb                         # leave-one-cancer-out; Figure 4C
├── NB09_LeakGuard.ipynb                    # name + correlation-based feature filter
├── NB10_TSS_BatchEffects.ipynb             # 710-site L2 norm stability; Figure 4D
├── NB11_UNI_Variants.ipynb                 # UNI BRCA / LUAD specific variants (Table 3)
├── NB12_OpenSlideFM_Internal.ipynb         # OpenSlideFM TCGA evaluation (Table 3, 0.812)
├── NB13_CPTAC_Loeffler_Labels.ipynb        # CPTAC HRD labels from Loeffler et al. (2024)
├── NB14_CPTAC_External_Frozen_OpenCLIP.ipynb  # frozen scorer on CPTAC; Figure 4B
├── NB15_CPTAC_External_OSFM.ipynb          # OSFM within-cohort CV on CPTAC
├── NB16_PTRC_Platinum.ipynb                # PTRC-HGSOC platinum resistance + Youden J
├── NB17_OffTarget_MMR.ipynb                # SurGen-CRC + TCGA-UCEC MMR
├── NB18_Pathway_Specificity.ipynb          # TP53, BRCA1/2, label-strategy ablations
└── NB19_External_ForestPlot.ipynb          # Figure 4A consolidator
```

The notebooks are independent at the level of input/output files — each one
reads parquets/CSVs/JSONs written by upstream notebooks and writes its own
outputs into the workspace. Running them in numerical order produces the
complete set of figures and tables in the published paper.

## Requirements

- Python ≥ 3.10
- See `requirements.txt`. Core dependencies:

```
numpy, pandas, scikit-learn, joblib, scipy
torch, torchvision, open_clip_torch
openslide-python, opencv-python, pyarrow, openpyxl
xenaPython, matplotlib, tqdm
```

A single GPU with at least 12 GB VRAM is recommended. The published runs used
an NVIDIA RTX 4090 (24 GB) with CUDA 12.1, mixed precision, and TF32 enabled.

## Environment variables

The pipeline reads paths from environment variables. Set them once before
running notebooks:

| Variable | Used by | Description |
|---|---|---|
| `WORKSPACE` | all | Output directory; default `./workspace` |
| `WSI_ROOT` | NB02, NB03, NB04 | TCGA diagnostic WSIs root directory |
| `DDR_DIR` | NB05 | TCGA DDR resource (`DDRscores.tsv`, `Scores.tsv`, `Samples.tsv`) |
| `MC3_MAF_PATH` | NB17, NB18 | TCGA `mc3.v0.2.8.PUBLIC.maf` |
| `UNI_FEATURES_DIR` | NB11 | Pre-computed UNI patient embeddings: `uni_<cancer>_patient_embeddings.parquet` |
| `OSFM_FEATURES_PARQUET` | NB12 | TCGA OpenSlideFM patient embeddings (single parquet) |
| `CPTAC_LABELS_XLSX` | NB13 | Loeffler et al. (2024) HRD table; CPTAC sheet |
| `CPTAC_WSI_MANIFEST` | NB13 | Local CPTAC slide manifest with patient + cancer columns |
| `CPTAC_OPENCLIP_DIR` | NB14 | Per-cohort `cptac_<cohort>_openclip_patient_embeddings.parquet` |
| `CPTAC_OSFM_DIR` | NB15 | Per-cohort `cptac_<cohort>_osfm_patient_embeddings.parquet` |
| `PTRC_OSFM_FEATURES_PARQUET` | NB16 | PTRC-HGSOC patient-level OSFM embeddings |
| `PTRC_LABELS_CSV` | NB16 | PTRC platinum response labels |
| `SURGEN_OPENCLIP_FEATURES_PARQUET` | NB17 | SurGen-CRC patient-level OpenCLIP ViT-B/16 embeddings |
| `SURGEN_MMR_LABELS_CSV` | NB17 | SurGen MMR/MSI status |

External feature extractions (UNI, OSFM, CPTAC OpenCLIP, CPTAC OSFM, PTRC OSFM,
SurGen OpenCLIP) are not part of this repository because they require
backbone-specific checkpoints and large compute. Each notebook documents
the expected file format in its docstring.

## Quickstart

```bash
git clone https://github.com/Sjtu-Fuxilab/IHGAMP.git
cd IHGAMP
pip install -r requirements.txt

export WORKSPACE=./workspace
export WSI_ROOT=/path/to/tcga/diagnostic_wsis
export DDR_DIR=/path/to/tcga_ddr_resource
export MC3_MAF_PATH=/path/to/mc3.v0.2.8.PUBLIC.maf

# core internal pipeline
jupyter nbconvert --to notebook --execute notebooks/NB01_Setup_Environment.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB02_Tiling.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB03_Feature_Extraction.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB04_Cohort_Labels.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB05_scarHRD_Labels.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB06_Feature_QC.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB07_Internal_Training.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB08_LOCO.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB09_LeakGuard.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB10_TSS_BatchEffects.ipynb

# external validations require their respective env vars set
jupyter nbconvert --to notebook --execute notebooks/NB14_CPTAC_External_Frozen_OpenCLIP.ipynb
jupyter nbconvert --to notebook --execute notebooks/NB16_PTRC_Platinum.ipynb
# ... etc
```

A `run_all.sh` script that walks the full set with environment checks is
included separately.

## Reproducibility cross-checks

Each notebook prints its computed values alongside the manuscript reference.
Expected matches under SEED = 42:

| Quantity | Manuscript | Notebook |
|---|---|---|
| TCGA test AUROC (OpenCLIP) | 0.766 (0.727–0.803) | NB07 |
| TCGA val AUROC (OpenCLIP) | 0.775 (0.739–0.808) | NB07 |
| TCGA test AUROC (OpenSlideFM) | 0.812 | NB12 |
| TCGA test HRD prevalence | 21.2% (177/833) | NB07 |
| Patients with scarHRD | 8,109 | NB05 |
| HRD prevalence overall | 20.4% (1,655) | NB05 |
| TSS sites | 710 | NB10 |
| TSS L2 mean / CV | 13.503 / 2.1% | NB10 |
| LOCO BRCA / LUAD / LUSC / HNSC / CESC | 0.660 / 0.627 / 0.628 / 0.563 / 0.502 | NB08 |
| LOCO mean adeno-serous / squamous | 0.60 / 0.56 | NB08 |
| CPTAC-LUAD frozen scorer | 0.671 (0.487–0.825) | NB14 |
| CPTAC-LUAD / LUSC / HNSCC OSFM | 0.723 / 0.527 / 0.475 | NB15 |
| PTRC-HGSOC platinum resistance | 0.673 (0.588–0.757), AP 0.631 | NB16 |
| PTRC Youden J / sens / spec / PPV / NPV | 0.439 / 0.612 / 0.725 / 0.621 / 0.717 | NB16 |
| SurGen-CRC MMR off-target | 0.674 (0.55–0.79), AP 0.134 | NB17 |
| TCGA-UCEC MMR off-target | 0.445 (0.353–0.545) | NB17 |
| TP53 partial r given scarHRD | 0.112 | NB18 |
| TP53 AUC within HRD+ | 0.504 | NB18 |
| BRCA1/2 prediction (off-target) | 0.683 | NB18 |
| BRCA1/2-trained self-prediction | 0.486 | NB18 |
| scarHRD-trained in BRCA-wildtype | 0.772 | NB18 |
| Label strategies AUC at ≥33 / ≥42 | 0.765 / 0.773 | NB18 |
| OpenCLIP weights SHA-256 | matches between runs | NB03 |

External validation numbers depend on availability of upstream feature
extractions; missing dependencies fail fast with a clear error message
indicating which env var is unset.

## Data availability

| Dataset | Source |
|---|---|
| TCGA WSIs | https://portal.gdc.cancer.gov/ |
| TCGA RNA-seq (Toil pipeline) | https://xenabrowser.net/datapages/?dataset=TcgaTargetGtex_rsem_gene_tpm |
| TCGA scarHRD scores (DDR resource) | https://gdc.cancer.gov/about-data/publications/PanCan-DDR-2018 |
| TCGA mc3 somatic mutations | https://gdc.cancer.gov/about-data/publications/mc3-2017 |
| CPTAC WSIs | https://www.cancerimagingarchive.net/ , https://pdc.cancer.gov/ |
| CPTAC HRD labels (Loeffler et al. 2024) | https://doi.org/10.1186/s12915-024-02022-9 |
| PTRC-HGSOC | Bergstrom et al. (PTRC Pan-Cancer Initiative) |
| SurGen-CRC | Loeffler et al. 2024 (companion cohort) |

The scarHRD scores were computed using the scarHRD R package
(https://github.com/sztup/scarHRD) as described by Sztupinszki et al.

## Trained model artifacts

`models/frozen_model.joblib` (saved by NB07) contains the fitted pipeline
ready for inference on new OpenCLIP ViT-B/16 patient embeddings:

```python
import joblib, numpy as np
m = joblib.load("models/frozen_model.joblib")
pipe, platt = m["pipe"], m["platt"]

# X is an (n_patients, 512) array of mean-pooled OpenCLIP ViT-B/16 features
z = pipe.predict(X).reshape(-1, 1)
prob_HRD = platt.predict_proba(z)[:, 1]
```

`models/frozen_model_osfm.joblib` (saved by NB12) provides the analogous
inference path for OpenSlideFM 1,536-d embeddings.

## Funding

This work was supported by the Shanghai Jiao Tong University Interdisciplinary
Research Program (grant YG2025QNA31).

## License

Code is released under the MIT License (see `LICENSE`). Trained model
weights inherit the licenses of their source datasets and pretrained
backbones (OpenCLIP / LAION-2B; OpenSlideFM; UNI).

## Acknowledgments

We thank the TCGA Research Network, the CPTAC Investigators, the PTRC
Investigators, the OpenCLIP and LAION-2B teams, and the authors of UNI
(Chen et al., 2024) for making their data and pretrained models publicly
available.

Corresponding author: wqin@sjtu.edu.cn
