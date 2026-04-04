#!/usr/bin/env python3
"""
IHGAMP: Pan-cancer HRD Prediction from Routine H&E Whole-Slide Images
        Using Foundation Models


Pipeline: WSI -> Tissue Detection -> Tiling (256x256, stride 512)
          -> OpenCLIP ViT-B/16 -> 512d embeddings -> Patient mean pooling
          -> PCA (384) -> Ridge (alpha=30) -> Platt calibration -> HRD probability

Usage: Run each section sequentially. Edit CONFIG paths below first.
"""
from __future__ import annotations

# CONFIGURATION - Edit paths to match your local setup
class CONFIG:
    TCGA_WSI_ROOT       = r"data/tcga/wsis"
    TCGA_LABELS          = r"artifacts/labels/labels.parquet"
    TCGA_EMBEDDINGS      = r"artifacts/embeddings/patient_means_openclip_vitb16.parquet"
    TCGA_OSFM_EMBEDDINGS = r"artifacts/embeddings/patient_means_openslidefm.parquet"
    RUNS_DIR             = r"artifacts"
    CPTAC_LABELS_XLSX    = r"data/cptac/labels/el_nahhas_hrd.xlsx"
    PTRC_WSI_DIR         = r"data/ptrc_hgsoc/wsis"
    PTRC_CLINICAL        = r"data/ptrc_hgsoc/clinical_data.xlsx"
    SURGEN_WSI_DIR       = r"data/surgen/wsis"
    OSFM_CHECKPOINT      = r"weights/openslidefm_student.pt"
    MC3_MAF              = r"artifacts/mc3.v0.2.8.PUBLIC.maf"
    OUTPUT_DIR           = r"results"
    PCA_N = 384; RIDGE_ALPHA = 30.0; HRD_THR = 33; SEED = 42
    TILE_SIZE = 256; STRIDE = 512; MAX_TILES = 2000; TISSUE_FRAC = 0.30
    BOOT_N = 2000; TOP_FRAC = 0.20


# SECTION 1: PROJECT SETUP & ENVIRONMENT CHECK

# Sets up a project 
import os, sys, platform, shutil, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Point this to your clean project folder (you said you created this):
PROJECT_ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")

# You can set this later when downloads finish; leave "" for now:
WSI_ROOT = Path(r"")  # e.g., Path(r"D:\个人文件夹\Sanwal\ALL_WSI")

EXPECTED_SLIDES = 20000
PATCH_SIZE      = 256
STRIDE          = 256
PATCH_CAP       = 3000
MACENKO_ON      = True
FOUNDATION_MODEL= "UNI"   # "UNI" | "Virchow" | "PLIP"

RUN_GPU_SMOKETEST = True  # set False if you just want quick checks

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_wsl() -> bool:
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False

def disk_free_bytes(path: Path) -> int:
    try:
        total, used, free = shutil.disk_usage(path)
        return free
    except Exception:
        return 0

def bytes_to_gb(x: int | float) -> float:
    try:
        return round(float(x) / (1024**3), 2)
    except Exception:
        return 0.0

@dataclass
class Estimates:
    expected_slides: int = 20000
    patch_cap: int = 3000
    jpeg_kb_per_patch: int = 60   # ~256x256 JPEG avg
    embed_dim: int = 1024
    embed_dtype_bytes: int = 2    # float16 storage

    def tiles_gb(self) -> float:
        total_kb = self.expected_slides * self.patch_cap * self.jpeg_kb_per_patch
        return round(total_kb / (1024**2), 2)  # KB -> GB

    def embeddings_gb(self) -> float:
        bytes_total = self.expected_slides * self.patch_cap * self.embed_dim * self.embed_dtype_bytes
        return round(bytes_total / (1024**3), 2)

def check_torch() -> dict:
    info = {"installed": False, "cuda_available": False, "version": None, "cuda_version": None,
            "device": None, "vram_gb": None}
    try:
        import torch
        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            dev = torch.cuda.current_device()
            info["device"] = torch.cuda.get_device_name(dev)
            info["vram_gb"] = round(torch.cuda.get_device_properties(dev).total_memory / (1024**3), 2)
            info["cuda_version"] = torch.version.cuda
    except Exception as e:
        info["error"] = str(e)
    return info

def check_openslide() -> dict:
    info = {"installed": False, "version": None}
    try:
        import openslide  # noqa
        info["installed"] = True
        info["version"] = getattr(sys.modules.get("openslide"), "__version__", "unknown")
    except Exception as e:
        info["error"] = str(e)
    return info

def check_cv2() -> dict:
    info = {"installed": False, "version": None}
    try:
        import cv2
        info["installed"] = True
        info["version"] = cv2.__version__
    except Exception as e:
        info["error"] = str(e)
    return info

def gpu_smoketest(seconds: int = 5) -> dict:
    """Tiny matmul loop in float16 to prove GPU works end-to-end."""
    try:
        if not torch.cuda.is_available():
            return {"ran": False, "reason": "cuda_not_available"}
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dev = torch.device("cuda:0")
        n = 4096
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        b = torch.randn(n, n, device=dev, dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 0
        while time.time() - t0 < seconds:
            _ = torch.matmul(a, b)
            iters += 1
        torch.cuda.synchronize()
        return {"ran": True, "iters": iters, "secs": seconds}
    except Exception as e:
        return {"ran": False, "reason": str(e)}

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
folders = [
    "artifacts/signatures",
    "scripts",
    "runs",
    "embeddings",
    "surrogates",
    "spatial",
    "models",
    "results/preflight",
    "figures",
    "data/examples",
]
for f in folders:
    (PROJECT_ROOT / f).mkdir(parents=True, exist_ok=True)

os_info = {
    "platform": platform.system(),
    "release": platform.release(),
    "python": sys.version.split()[0],
    "is_wsl2": is_wsl(),
}
torch_info     = check_torch()
openslide_info = check_openslide()
cv2_info       = check_cv2()

est = Estimates(EXPECTED_SLIDES, PATCH_CAP)
free_root_gb = bytes_to_gb(disk_free_bytes(PROJECT_ROOT))
free_wsi_gb  = bytes_to_gb(disk_free_bytes(WSI_ROOT)) if str(WSI_ROOT) else None

hints = {
    "tiles_estimated_gb": est.tiles_gb(),
    "embeddings_estimated_gb": est.embeddings_gb(),
    "free_space_project_gb": free_root_gb,
    "free_space_wsi_gb": free_wsi_gb,
    "recommendation": (
        "Plan to delete tiles after verifying embeddings to save TBs."
        if est.tiles_gb() > 2 * est.embeddings_gb() else
        "You can keep tiles if you have ample space."
    ),
}

config_yaml = f"""# artifacts/config.yaml
paths:
  project_root: {PROJECT_ROOT.as_posix()}
  wsi_root: {WSI_ROOT.as_posix() if str(WSI_ROOT) else ""}
  runs_dir: {(PROJECT_ROOT / "runs").as_posix()}
  embeddings_dir: {(PROJECT_ROOT / "embeddings").as_posix()}
  surrogates_dir: {(PROJECT_ROOT / "surrogates").as_posix()}
  spatial_dir: {(PROJECT_ROOT / "spatial").as_posix()}
  models_dir: {(PROJECT_ROOT / "models").as_posix()}
  results_dir: {(PROJECT_ROOT / "results").as_posix()}
  figures_dir: {(PROJECT_ROOT / "figures").as_posix()}

preprocess:
  patch_size: {PATCH_SIZE}
  stride: {STRIDE}
  tissue_thresh: 0.60
  patch_cap: {PATCH_CAP}
  level: 0
  macenko: {"true" if MACENKO_ON else "false"}
  macenko_target_image: ""   # set later if using normalization

foundation_model:
  name: {FOUNDATION_MODEL}
  embedding_dim: 1024
  precision: fp16

training:
  seed: 42
  num_workers: 12
  amp: true
  site_disjoint_splits: true

reporting:
  calibration_ece_target: 0.07
  ood_drop_rel_target: 0.15
  fairness_gap_target: 0.07
"""
(PROJECT_ROOT / "artifacts" / "config.yaml").write_text(config_yaml, encoding="utf-8")

gitignore = """
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
*.ckpt
*.pt
*.pth
*.onnx
*.h5
*.npy
*.npz
*.parquet
*.feather
*.log
*.tmp
runs/**
embeddings/**
surrogates/**
spatial/**
models/**
results/**
figures/**
data/**
!data/examples/**
"""
(PROJECT_ROOT / ".gitignore").write_text(gitignore.strip() + "\n", encoding="utf-8")

gate = {
    "timestamp": now(),
    "os": os_info,
    "torch": torch_info,
    "openslide": openslide_info,
    "opencv": cv2_info,
    "paths": {"project_root": str(PROJECT_ROOT), "wsi_root": str(WSI_ROOT)},
    "estimates": hints,
    "gpu_smoketest": None,
    "pass": True,
    "notes": []
}

# Torch/CUDA checks
if not torch_info.get("installed"):
    gate["pass"] = False
    gate["notes"].append(
        "PyTorch not installed. Install CUDA-enabled build (e.g., pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision)."
    )
elif not torch_info.get("cuda_available"):
    gate["pass"] = False
    gate["notes"].append(
        "CUDA not available in PyTorch. Check NVIDIA driver and install a CUDA-enabled torch (cu121+ for RTX 4090)."
    )

# OpenSlide
if not openslide_info.get("installed"):
    gate["pass"] = False
    gate["notes"].append(
        "openslide-python missing or OpenSlide libs not found. On Windows: install OpenSlide binaries and add to PATH, then pip install openslide-python."
    )

# Disk space (require embeddings + 50GB buffer at project root)
need_min_gb = est.embeddings_gb() + 50
if free_root_gb < need_min_gb:
    gate["pass"] = False
    gate["notes"].append(
        f"Low disk space at project root. Need at least ~{need_min_gb:.1f} GB free (embeddings + buffer)."
    )

# Optional: WSI root sanity hint
if free_wsi_gb is not None and free_wsi_gb < 10:
    gate["notes"].append("WSI root has <10 GB free; ensure downloads target a drive with ample space.")

# GPU smoketest (optional)
if RUN_GPU_SMOKETEST:
    try:
        smoke = gpu_smoketest(seconds=5)
    except Exception as e:
        smoke = {"ran": False, "reason": str(e)}
    gate["gpu_smoketest"] = smoke
    if not smoke.get("ran", False):
        gate["notes"].append(f"GPU smoketest did not run: {smoke.get('reason', 'unknown')}")

preflight_dir = PROJECT_ROOT / "results" / "preflight"
preflight_dir.mkdir(parents=True, exist_ok=True)
(preflight_dir / "gate_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

summary_lines = []
summary_lines.append("# Preflight Summary")
summary_lines.append(f"- Time: {now()}")
summary_lines.append(f"- OS: {os_info}")
summary_lines.append(f"- Torch: {torch_info}")
summary_lines.append(f"- OpenSlide: {openslide_info}")
summary_lines.append(f"- OpenCV: {cv2_info}")
summary_lines.append(f"- Estimates: {hints}")
summary_lines.append(f"- PASS: {gate['pass']}")
if gate["notes"]:
    summary_lines.append("## Notes")
    for n in gate["notes"]:
        summary_lines.append(f"- {n}")
(PROJECT_ROOT / "results" / "preflight" / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n=== IHGAMP Preflight ===")
print("Project root:", PROJECT_ROOT)
print("WSI root    :", "(not set yet)" if not str(WSI_ROOT) else WSI_ROOT)
print("GPU         :", torch_info.get("device"), "| VRAM (GB):", torch_info.get("vram_gb"), "| CUDA:", torch_info.get("cuda_version"))
print("OpenSlide   :", "OK" if openslide_info.get("installed") else "MISSING")
print("OpenCV      :", "OK" if cv2_info.get("installed") else "MISSING")
print("Estimate    : tiles ~", hints["tiles_estimated_gb"], "GB | embeddings ~", hints["embeddings_estimated_gb"], "GB")
print("Free space  :", free_root_gb, "GB @ project root", end="")
if free_wsi_gb is not None:
    print(" |", free_wsi_gb, "GB @ WSI root")
else:
    print()

if RUN_GPU_SMOKETEST:
    print("GPU smoketest:", gate["gpu_smoketest"])

print("\nGate:", "✅ PASS" if gate["pass"] else "❌ FAIL", "— see", preflight_dir / "SUMMARY.md")
print("Config written to:", PROJECT_ROOT / "artifacts" / "config.yaml")
print("Gitignore at     :", PROJECT_ROOT / ".gitignore")


# SECTION 2: WSI PREPROCESSING & REGISTRY

# Script 01 — Preprocess WSIs


import os
import re
import json
import time
import math
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    pass
except Exception:
    cv2 = None

try:
    pass
except Exception as e:
    raise RuntimeError(
        "openslide-python is required (and OpenSlide DLLs on Windows). "
        "Install OpenSlide, add its bin to PATH, then `pip install openslide-python`."
    ) from e

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")
WSI_ROOT     = Path(r"D:\个人文件夹\Sanwal\DL_V2\Histo slides 20k")

PATCH_SIZE     = 256
STRIDE         = 384      # faster; reduce candidates
TISSUE_THRESH  = 0.60
PATCH_CAP      = 2000
LEVEL          = 0

MACENKO_ON             = False   # Off for speed; we’ll rely on strong color aug downstream
MACENKO_TARGET_IMAGE   = ""      # leave empty (unused when MACENKO_ON=False)

NUM_WORKERS    = 20      # good balance for heavy I/O on a fast workstation
JPEG_QUALITY   = 90
SAVE_THUMB     = True

# Diagnostic Gate thresholds (Nature-level, but practical for STRIDE=384)
MIN_PATCHES_PER_SLIDE = 600      # slides with fewer are flagged
COVERAGE_MIN          = 0.98     # ≥98% slides should meet the minimum
CAP_HIT_MAX           = 0.20     # ≤20% slides should hit the cap with these settings


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_ID   = f"run_{ts()}_L{LEVEL}_ps{PATCH_SIZE}_st{STRIDE}_cap{PATCH_CAP}" + ("_mac" if MACENKO_ON else "_nomac")
RUN_DIR  = PROJECT_ROOT / "runs" / RUN_ID
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_ID
FIG_DIR  = PROJECT_ROOT / "figures" / RUN_ID
for _p in [RUN_DIR, RESULTS_DIR, FIG_DIR]:
    _p.mkdir(parents=True, exist_ok=False)  # fail fast if exists

LOG_PATH = RUN_DIR / "run.log"
def log(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg)


WSI_EXTS = {".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".svslide"}

def is_wsi(p: Path) -> bool:
    return p.suffix.lower() in WSI_EXTS

def patient_from_name(name: str) -> str:
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", name, re.I)
    if m: return m.group(1).upper()
    return Path(name).stem.upper()

def cancer_from_path(p: Path, root: Path) -> str:
    try:
        rel = p.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def tissue_mask_rgb(img_rgb: np.ndarray, thresh: float=0.60) -> np.ndarray:
    if cv2 is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        mask = (sat > 0.10) & (val < 0.98)
    else:
        gray = img_rgb.mean(axis=2)
        mask = gray < 240
    return mask if mask.mean() >= thresh else np.zeros_like(mask, dtype=bool)


@dataclass
class MacenkoTarget:
    V: np.ndarray      # (3,2)
    maxC: np.ndarray   # (2,)

def macenko_fit_target_from_image(target_rgb: np.ndarray) -> Optional[MacenkoTarget]:
    if cv2 is None:
        return None
    Ih = (target_rgb.astype(np.float32) + 1) / 255.0
    od = -np.log(np.clip(Ih, 1e-6, 1.0)).reshape(-1, 3)
    od = od[np.all(od > 0.05, axis=1)]
    if od.size == 0:
        return None
    cov = np.cov(od, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order[:2]].astype(np.float32)
    for i in range(2):
        V[:, i] /= (np.linalg.norm(V[:, i]) + 1e-8)
    C = od @ V
    maxC = np.percentile(C, 99, axis=0).astype(np.float32)
    return MacenkoTarget(V=V, maxC=maxC)

def macenko_normalize(img_rgb: np.ndarray, target: Optional[MacenkoTarget]) -> np.ndarray:
    if cv2 is None or target is None:
        return img_rgb
    Ih = (img_rgb.astype(np.float32) + 1) / 255.0
    OD = -np.log(np.clip(Ih, 1e-6, 1.0)).reshape(-1, 3)
    V = target.V
    C, _, _, _ = np.linalg.lstsq(V, OD.T, rcond=None)
    Ct = C.copy()
    for i in range(min(2, Ct.shape[0])):
        scale = (target.maxC[i] / (np.percentile(C[i, :], 99) + 1e-8))
        Ct[i, :] = Ct[i, :] * scale
    OD_hat = (V @ Ct).T
    Ihat = np.exp(-OD_hat).reshape(img_rgb.shape)
    return (Ihat * 255.0).clip(0, 255).astype(np.uint8)


def build_registry(wsi_root: Path) -> pd.DataFrame:
    rows = []
    for p in wsi_root.rglob("*"):
        if p.is_file() and is_wsi(p):
            rows.append({
                "slide_path": str(p),
                "slide_id": p.stem,
                "patient_id": patient_from_name(p.name),
                "cancer_type": cancer_from_path(p, wsi_root)
            })
    df = pd.DataFrame(rows).drop_duplicates("slide_id")
    return df


def read_patch(slide, x: int, y: int, level: int, size: int) -> Optional[np.ndarray]:
    try:
        region = slide.read_region((x, y), level, (size, size)).convert("RGB")
        return np.array(region)
    except Exception:
        return None

def tile_one_slide(
    slide_path: Path,
    out_dir: Path,
    level: int,
    patch_size: int,
    stride: int,
    tissue_thresh: float,
    patch_cap: int,
    save_thumb: bool,
    jpeg_quality: int,
    macenko_target: Optional[MacenkoTarget]
) -> Dict[str, object]:
    t0 = time.time()
    sl = openslide.OpenSlide(str(slide_path))
    lvl_w, lvl_h = sl.level_dimensions[level]

    tiles_dir = out_dir / ("tiles_norm" if macenko_target is not None else "tiles")
    ensure_dir(tiles_dir)

    xs = list(range(0, max(0, lvl_w - patch_size + 1), stride))
    ys = list(range(0, max(0, lvl_h - patch_size + 1), stride))
    grid = [(x, y) for y in ys for x in xs]
    random.shuffle(grid)

    kept = 0
    pre_sum = np.zeros(3, dtype=np.float64)
    post_sum = np.zeros(3, dtype=np.float64)

    for (x, y) in grid:
        if kept >= patch_cap:
            break
        img = read_patch(sl, x, y, level, patch_size)
        if img is None:
            continue
        mask = tissue_mask_rgb(img, tissue_thresh)
        if mask.sum() == 0:
            continue

        pre_sum += img.mean(axis=(0,1))

        if macenko_target is not None:
            try:
                img = macenko_normalize(img, macenko_target)
            except Exception:
                pass

        post_sum += img.mean(axis=(0,1))

        fname = f"{slide_path.stem}_x{x}_y{y}_L{level}.jpg"
        fpath = tiles_dir / fname
        if cv2 is not None:
            cv2.imwrite(str(fpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        else:
            Image.fromarray(img).save(str(fpath), quality=jpeg_quality, subsampling=1)

        kept += 1

    if save_thumb:
        try:
            thumb = sl.get_thumbnail((1024, int(1024 * lvl_h / max(1, lvl_w)))).convert("RGB")
            Image.fromarray(np.array(thumb)).save(str(out_dir / f"{slide_path.stem}_thumb.jpg"), quality=90)
        except Exception:
            pass

    sl.close()
    dt = time.time() - t0
    pre_mean = (pre_sum / max(1, kept)).tolist()
    post_mean = (post_sum / max(1, kept)).tolist()
    return {"n_patches": kept, "seconds": dt, "pre_rgb_mean": pre_mean, "post_rgb_mean": post_mean}


# 1) Registry
log(f"Building registry from: {WSI_ROOT}")
reg = build_registry(WSI_ROOT)
reg_path = RUN_DIR / "registry.csv"
reg.to_csv(reg_path, index=False)
log(f"Slides found: {len(reg):,} → {reg_path}")

# 2) Macenko target (disabled by default for speed)
macenko_target = None
if MACENKO_ON:
    if MACENKO_TARGET_IMAGE and Path(MACENKO_TARGET_IMAGE).exists():
        log("Fitting Macenko target from the provided target image...")
        target_rgb = np.array(Image.open(MACENKO_TARGET_IMAGE).convert("RGB"))
        macenko_target = macenko_fit_target_from_image(target_rgb)
        log("Macenko target: " + ("OK" if macenko_target is not None else "FAILED (skipping normalization)"))
    else:
        log("MACENKO_ON=True but no target image provided; skipping (set MACENKO_TARGET_IMAGE if desired).")
        macenko_target = None
else:
    log("Macenko normalization is OFF for this run (faster).")

# 3) Save run config
run_cfg = {
    "RUN_ID": RUN_ID,
    "params": {
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "tissue_thresh": TISSUE_THRESH,
        "patch_cap": PATCH_CAP,
        "level": LEVEL,
        "macenko_on": MACENKO_ON and (macenko_target is not None),
        "jpeg_quality": JPEG_QUALITY,
        "num_workers": NUM_WORKERS,
        "save_thumb": SAVE_THUMB
    },
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "wsi_root": str(WSI_ROOT),
        "run_dir": str(RUN_DIR),
        "results_dir": str(RESULTS_DIR),
        "fig_dir": str(FIG_DIR)
    }
}
(RUN_DIR / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

# 4) Process slides (parallel, continues on errors)
log("Starting tiling (optimized). New run folder; prior outputs untouched.")

def process_row(row) -> Dict[str, object]:
    slide_path = Path(row["slide_path"])
    slide_id   = row["slide_id"]
    slide_dir  = RUN_DIR / slide_id
    ensure_dir(slide_dir)
    info = {}
    try:
        info = tile_one_slide(
            slide_path=slide_path,
            out_dir=slide_dir,
            level=LEVEL,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            tissue_thresh=TISSUE_THRESH,
            patch_cap=PATCH_CAP,
            save_thumb=SAVE_THUMB,
            jpeg_quality=JPEG_QUALITY,
            macenko_target=macenko_target if MACENKO_ON else None
        )
        log(f"OK {slide_id} patches={info['n_patches']} time_sec={info['seconds']:.1f}")
    except Exception as e:
        log(f"FAIL {slide_id} err={e}")
        info = {"n_patches": 0, "seconds": 0.0, "pre_rgb_mean": [0,0,0], "post_rgb_mean": [0,0,0]}
    return {
        "slide_id": slide_id,
        "patient_id": row["patient_id"],
        "cancer_type": row["cancer_type"],
        **info
    }

results: List[Dict[str, object]] = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futs = [ex.submit(process_row, r) for _, r in reg.iterrows()]
    for i, fu in enumerate(as_completed(futs), 1):
        try:
            results.append(fu.result())
            if i % 200 == 0:
                log(f"…progress: {i}/{len(futs)} slides")
        except Exception as e:
            log(f"Worker error: {e}")

summary = pd.DataFrame(results).sort_values("slide_id")
summary_path = RUN_DIR / "summary.csv"
summary.to_csv(summary_path, index=False)
log(f"Saved summary: {summary_path}")

# 5) Diagnostics & Acceptance Gate
TOTAL_SLIDES = len(reg)
DONE_SLIDES  = summary.shape[0]
COVERED      = int((summary["n_patches"] >= MIN_PATCHES_PER_SLIDE).sum())
COVERAGE     = COVERED / max(1, TOTAL_SLIDES)
CAP_HITS     = int((summary["n_patches"] >= int(PATCH_CAP * 0.95)).sum())
CAP_RATE     = CAP_HITS / max(1, TOTAL_SLIDES)
TOTAL_PATCHES= int(summary["n_patches"].fillna(0).sum())
TOTAL_SECS   = float(summary["seconds"].fillna(0).sum())
THROUGHPUT   = TOTAL_SLIDES / (TOTAL_SECS/60.0) if TOTAL_SECS > 0 else 0.0

# Histogram of patches/slide
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7,4.5))
summary["n_patches"].fillna(0).astype(int).clip(0, PATCH_CAP).hist(bins=40)
plt.title(f"Patches per slide (cap={PATCH_CAP}, stride={STRIDE})")
plt.xlabel("n_patches")
plt.ylabel("count")
plt.tight_layout()
patch_hist_path = FIG_DIR / "patches_per_slide_hist.png"
plt.savefig(patch_hist_path, dpi=150)
plt.close()

# Save low coverage and cap-hit lists
low_cov = summary[summary["n_patches"] < MIN_PATCHES_PER_SLIDE][["slide_id","cancer_type","n_patches","seconds"]]
cap_hit = summary[summary["n_patches"] >= int(PATCH_CAP * 0.95)][["slide_id","cancer_type","n_patches","seconds"]]
low_cov_path = RESULTS_DIR / "low_coverage_slides.csv"
cap_hit_path = RESULTS_DIR / "cap_hit_slides.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
low_cov.to_csv(low_cov_path, index=False)
cap_hit.to_csv(cap_hit_path, index=False)

# Acceptance decision
gate_pass = True
notes = []

if COVERAGE < COVERAGE_MIN:
    gate_pass = False
    notes.append(f"Coverage too low: {COVERAGE:.2%} (min {COVERAGE_MIN:.0%}). "
                 f"Consider STRIDE=256 for more patches or lowering MIN_PATCHES_PER_SLIDE to 500.")

if CAP_RATE > CAP_HIT_MAX:
    gate_pass = False
    notes.append(f"Too many slides hit cap: {CAP_RATE:.2%} (max {CAP_HIT_MAX:.0%}). "
                 "Consider STRIDE=512 or raise PATCH_CAP slightly (e.g., 2400).")

gate = {
    "run_id": RUN_ID,
    "totals": {
        "slides_total": TOTAL_SLIDES,
        "slides_processed": DONE_SLIDES,
        "patches_total": TOTAL_PATCHES,
        "seconds_total": TOTAL_SECS,
        "throughput_slides_per_min": round(THROUGHPUT, 2)
    },
    "coverage": {
        "min_patches_per_slide": MIN_PATCHES_PER_SLIDE,
        "slides_meeting_min": COVERED,
        "coverage_rate": round(COVERAGE, 4),
        "low_coverage_list_csv": str(low_cov_path)
    },
    "cap": {
        "patch_cap": PATCH_CAP,
        "slides_hitting_cap": CAP_HITS,
        "cap_rate": round(CAP_RATE, 4),
        "cap_hit_list_csv": str(cap_hit_path)
    },
    "stain": None,  # Macenko off
    "figures": {"patch_hist": str(patch_hist_path)},
    "params": {
        "patch_size": PATCH_SIZE, "stride": STRIDE, "tissue_thresh": TISSUE_THRESH,
        "patch_cap": PATCH_CAP, "level": LEVEL, "macenko_on": False,
        "jpeg_quality": JPEG_QUALITY, "num_workers": NUM_WORKERS, "save_thumb": SAVE_THUMB
    },
    "pass": gate_pass,
    "notes": notes
}

gate_path = RESULTS_DIR / "gate_report.json"
gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")

# Console summary
print("\n=== Diagnostic Gate — Preprocess (Optimized Relaunch) ===")
print(f"Run ID            : {RUN_ID}")
print(f"Slides (total)    : {TOTAL_SLIDES:,}")
print(f"Slides processed  : {DONE_SLIDES:,}")
print(f"Coverage (>= {MIN_PATCHES_PER_SLIDE}) : {COVERED:,}  ({COVERAGE:.2%})  → low list: {low_cov_path.name}")
print(f"Cap hits (>= {int(PATCH_CAP*0.95)})    : {CAP_HITS:,}  ({CAP_RATE:.2%}) → list: {cap_hit_path.name}")
print(f"Total patches     : {TOTAL_PATCHES:,}")
print(f"Total time (sec)  : {int(TOTAL_SECS):,}  | Throughput: {THROUGHPUT:.2f} slides/min")
print("\nGate verdict      :", "✅ PASS" if gate_pass else "❌ RETRY")
if notes:
    print("Notes/Fixes       :")
    for n in notes:
        print("  -", n)

print(f"\nArtifacts:")
print(f" - Registry       : {reg_path}")
print(f" - Summary        : {summary_path}")
print(f" - Histogram      : {patch_hist_path}")
print(f" - Gate report    : {gate_path}")
print(f" - Run dir        : {RUN_DIR}")


# SECTION 3: TILE COORDINATE EXTRACTION

#Script -02 (IHGAMP Tiler)

import os, sys, time, json, gc, re
from contextlib import contextmanager

# Tame CPU oversubscription for NumPy/OpenCV/OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


try:
    pass
except Exception as e:
    raise SystemExit("OpenSlide not available. Install openslide-python and ensure OpenSlide DLLs are on PATH.") from e

try:
    cv2.setNumThreads(1)
except Exception:
    pass

PROJECT_ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")
WSI_ROOT     = PROJECT_ROOT / "Histo slides 20k"   # your WSI root
RUN_NAME     = f"run_{datetime.now():%Y%m%d_%H%M%S}_L0_ps256_st512_cap2000_fastT"
RUN_DIR      = PROJECT_ROOT / "runs" / RUN_NAME

PATCH_SIZE   = 256
STRIDE       = 512
CAP_PER_WSI  = 2000
TISSUE_LEVEL = 3
MIN_TISSUE_FRAC = 0.30
EXTS = (".svs",".tif",".tiff",".ndpi",".mrxs",".scn",".bif",".svslide")

NUM_WORKERS  = 20      # thread workers
LOG_EVERY    = 10

SAVE_COORDS  = True    # must be True (we only write coords)
# EMBED_INLINE kept out here to keep this stage lightweight; we embed in Script-2.

(RUN_DIR / "coords").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
summary_csv = RUN_DIR / "summary.csv"
run_log     = RUN_DIR / "run.log"

def log_line(msg: str):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(ts + msg + "\n")

def find_all_wsis(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            files.append(p)
    return sorted(files)

def already_done_slides():
    """Parse all existing run.log files under runs/ to skip completed slides."""
    done = set()
    runs_root = PROJECT_ROOT / "runs"
    for log in runs_root.rglob("run.log"):
        try:
            for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re.search(r"OK\s+([^\s]+)\s+patches=\d+", ln)
                if m:
                    done.add(m.group(1))
        except Exception:
            continue
    return done

def slide_id_from_path(p: Path) -> str:
    return p.stem

def level_for_mask(slide: "openslide.OpenSlide", prefer_level=TISSUE_LEVEL):
    return min(max(0, prefer_level), slide.level_count - 1)

def read_thumbnail(slide, level):
    w, h = slide.level_dimensions[level]
    MAX_PIX = 6000 * 6000
    scale = 1.0
    if w * h > MAX_PIX:
        scale = (MAX_PIX / (w * h)) ** 0.5
        w = int(w * scale); h = int(h * scale)
    img = slide.get_thumbnail((w, h)).convert("RGB")
    return np.array(img)

def grid_coords(slide, mask_level, stride_l0, patch_size_l0, min_tissue_frac):
    lvl_down = float(slide.level_downsamples[mask_level])
    w0, h0 = slide.dimensions
    win = int(round(patch_size_l0 / lvl_down))
    step = max(1, int(round(stride_l0 / lvl_down)))

    thumb = read_thumbnail(slide, mask_level)
    m = tissue_mask_rgb(thumb)

    H, W = m.shape
    lvl_w, lvl_h = slide.level_dimensions[mask_level]
    sx = W / float(lvl_w); sy = H / float(lvl_h)

    coords = []
    tissue_thresh = int(min_tissue_frac * (win * win))
    for yy in range(0, H - win + 1, step):
        row = m[yy:yy+win, :]
        for xx in range(0, W - win + 1, step):
            sub = row[:, xx:xx+win]
            if int(np.count_nonzero(sub)) >= tissue_thresh:
                x0 = int(round((xx / sx) * lvl_down))
                y0 = int(round((yy / sy) * lvl_down))
                coords.append((x0, y0))
                if CAP_PER_WSI and len(coords) >= CAP_PER_WSI:
                    return coords
    return coords

def write_coords_csv(out_csv: Path, coords, patch_size):
    if not coords:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("x,y,w,h,level\n")
        return
    arr = np.array(coords, dtype=np.int64)
    df = pd.DataFrame(arr, columns=["x","y"])
    df["w"] = patch_size; df["h"] = patch_size; df["level"] = 0
    df.to_csv(out_csv, index=False)

@contextmanager
def timing():
    t0 = time.time()
    yield lambda: time.time() - t0

def process_one(slide_path: Path) -> dict:
    sid = slide_id_from_path(slide_path)
    out_csv = RUN_DIR / "coords" / f"{sid}.csv"
    if out_csv.exists():
        return {"slide_id": sid, "n": None, "secs": 0.0, "skipped": True}

    try:
        with timing() as elapsed:
            slide = openslide.OpenSlide(str(slide_path))
            lvl = level_for_mask(slide, TISSUE_LEVEL)
            coords = grid_coords(slide, lvl, STRIDE, PATCH_SIZE, MIN_TISSUE_FRAC)
            n = len(coords)
            write_coords_csv(out_csv, coords, PATCH_SIZE)
            sec = elapsed()
    except Exception as e:
        return {"slide_id": sid, "n": -1, "secs": 0.0, "error": repr(e), "path": str(slide_path)}
    finally:
        try:
            slide.close()
        except Exception:
            pass
        gc.collect()

    return {"slide_id": sid, "n": n, "secs": sec, "skipped": False, "path": str(slide_path)}

all_wsis = find_all_wsis(WSI_ROOT)
if not all_wsis:
    raise SystemExit(f"No WSIs found under: {WSI_ROOT}")

done_set = already_done_slides()
existing_coords = {p.stem for p in (RUN_DIR/"coords").glob("*.csv")}
skip_names = done_set | existing_coords
todo = [p for p in all_wsis if slide_id_from_path(p) not in skip_names]

log_line(f"Fast-tiler (threaded) start at {RUN_DIR}")
log_line(f"WSIs found: {len(all_wsis):,} | Already done across runs: {len(skip_names):,} | To process now: {len(todo):,}")

cfg = {
    "project_root": str(PROJECT_ROOT),
    "wsi_root": str(WSI_ROOT),
    "run_dir": str(RUN_DIR),
    "patch_size": PATCH_SIZE,
    "stride": STRIDE,
    "cap_per_wsi": CAP_PER_WSI,
    "tissue_level": TISSUE_LEVEL,
    "min_tissue_frac": MIN_TISSUE_FRAC,
    "num_workers_threads": NUM_WORKERS,
    "timestamp": datetime.now().isoformat(timespec="seconds")
}
(RUN_DIR/"config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

if not summary_csv.exists():
    pd.DataFrame(columns=["slide_id","seconds","n_patches","path"]).to_csv(summary_csv, index=False)

n_done = 0; n_err = 0; total = len(todo)
t_start = time.time()

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futs = {ex.submit(process_one, p): p for p in todo}
    for i, fut in enumerate(as_completed(futs), 1):
        res = fut.result()
        sid = res.get("slide_id")
        if "error" in res:
            n_err += 1
            log_line(f"ERROR {sid} {res['error']}")
        elif not res.get("skipped"):
            n_done += 1
            n   = int(res["n"])
            sec = float(res["secs"])
            pd.DataFrame([[sid, sec, n, res.get("path","")]],
                         columns=["slide_id","seconds","n_patches","path"]
                        ).to_csv(summary_csv, mode="a", header=False, index=False)
            log_line(f"OK {sid} patches={n} time_sec={sec:.1f}")

        if i % LOG_EVERY == 0 or i == total:
            try:
                df = pd.read_csv(summary_csv)
                done_count = len(df)
                avg_sec = df["seconds"].clip(lower=1).mean() if done_count else float("nan")
            except Exception:
                done_count = 0; avg_sec = float("nan")

            if done_count and avg_sec == avg_sec and avg_sec > 0:
                rem = max(0, total - done_count)
                eta_sec = rem * (avg_sec / max(1, NUM_WORKERS))
                eta_hr  = eta_sec/3600
                eta_time= datetime.now() + pd.to_timedelta(eta_sec, unit="s")
                log_line(f"…progress: {done_count}/{total} (this fast-run) | avg_s/slide={avg_sec:.1f} | workers={NUM_WORKERS} | ETA ~ {eta_hr:.1f}h ({eta_time:%Y-%m-%d %H:%M})")
            else:
                log_line(f"…progress: {done_count}/{total} (this fast-run) | computing average…")

tot_elapsed = time.time() - t_start
log_line(f"Fast-tiler (threaded) complete. Slides processed now: {n_done} | errors: {n_err} | wall-time: {tot_elapsed/3600:.1f}h")
print(f"FAST RUN (threaded) COMPLETE → {RUN_DIR}")
print(f"Processed now: {n_done} slides | Errors: {n_err} | See {run_log}")


# SECTION 4: FEATURE EXTRACTION (OpenCLIP ViT-B/16)

# IHGAMP Script-03 
import os, sys, json, time, math, gc, re, hashlib
from concurrent.futures import ThreadPoolExecutor
import torch, torch.nn as nn, torch.nn.functional as F

ROOT     = Path(r"D:\个人文件夹\Sanwal\DL_V2")
WSI_ROOT = ROOT / "Histo slides 20k"
RUN_NAME = f"run_{datetime.now():%Y%m%d_%H%M%S}_emb_openclip_vitb16_turbo"
RUN_DIR  = ROOT / "runs" / RUN_NAME
EMB_DIR  = RUN_DIR / "embeddings"
for p in [RUN_DIR, EMB_DIR]: p.mkdir(parents=True, exist_ok=True)
report_md = RUN_DIR/"REPORT.md"; status_js = RUN_DIR/"STATUS.json"; run_log = RUN_DIR/"run.log"

COORD_SOURCES = list((ROOT/"runs").glob("run_*"))
WSI_EXTS = (".svs",".tif",".tiff",".ndpi",".mrxs",".scn",".bif",".svslide")

BATCH_SIZE = 256              # safer starting point; auto-halves on OOM
USE_AMP    = True
SAVE_DTYPE = "float16"
TARGET_SZ  = 224              # CLIP image size
PREFETCH_THREADS = 2          # tile-read prefetch threads (Windows-safe)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # quiet HF cache warnings

def pip_install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

try:
    import open_clip
except Exception:
    pip_install("open_clip_torch")

# Speed on 4090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def index_wsis(root: Path):
    idx={}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in WSI_EXTS:
            idx[p.stem]=p
    return idx

def gather_coords():
    coord_map={}
    for rd in COORD_SOURCES:
        for c in rd.glob("coords/*.csv"):
            coord_map.setdefault(c.stem, c)
    return coord_map

def infer_site(slide_id: str):
    m = re.match(r"([A-Z0-9]+-[A-Z0-9]+)", slide_id)
    return m.group(1) if m else "UNKNOWN"

def read_region_rgb(slide, x, y, w, h, level=0):
    im = slide.read_region((int(x), int(y)), int(level), (int(w), int(h))).convert("RGB")
    return np.array(im, dtype=np.uint8)

model, _preprocess, _ = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='laion2b_s34b_b88k', device=DEV
)
model.eval()
FEAT_DIM = getattr(getattr(model, "visual", None), "output_dim", 512)

# Provenance hash of weights (deterministic)
def weights_sha256(mod: torch.nn.Module) -> str:
    h = hashlib.sha256()
    sd = mod.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode('utf-8')); h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()
WEIGHTS_HASH = weights_sha256(model)

# CLIP mean/std (OpenAI/OpenCLIP)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1,3,1,1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1,3,1,1)

@torch.inference_mode()
def preprocess_gpu_u8_to_clip_tensor(batch_imgs_np: list):
    """
    batch_imgs_np: list of uint8 HxWx3 arrays (256x256)
    Returns: torch.FloatTensor [N,3,224,224] normalized to CLIP mean/std on GPU
    """
    arr = np.stack(batch_imgs_np, axis=0)  # NxHxWx3 uint8
    t = torch.from_numpy(arr).to(DEV, non_blocking=True).permute(0,3,1,2).float() / 255.0
    # Resize to 224x224 (no crop; our tiles are square tissue crops already)
    t = F.interpolate(t, size=(TARGET_SZ, TARGET_SZ), mode="bilinear", align_corners=False)
    t = (t - CLIP_MEAN) / CLIP_STD
    return t

@torch.inference_mode()
def encode_clip(t_img):
    with torch.amp.autocast('cuda', enabled=USE_AMP):
        z = model.encode_image(t_img)          # NxD
    if z.ndim > 2:
        z = z.mean(dim=(2,3))
    return z.float()

wsi_idx   = index_wsis(WSI_ROOT)
coord_map = gather_coords()
todo = sorted([sid for sid,csv in coord_map.items() if sid in wsi_idx])
print(f"Slides with coords & WSI present: {len(todo):,}")

done = 0; failed = []; per_site = {}
t0_all = time.time()

def read_batch(slide, coords_df, a, b):
    imgs = []
    for k in range(a, b):
        x,y,w,h,lvl = coords_df.iloc[k][["x","y","w","h","level"]]
        imgs.append(read_region_rgb(slide, x, y, w, h, level=int(lvl)))
    return imgs

for si, sid in enumerate(todo, 1):
    out_npy = EMB_DIR/f"{sid}.npy"
    if out_npy.exists():
        done += 1
        continue

    csv = coord_map[sid]
    coords = pd.read_csv(csv)
    n = len(coords)

    # Log slide start so monitors show progress even before first "OK"
    log(f"START {sid} n_coords={n}")

    if n == 0:
        np.save(out_npy, np.zeros((0, FEAT_DIM), dtype=np.float16 if SAVE_DTYPE=="float16" else np.float32))
        log(f"OK {sid} patches=0 time_sec=0.0 out={out_npy.name}")
        done += 1
        continue

    # Open slide
    try:
        slide = openslide.OpenSlide(str(wsi_idx[sid]))
    except Exception as e:
        log(f"ERROR_OPEN {sid} {repr(e)}"); failed.append(sid); continue

    bs = BATCH_SIZE
    feats = []
    i = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=PREFETCH_THREADS) as ex:
            # schedule first batch
            j = min(n, i+bs)
            fut = ex.submit(read_batch, slide, coords, i, j)
            i = j

            while True:
                imgs = fut.result()  # wait for current
                # schedule next (if any) while we process current
                if i < n:
                    j = min(n, i+bs)
                    fut = ex.submit(read_batch, slide, coords, i, j)
                    i = j
                    has_next = True
                else:
                    fut = None
                    has_next = False

                # GPU preprocess + encode current
                t_img = preprocess_gpu_u8_to_clip_tensor(imgs)
                z = encode_clip(t_img)
                feats.append(z)

                # heartbeats every ~5 batches
                if len(feats) % 5 == 0:
                    done_batches = sum(f.shape[0] for f in feats)
                    log(f"PROGRESS {sid} done_tiles={done_batches}/{n}")

                if not has_next:
                    break

        emb = torch.cat(feats, dim=0).detach().cpu().numpy()
        if SAVE_DTYPE == "float16":
            emb = emb.astype(np.float16, copy=False)
        np.save(out_npy, emb)
        sec = time.time()-t0
        log(f"OK {sid} patches={n} time_sec={sec:.1f} out={out_npy.name}")

        site = infer_site(sid)
        vmean = float(np.linalg.norm(emb.astype(np.float32), axis=1).mean()) if emb.size else 0.0
        s = per_site.setdefault(site, {"slides":0,"norm_sum":0.0,"tiles":0})
        s["slides"] += 1; s["norm_sum"] += vmean; s["tiles"] += n

        done += 1

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if bs <= 32:
            log(f"ERROR_OOM {sid} at BATCH={bs}"); failed.append(sid)
        else:
            # restart slide with smaller batch
            bs = max(32, bs//2)
            log(f"RETRY_OOM {sid} new_batch={bs}")
            feats=[]; i=0; t0=time.time()
            try:
                slide.close()
                slide = openslide.OpenSlide(str(wsi_idx[sid]))
                # simple re-run without prefetch nesting to keep code compact
                while i < n:
                    j = min(n, i+bs)
                    imgs = read_batch(slide, coords, i, j)
                    t_img = preprocess_gpu_u8_to_clip_tensor(imgs)
                    z = encode_clip(t_img)
                    feats.append(z); i = j
                emb = torch.cat(feats, dim=0).detach().cpu().numpy()
                if SAVE_DTYPE == "float16":
                    emb = emb.astype(np.float16, copy=False)
                np.save(out_npy, emb)
                sec = time.time()-t0
                log(f"OK {sid} patches={n} time_sec={sec:.1f} out={out_npy.name}")
                site = infer_site(sid)
                vmean = float(np.linalg.norm(emb.astype(np.float32), axis=1).mean()) if emb.size else 0.0
                s = per_site.setdefault(site, {"slides":0,"norm_sum":0.0,"tiles":0})
                s["slides"] += 1; s["norm_sum"] += vmean; s["tiles"] += n
                done += 1
            except Exception as e:
                log(f"ERROR_REDO {sid} {repr(e)}"); failed.append(sid)

    except Exception as e:
        log(f"ERROR {sid} {repr(e)}"); failed.append(sid)

    finally:
        try: slide.close()
        except: pass
        gc.collect(); torch.cuda.empty_cache()

t_all = time.time()-time.time() + t0_all  # correct elapsed

files = list(EMB_DIR.glob("*.npy"))
nan_slides=[]; norms=[]
for f in files:
    arr = np.load(f, mmap_mode="r")
    if arr.size>0:
        if not np.isfinite(arr).all(): nan_slides.append(f.stem)
        norms.append(float(np.linalg.norm(arr.astype(np.float32), axis=1).mean()))
    else:
        norms.append(0.0)

slides_h   = done / max(1.0, (time.time()-t0_all)/3600.0)
tiles_total= sum(s["tiles"] for s in per_site.values())
tiles_s    = tiles_total / max(1.0, (time.time()-t0_all))

# Save per-site summary
rows=[]
for site, s in per_site.items():
    rows.append({"site":site, "slides":s["slides"], "mean_norm":s["norm_sum"]/max(1,s["slides"]), "tiles":s["tiles"]})
pd.DataFrame(rows).to_parquet(RUN_DIR/"emb_stats.parquet", index=False)

rep=[]
rep.append(f"# IHGAMP Embeddings Report — {RUN_DIR.name}")
rep.append(f"- Backbone: **OpenCLIP ViT-B/16 (laion2b_s34b_b88k)** | FEAT_DIM={FEAT_DIM} | AMP={USE_AMP} | TF32=on")
rep.append(f"- Preprocess: **GPU resize→224 (no crop) + CLIP mean/std**")
rep.append(f"- Weights SHA256: `{WEIGHTS_HASH}`")
rep.append(f"- Slides embedded: **{done:,}**  | Failed: **{len(failed)}**  | Elapsed: **{(time.time()-t0_all)/3600:.1f} h**")
rep.append(f"- Throughput: **{slides_h:.1f} slides/h**, **{tiles_s:.0f} tiles/s** (approx)")
rep.append(f"- NaN slides: **{len(nan_slides)}**")
rep.append("\n## Per-site summary")
if rows:
    for r in sorted(rows, key=lambda x: x["slides"], reverse=True):
        rep.append(f"- {r['site']}: slides={int(r['slides'])}, mean_norm={r['mean_norm']:.3f}, tiles={int(r['tiles'])}")
else:
    rep.append("- (no stats)")
report_md.write_text("\n".join(rep), encoding="utf-8")

status = {
    "run_id": RUN_DIR.name,
    "backbone": "openclip_vitb16_laion2b_s34b_b88k",
    "feat_dim": int(FEAT_DIM),
    "amp": USE_AMP,
    "tf32": True,
    "preprocess": "gpu_resize_224_clip_meanstd",
    "weights_sha256": WEIGHTS_HASH,
    "slides_embedded": int(done),
    "failed": failed,
    "elapsed_sec": float(time.time()-t0_all),
    "slides_per_hour": float(slides_h),
    "tiles_per_second": float(tiles_s),
}
status_js.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(f"Wrote:\n  {report_md}\n  {status_js}\n  {RUN_DIR/'emb_stats.parquet'}")
print(f"Failed slides (first 10): {failed[:10]}")


# SECTION 5: COHORT LABELS (HRD, IFNG6, ANGIO)

# Script-04 — Cohort labels (IFNG6, ANGIO, HRD) from UCSC Xena

import os, sys, json, time, math, gzip, re, io, shutil, textwrap, random, warnings, hashlib

PROJECT_ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs"
ART_DIR      = PROJECT_ROOT / "artifacts"
RES_DIR      = PROJECT_ROOT / "results"
LABEL_DIR    = ART_DIR / "labels"
FIG_DIR      = RES_DIR / "labels" / "figs"
for p in [ART_DIR, RES_DIR, LABEL_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

WSI_ROOT_HINT = PROJECT_ROOT / "Histo slides 20k"  # only for cohort name inference if needed

# Xena hubs & dataset (Toil recompute RNA-seq TPM; robust cross-cohort)
XENA_HUBS  = [
    "https://toil.xenahubs.net",      # Toil recompute (TcgaTargetGtex_rsem_gene_tpm)
    "https://gdc.xenahubs.net",       # GDC pan-cancer hub
    "https://pancanatlas.xenahubs.net"
]
XENA_RNA_DATASET = "TcgaTargetGtex_rsem_gene_tpm"  # log2(TPM+0.001)
TIMEOUT_SEC = 30

# Gene panels
IFNG6 = ["IFNG","STAT1","IDO1","CXCL9","CXCL10","HLA-DRA"]  # Ayers-style compact set
ANGIO = ["VEGFA","KDR","FLT1","ANGPT2","TEK","ENG","PECAM1","VWF","COL4A1","COL4A2",
         "MMP2","MMP9","HIF1A","PDGFB","PDGFRB","ITGAV","ITGB3","ICAM1","SELE","PGF","ELN"]
HRR_CORE = ["BRCA1","BRCA2","PALB2","RAD51","RAD51C","RAD51D","BARD1","BRIP1","ATM","ATR",
            "CHEK1","CHEK2","XRCC2","XRCC3","MRE11","RAD50","NBN","FANCA","FANCD2","FANCM","RPA1","RFC2","BLM"]

for pkg in ["pandas", "numpy", "pyarrow", "matplotlib", "xenaPython"]:
    pip_install(pkg)

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import xenaPython as xena

plt.rcParams.update({"figure.figsize": (7.5, 4.8), "figure.dpi": 120})
warnings.filterwarnings("ignore")


def find_latest_registry(runs_dir: Path) -> Path|None:
    candidates = []
    for d in runs_dir.glob("run_*"):
        reg = d / "registry.csv"
        if reg.exists():
            candidates.append((reg.stat().st_mtime, reg))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

registry_path = find_latest_registry(RUNS_DIR)
if registry_path is None:
    print("WARNING: No registry.csv found under runs/. I will infer from filenames.")
    slide_rows = []
    for slide in (WSI_ROOT_HINT if WSI_ROOT_HINT.exists() else PROJECT_ROOT).rglob("*.svs"):
        sid = slide.stem
        m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", sid, re.I)
        patient = m.group(1).upper() if m else None
        cancer = slide.parent.name.upper()
        slide_rows.append({"slide_id": sid, "slide_path": str(slide), "patient": patient, "cancer": cancer})
    slides_df = pd.DataFrame(slide_rows)
else:
    slides_df = pd.read_csv(registry_path)
    # best-effort harmonization
    if "slide_id" not in slides_df.columns:
        # derive from path
        slides_df["slide_id"] = slides_df.get("wsi_path", slides_df.get("path", "")).apply(lambda s: Path(str(s)).stem)
    if "slide_path" not in slides_df.columns:
        slides_df["slide_path"] = slides_df.get("wsi_path", slides_df.get("path", ""))
    if "patient" not in slides_df.columns:
        slides_df["patient"] = slides_df["slide_id"].str.extract(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", expand=False).str.upper()
    if "cancer" not in slides_df.columns:
        slides_df["cancer"] = slides_df["slide_path"].apply(lambda p: Path(str(p)).parent.name.upper())

# basic cleaning
slides_df = slides_df.dropna(subset=["patient"]).copy()
slides_df["patient"] = slides_df["patient"].str.upper().str.slice(0, 12)
slides_df["cancer"]  = slides_df["cancer"].str.upper()
slides_df = slides_df.drop_duplicates(subset=["slide_id"])
print(f"Slides in registry: {len(slides_df):,} | patients: {slides_df['patient'].nunique():,}")

# patient → slides list
sl_by_pt = slides_df.groupby("patient")["slide_id"].apply(list).rename("slides")
pt_cancer = slides_df.groupby("patient")["cancer"].agg(lambda x: x.mode().iat[0] if len(x)>0 else "NA")


def try_hub_call(func, *args, **kwargs):
    for hub in XENA_HUBS:
        try:
            return func(hub, *args, **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"All Xena hubs failed for {func.__name__}")

def fetch_rna_matrix(genes: list[str]) -> pd.DataFrame:
    """Return DataFrame [sample x gene] from Xena Toil dataset (log2(TPM+0.001))."""
    hub = None
    for h in XENA_HUBS:
        try:
            # Check dataset availability by requesting a few samples
            _ = xena.dataset_samples(h, XENA_RNA_DATASET, 5)
            hub = h
            break
        except Exception:
            continue
    if hub is None:
        print("WARNING: RNA dataset not reachable. Returning empty frame.")
        return pd.DataFrame()

    # all samples (can be ~20k; ok)
    samples = xena.dataset_samples(hub, XENA_RNA_DATASET, None)
    samples = [s for s in samples if s.startswith("TCGA-")]  # keep TCGA only
    # Query gene-by-gene to avoid probeMap ambiguity
    out = {}
    for g in genes:
        try:
            _, vals = xena.dataset_probe_values(hub, XENA_RNA_DATASET, samples, [g])
            out[g] = pd.Series(vals[0], index=samples, dtype="float32")
        except Exception:
            # gene may be absent; fill NaN
            out[g] = pd.Series([np.nan]*len(samples), index=samples, dtype="float32")
    mat = pd.DataFrame(out)
    mat.index.name = "sample"
    return mat

def collapse_to_patient(mat: pd.DataFrame) -> pd.DataFrame:
    """Collapse TCGA sample → patient (first 12 chars), prefer tumor (01) if multiple."""
    if mat.empty:
        return mat
    df = mat.copy()
    df["patient"] = df.index.str.slice(0,12)
    # prefer primary tumor samples ending with "-01"
    df["is_primary"] = df.index.str.contains(r"-01$", regex=True)
    df = df.sort_values("is_primary", ascending=False)
    df = df.drop(columns=["is_primary"])
    return df.groupby("patient").first()


GENE_UNION = sorted(set(IFNG6 + ANGIO + HRR_CORE))
rna = fetch_rna_matrix(GENE_UNION)
pt_expr = collapse_to_patient(rna)

# z-score helper (per-gene across patients)
def zscore(df):
    return (df - df.mean(axis=0, skipna=True)) / (df.std(axis=0, ddof=0, skipna=True) + 1e-8)

scores = pd.DataFrame(index=pt_expr.index)
if not pt_expr.empty:
    expr_z = zscore(pt_expr)

    # IFN-γ 6-gene
    avail_ifng = [g for g in IFNG6 if g in expr_z.columns]
    scores["IFNG6"] = expr_z[avail_ifng].mean(axis=1) if avail_ifng else np.nan

    # Angiogenesis
    avail_ang = [g for g in ANGIO if g in expr_z.columns]
    scores["ANGIO"] = expr_z[avail_ang].mean(axis=1) if avail_ang else np.nan

    # HRD surrogate (HRR down)
    avail_hrr = [g for g in HRR_CORE if g in expr_z.columns]
    scores["HRD_expr"] = (-expr_z[avail_hrr].mean(axis=1)) if avail_hrr else np.nan
else:
    scores = pd.DataFrame(columns=["IFNG6","ANGIO","HRD_expr"])

# Non-fatal; if found, adds 'HRD_scar' column
def try_fetch_explicit_hrd() -> pd.Series|None:
    cand_terms = ["hrd", "scar", "homologous", "recomb"]
    for hub in XENA_HUBS:
        try:
            datasets = xena.datasets(hub)  # may not exist in older xenaPython; guard below
        except Exception:
            # fallback: nothing we can do without listing
            continue
        # Filter likely datasets
        d_hits = [d for d in datasets if any(t in d.lower() for t in cand_terms)]
        for ds in d_hits:
            try:
                fields = xena.dataset_fields(hub, ds)
            except Exception:
                continue
            fld_hits = [f for f in fields if any(t in f.lower() for t in cand_terms)]
            if not fld_hits:
                continue
            # pull first matching field
            fld = fld_hits[0]
            samples = xena.dataset_samples(hub, ds, None)
            samples = [s for s in samples if s.startswith("TCGA-")]
            pos, vals = xena.dataset_probe_values(hub, ds, samples, [fld])
            ser = pd.Series(vals[0], index=samples, dtype="float32")
            ser.index = ser.index.str.slice(0,12)
            return ser.groupby(level=0).mean()
    return None

HRD_scar = None
try:
    # Only attempt if xenaPython exposes 'datasets'
    if hasattr(xena, "datasets"):
        HRD_scar = try_fetch_explicit_hrd()
except Exception:
    HRD_scar = None

if HRD_scar is not None and len(HRD_scar) > 0:
    scores = scores.join(HRD_scar.rename("HRD_scar"), how="left")
    # Normalize directions: higher = more deficient
    # If HRD_scar correlates negatively with HRD_expr (rare), flip sign.
    try:
        cc = pd.concat([scores["HRD_expr"], scores["HRD_scar"]], axis=1).dropna()
        if not cc.empty and cc.corr().iloc[0,1] < 0:
            scores["HRD_scar"] = -scores["HRD_scar"]
    except Exception:
        pass


labels = pd.DataFrame(index=sorted(set(scores.index).intersection(sl_by_pt.index)))
labels = labels.join(scores, how="left")
labels = labels.join(sl_by_pt, how="left")
labels = labels.join(pt_cancer.rename("cancer"), how="left")

labels.reset_index(inplace=True)
labels = labels.rename(columns={"index":"patient"})

# simple train/val/test split by patient, stratified-ish by cancer
rng = np.random.RandomState(42)
labels["_rand"] = rng.rand(len(labels))
labels["split"] = (
    labels.groupby("cancer")["_rand"]
    .transform(lambda s: pd.qcut(s.rank(method="first"), q=[0,0.8,0.9,1.0], labels=["train","val","test"]))
)
labels = labels.drop(columns=["_rand"])

# Save
labels.to_parquet(LABEL_DIR / "labels.parquet", index=False)
labels.to_csv(LABEL_DIR / "labels.csv", index=False)

# README
readme = f"""# IHGAMP Labels (Script-04)

**Dataset**: UCSC Xena Toil RNA-seq TPM (log2(TPM+0.001)): `{XENA_RNA_DATASET}`  
**Hubs tried**: {", ".join(XENA_HUBS)}

**Targets**
- IFNG6: mean z of {", ".join(IFNG6)} (Ayers-style compact IFN-γ signature)
- ANGIO: mean z of {len(ANGIO)} hallmark-like angiogenesis genes
- HRD_expr: −mean z of core HRR genes ({len(HRR_CORE)}). Higher = *more deficient*.
- HRD_scar (optional): discovered on hubs if available; direction harmonized to higher=worse.

**Notes**
- No SciPy/lifelines imports (Windows-safe).
- Patient ID = first 12 chars of TCGA barcode.
- Slides mapped via latest `runs/*/registry.csv` (or filename fallback).
- Splits: 80/10/10 by patient, per cancer.

Files:
- `labels.parquet` / `labels.csv`
- `../results/labels/figs/*.png` quick sanity plots
"""
(LABEL_DIR / "README.md").write_text(readme, encoding="utf-8")


DIAG = []

def safe_hist(series, title, fname):
    s = pd.Series(series).dropna()
    if s.empty: 
        return
    plt.figure()
    plt.hist(s.values, bins=40, edgecolor="black")
    plt.title(title)
    plt.xlabel("score (z)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname)
    plt.close()

# coverage
cov = {
    "n_patients_with_slides": int(sl_by_pt.index.nunique()),
    "n_patients_labeled": int(labels["patient"].nunique()),
    "IFNG6_nonNA": int(labels["IFNG6"].notna().sum()),
    "ANGIO_nonNA": int(labels["ANGIO"].notna().sum()),
    "HRD_expr_nonNA": int(labels["HRD_expr"].notna().sum()),
    "HRD_scar_present": bool("HRD_scar" in labels.columns and labels["HRD_scar"].notna().any()),
}

# per-cancer counts
counts = labels.groupby("cancer")["patient"].nunique().sort_values(ascending=False)
DIAG.append("## Coverage\n" + json.dumps(cov, indent=2))
DIAG.append("\n## Patients per cancer (top 15)\n" + counts.head(15).to_string())

# distributions
safe_hist(labels["IFNG6"], "IFNG6 distribution (z)", "ifng6_hist.png")
safe_hist(labels["ANGIO"],  "ANGIO distribution (z)", "angio_hist.png")
safe_hist(labels["HRD_expr"], "HRD_expr surrogate (−mean HRR z)", "hrd_expr_hist.png")
if "HRD_scar" in labels.columns:
    safe_hist(labels["HRD_scar"], "HRD_scar (as discovered)", "hrd_scar_hist.png")

# cross-axis correlations (Pearson, numpy only)
def pearson(a, b):
    x = pd.Series(a).astype("float64")
    y = pd.Series(b).astype("float64")
    m = x.notna() & y.notna()
    if m.sum() < 3: 
        return np.nan
    x = x[m] - x[m].mean()
    y = y[m] - y[m].mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())) + 1e-12
    return float((x*y).sum() / denom)

corrs = {}
for a,b in [("IFNG6","ANGIO"), ("IFNG6","HRD_expr"), ("ANGIO","HRD_expr")]:
    corrs[f"{a}~{b}"] = pearson(labels[a], labels[b])

DIAG.append("\n## Cross-axis Pearson correlations\n" + json.dumps(corrs, indent=2))

# write DIAGNOSTICS
(RES_DIR / "labels").mkdir(parents=True, exist_ok=True)
(RES_DIR / "labels" / "DIAGNOSTICS.md").write_text("\n\n".join(DIAG), encoding="utf-8")

print("=== Script-04 COMPLETE ===")
print(f"Wrote: {LABEL_DIR/'labels.parquet'}  ({len(labels):,} patients)")
print(f"      {LABEL_DIR/'labels.csv'}")
print(f"      {LABEL_DIR/'README.md'}")
print(f"      {RES_DIR/'labels'/'DIAGNOSTICS.md'}")
print(f"      figs → {FIG_DIR}")


# SECTION 6: FEATURE SANITIZATION & QUALITY CONTROL

# Script-05 sanitize & resume

import os, sys, json, time

# Reuse globals from Script-05; if missing, set minimal defaults
if "ROOT" not in globals(): ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")
if "ART"  not in globals(): ART  = ROOT / "artifacts"
if "RES"  not in globals(): RES  = ROOT / "results" / "models"
if "EMB_RUN_ID" not in globals(): 
    # find newest emb run folder
    runs = sorted((ROOT / "runs").glob("run_*emb*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs: raise SystemExit("No embeddings run found. Run Script-03/05 first.")
    EMB_RUN_ID = runs[0].name
if "OUT_DIR" not in globals():
    OUT_DIR = (RES / EMB_RUN_ID); OUT_DIR.mkdir(parents=True, exist_ok=True)
if "TARGETS" not in globals(): TARGETS = ["IFNG6","ANGIO","HRD"]
if "SEED" not in globals(): SEED = 1337
if "N_FOLDS" not in globals(): N_FOLDS = 5
if "TOP_FRAC" not in globals(): TOP_FRAC = 0.20
if "GOALS" not in globals(): GOALS = {"IFNG6":0.30, "ANGIO":0.35, "HRD":0.25}

CACHE = ART / "embeddings"
cand = sorted(CACHE.glob(f"patient_means_{EMB_RUN_ID}*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cand: 
    raise SystemExit("patient_means parquet not found. Run Script-05 up to Step 2 once.")
patient_means_p = cand[0]
Xp_raw = pd.read_parquet(patient_means_p)

def load_labels(root: Path) -> pd.DataFrame:
    for p in [root / "artifacts" / "labels" / "labels.parquet",
              root / "artifacts" / "labels" / "labels.csv"]:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if "patient" in df.columns:
                df["patient"] = df["patient"].astype(str).str.upper().str.slice(0,12)
                df = df.drop_duplicates("patient").set_index("patient")
            else:
                df.index = df.index.astype(str).str.upper().str.slice(0,12)
            return df
    raise SystemExit("No labels file present at artifacts/labels/. Run Script-04R3f.")

labels = load_labels(ROOT)

def sanitize_features(X: pd.DataFrame, min_non_na_col_ratio: float = 0.98, min_var: float = 1e-12):
    X0 = X.copy()
    # Count pre conditions
    pre = {
        "rows": int(X0.shape[0]),
        "cols": int(X0.shape[1]),
        "rows_all_nan": int(X0.isna().all(axis=1).sum()),
        "cells_nan": int(X0.isna().sum().sum()),
        "cells_inf": int(np.isinf(X0.to_numpy(dtype=np.float64)).sum()),
    }
    # Replace inf with NaN
    X1 = X0.replace([np.inf, -np.inf], np.nan)
    # Drop rows that are completely NaN (should be rare)
    all_nan_rows = X1.isna().all(axis=1)
    X2 = X1.loc[~all_nan_rows].copy()
    dropped_rows = int(all_nan_rows.sum())

    # Drop columns with excessive missingness
    col_non_na = X2.notna().mean()
    keep_cols = col_non_na[col_non_na >= min_non_na_col_ratio].index
    drop_cols_missing = sorted(set(X2.columns) - set(keep_cols))
    X3 = X2[keep_cols].copy()

    # Impute remaining missing with column means
    col_means = X3.mean(axis=0)
    na_before = int(X3.isna().sum().sum())
    X4 = X3.fillna(col_means)
    na_after = int(X4.isna().sum().sum())

    # Drop near-constant columns
    var = X4.var(axis=0)
    keep_cols2 = var[var > min_var].index
    drop_cols_const = sorted(set(X4.columns) - set(keep_cols2))
    X5 = X4[keep_cols2].astype(np.float32)

    report = {
        "pre": pre,
        "dropped_rows_all_nan": dropped_rows,
        "dropped_cols_missing": len(drop_cols_missing),
        "dropped_cols_const": len(drop_cols_const),
        "imputed_cells": na_before - na_after,
        "final_shape": (int(X5.shape[0]), int(X5.shape[1])),
        "min_non_na_col_ratio": min_non_na_col_ratio,
        "min_var": min_var,
    }
    return X5, report, drop_cols_missing, drop_cols_const, X0.index[all_nan_rows].tolist()

print(f"[{ts()}] Sanitizing patient features to remove NaNs/inf…")
Xp_clean, rep, drop_missing, drop_const, dropped_row_ids = sanitize_features(Xp_raw)

common = Xp_clean.index.intersection(labels.index)
Xp2 = Xp_clean.loc[common].sort_index()
y_all = labels.loc[common, [t for t in TARGETS if t in labels.columns]].sort_index()

clean_name = ART / "embeddings" / f"patient_means_clean_{EMB_RUN_ID}.parquet"
Xp_clean.to_parquet(clean_name)
with open(OUT_DIR / "SANITIZATION_REPORT.json","w",encoding="utf-8") as f:
    json.dump({
        "patient_means_file": str(patient_means_p),
        "cleaned_patient_means_file": str(clean_name),
        "dropped_row_ids_all_nan": dropped_row_ids[:50],  # preview first 50
        "dropped_cols_missing_count": rep["dropped_cols_missing"],
        "dropped_cols_const_count": rep["dropped_cols_const"],
        "imputed_cells": rep["imputed_cells"],
        "final_shape": rep["final_shape"],
        "min_non_na_col_ratio": rep["min_non_na_col_ratio"],
        "min_var": rep["min_var"],
    }, f, indent=2)

print(f"    shape raw: {Xp_raw.shape} → clean: {Xp_clean.shape} | aligned patients: {len(common)}")
print(f"    dropped all-NaN patients: {len(dropped_row_ids)} | imputed cells: {rep['imputed_cells']}")
print(f"    dropped cols (missing): {rep['dropped_cols_missing']} | dropped cols (const): {rep['dropped_cols_const']}")

# If not present (you restarted the kernel), define a minimal version inline.

if 'train_and_eval_target' not in globals():
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import pearsonr, spearmanr
    import joblib
    try:
        _PLOTS_OK = True
    except Exception:
        _PLOTS_OK = False

    def safe_pearson(a,b):
        a, b = np.asarray(a), np.asarray(b)
        if np.isfinite(a).sum()<3 or np.isfinite(b).sum()<3: return np.nan
        try: return float(pearsonr(a,b)[0])
        except Exception: return np.nan
    def safe_spearman(a,b):
        a, b = np.asarray(a), np.asarray(b)
        if np.isfinite(a).sum()<3 or np.isfinite(b).sum()<3: return np.nan
        try: return float(spearmanr(a,b)[0])
        except Exception: return np.nan
    def quantile_mask(y: pd.Series, top_frac: float):
        thr = y.quantile(1.0 - top_frac)
        return (y >= thr).astype(int), float(thr)

    def train_and_eval_target(target: str, X: pd.DataFrame, y: pd.Series, outdir: Path) -> pd.DataFrame:
        out = (outdir / target); out.mkdir(parents=True, exist_ok=True)
        y = y.copy()
        msk = y.notna()
        X, y = X.loc[msk], y.loc[msk]

        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        idx = np.arange(len(y))
        folds = [(tr, te) for tr, te in splitter.split(idx)]
        ridge = Pipeline([("scaler", StandardScaler()), ("mdl", RidgeCV(alphas=np.array([0.1,0.3,1,3,10,30])))])
        hgb = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_depth=6, max_iter=500,
                                            l2_regularization=1e-3, random_state=SEED)

        fold_rows, all_true, all_pred = [], [], []
        for k, (tr, te) in enumerate(folds, 1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            ridge.fit(Xtr, ytr); yhat_r = ridge.predict(Xte)
            hgb.fit(Xtr, ytr);   yhat_h = hgb.predict(Xte)
            yhat = 0.5*yhat_r + 0.5*yhat_h

            r_p = safe_pearson(yte.values, yhat)
            r_s = safe_spearman(yte.values, yhat)
            ybin_tr, thr = quantile_mask(ytr, TOP_FRAC)
            ybin_te = (yte >= thr).astype(int)
            lr = LogisticRegression(max_iter=1000)
            lr.fit(ridge.predict(Xtr).reshape(-1,1), ybin_tr)
            prob = lr.predict_proba(ridge.predict(Xte).reshape(-1,1))[:,1]
            try:
                auc = roc_auc_score(ybin_te, prob); ap = average_precision_score(ybin_te, prob)
            except Exception:
                auc, ap = np.nan, np.nan

            fold_rows.append({"target":target,"fold":k,"n_te":len(yte),
                              "pearson_r":r_p,"spearman_r":r_s,"auc_top20":auc,"ap_top20":ap,
                              "ridge_alpha": getattr(ridge.named_steps["mdl"], "alpha_", np.nan)})
            all_true.append(yte.values); all_pred.append(yhat)
            joblib.dump(ridge, out / f"ridge_fold{k}.joblib"); joblib.dump(hgb, out / f"hgb_fold{k}.joblib")

        df_folds = pd.DataFrame(fold_rows)
        df_folds.to_csv(out / "metrics_fold.csv", index=False)
        df_cv = (df_folds.groupby("target")[["pearson_r","spearman_r","auc_top20","ap_top20"]]
                           .agg(["mean","std"]).reset_index())
        df_cv.to_csv(out / "metrics_cv.csv", index=False)
        return df_folds

print(f"[{ts()}] Re-running CV on sanitized features…")
avail = [t for t in TARGETS if t in y_all.columns]
all_fold_metrics = []
for tgt in avail:
    m = train_and_eval_target(tgt, Xp2, y_all[tgt], OUT_DIR)
    m["emb_run"] = EMB_RUN_ID
    all_fold_metrics.append(m)

df_all = pd.concat(all_fold_metrics, ignore_index=True)
df_all.to_csv(OUT_DIR / "ALL_targets_metrics_folds.csv", index=False)
df_cv = (df_all
         .groupby("target")[["pearson_r","spearman_r","auc_top20","ap_top20"]]
         .agg(["mean","std"])
         .reset_index())
df_cv.to_csv(OUT_DIR / "ALL_targets_metrics_cv.csv", index=False)

summary = []
for _, row in df_cv.iterrows():
    tgt = row[("target","")]
    r_mean = float(row[("pearson_r","mean")])
    goal = GOALS.get(tgt, 0.30)
    summary.append(f"- **{tgt}**: r={r_mean:.3f} | goal≥{goal:.2f} → **{'PASS' if r_mean>=goal else 'FLAG'}**")

with open(OUT_DIR / "REPORT.md","w",encoding="utf-8") as f:
    f.write(f"# IHGAMP Script-05 — Results (sanitized) — {EMB_RUN_ID}\n\n")
    f.write("## Cross-validated performance (mean±std)\n\n")
    f.write(df_cv.to_string(index=False))
    f.write("\n\n## Goal checks (Pearson r mean)\n\n")
    f.write("\n".join(summary))
    f.write("\n\n## Notes\n- Feature matrix sanitized to remove NaN/inf; see SANITIZATION_REPORT.json\n")

print(f"[{ts()}] DONE — sanitized training complete.")
print(f"  Results → {OUT_DIR}")
print(f"  Cleaning report → {OUT_DIR/'SANITIZATION_REPORT.json'}")


# SECTION 7: DDR GENE SIGNATURE CONSTRUCTION

# S06 — DDR repair:

import pandas as pd, numpy as np, re

ROOT    = Path(r"D:\个人文件夹\Sanwal\DL_V2").resolve()
LBL_DIR = ROOT / "artifacts" / "labels"
DDR_DIR = ROOT / r"artifacts\labels\external\DDR\TCGA_DDR_Data_Resources"

def load_first(*paths):
    for p in paths:
        if p.exists():
            return (pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)), p
    return None, None

def norm12(s):
    s = pd.Series(s, dtype=str).str.upper().str.strip()
    s = s.str.replace(r"[^A-Z0-9\-]", "", regex=True)
    return s.str.slice(0, 12)

def looks_like_tcga(x: str) -> bool:
    # accept patient- or sample-level TCGA barcodes
    return bool(re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", str(x).upper().strip()))

labels, p_lab = load_first(LBL_DIR/"labels.parquet", LBL_DIR/"labels.csv")
assert labels is not None, "Missing artifacts/labels/labels.(parquet/csv)"
labels["patient"] = norm12(labels["patient"])
labels = labels.drop_duplicates(subset=["patient"]).copy().set_index("patient")

ddr_scores = DDR_DIR/"DDRscores.tsv"   # matrix
ddr_rows   = DDR_DIR/"Scores.tsv"      # score names
ddr_cols   = DDR_DIR/"Samples.tsv"     # sample metadata (should contain barcodes)

assert ddr_scores.exists(), f"Missing {ddr_scores}"
assert ddr_rows.exists(),   f"Missing {ddr_rows}"
assert ddr_cols.exists(),   f"Missing {ddr_cols}"

S = pd.read_csv(ddr_scores, sep="\t", header=None, dtype=str)
row_meta = pd.read_csv(ddr_rows, sep="\t", header=None, dtype=str)
col_meta = pd.read_csv(ddr_cols, sep="\t", header=None, dtype=str)

# Orient check: expected shape (n_samples x n_scores) == (len(Samples.tsv) x len(Scores.tsv))
if not (S.shape[0] == len(col_meta) and S.shape[1] == len(row_meta)):
    raise SystemExit(f"Shape mismatch: DDRscores={S.shape}, Samples.tsv={col_meta.shape}, Scores.tsv={row_meta.shape}")

# Name columns with score names
row_names = row_meta.iloc[:,0].astype(str).tolist()
S.columns = row_names

# try each column; pick the one with highest share of TCGA-like entries
best_col, best_share = None, -1.0
for c in col_meta.columns:
    col = col_meta[c].astype(str)
    share = float(pd.Series(col).map(looks_like_tcga).mean())
    if share > best_share:
        best_share, best_col = share, c

if best_share < 0.5:
    # show a quick snapshot to help you choose manually
    print("[DDR] Could not find a barcode column (≥50% TCGA-like). Column shares:")
    for c in col_meta.columns:
        col = col_meta[c].astype(str)
        share = float(pd.Series(col).map(looks_like_tcga).mean())
        print(f"  - col{int(c)}: {share:.2%}")
    # show first few rows so you can eyeball
    print("\nHead of Samples.tsv (first 5 rows):")
    display(col_meta.head())
    raise SystemExit("Samples.tsv seems to contain project codes (ACC/BRCA/...). Provide a Samples.tsv with TCGA barcodes.")

# Use the detected column
samples_raw = col_meta[best_col].astype(str)
patients12  = norm12(samples_raw)
S.index     = patients12

print(f"[DDR] Using Samples.tsv column index {int(best_col)} as barcodes (TCGA-like share={best_share:.1%})")
print(f"[DDR] unique patients (12-char): {S.index.nunique():,}")

lut = {c.lower().replace("_",""): c for c in S.columns}
def pick(name): return lut.get(name.lower().replace("_",""))

col_hrd = pick("HRD_Score") or pick("HRDScore") or pick("HRDsum") or pick("HRD_Sum")
col_loh, col_tai, col_lst = pick("LOH"), pick("TAI"), pick("LST")

if col_hrd:
    hrd_numeric = pd.to_numeric(S[col_hrd], errors="coerce")
    hrd_source  = "HRD_Score"
elif all([col_loh, col_tai, col_lst]):
    hrd_numeric = (pd.to_numeric(S[col_loh], errors="coerce") +
                   pd.to_numeric(S[col_tai], errors="coerce") +
                   pd.to_numeric(S[col_lst], errors="coerce"))
    hrd_source  = "LOH+TAI+LST"
else:
    raise SystemExit(f"No HRD_Score or LOH/TAI/LST in DDR. Available (first 20): {list(S.columns)[:20]}")

geno = pd.DataFrame({"patient": S.index, "HRD_genomic": hrd_numeric}).dropna(subset=["patient"]).drop_duplicates(subset=["patient"])
geno = geno.set_index("patient")

print(f"[DDR] HRD_genomic non-null (raw): {int(geno['HRD_genomic'].notna().sum()):,} | source={hrd_source}")

common = labels.index.intersection(geno.index)
print(f"[JOIN] overlap labels↔DDR: {len(common):,}")

labels["HRD_genomic"] = pd.to_numeric(geno["HRD_genomic"].reindex(labels.index), errors="coerce")

# unified HRD = genomic → expr → scar → sig3
u = pd.Series(np.nan, index=labels.index, dtype=float)
for c in [c for c in ["HRD_genomic","HRD_expr","HRD_scar","HRD_sig3"] if c in labels.columns]:
    v = pd.to_numeric(labels[c], errors="coerce")
    take = u.isna() & v.notna()
    u.loc[take] = v.loc[take]
labels["HRD"] = u

print("\n=== After DDR repair ===")
print("Rows (patients):", len(labels))
print("Non-null HRD_genomic:", int(labels["HRD_genomic"].notna().sum()))
print("Non-null HRD       :", int(labels["HRD"].notna().sum()))
display(labels.loc[labels["HRD_genomic"].notna(), ["HRD_genomic"]].head(10))

labels.reset_index().to_parquet(LBL_DIR/"labels.parquet", index=False)
labels.reset_index().to_csv(LBL_DIR/"labels.csv", index=False)
print("\nWROTE →", LBL_DIR/"labels.parquet")
print("WROTE →", LBL_DIR/"labels.csv")


# SECTION 8: INTERNAL AUC CALIBRATION

# Script 7 AUC calibration 
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression

ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")

# try to reuse last paper lock; else auto-discover embeddings
plocks = sorted((ROOT/"results/paper_lock").glob("paperlock_*"), key=lambda p: p.stat().st_mtime, reverse=True)
if plocks and (plocks[0]/"summary.json").exists():
    SUM = json.loads((plocks[0]/"summary.json").read_text(encoding="utf-8"))
    EMB = Path(SUM["embeddings_file"])
    PCA_N = int(SUM.get("pca_n", 384))
    RIDGE_ALPHA = float(SUM.get("ridge_alpha", 30.0))
    TOP_THR = float(SUM.get("thr_train", 33.0))  # top-20% threshold on TRAIN HRD
else:
    # fallback discovery
    cand = sorted((ROOT/"artifacts/embeddings").glob("patient_means_*.parquet"), key=lambda p:p.stat().st_mtime, reverse=True)
    if not cand: 
        raise FileNotFoundError("No embeddings parquet found under artifacts/embeddings")
    EMB = cand[0]
    PCA_N = 384
    RIDGE_ALPHA = 30.0
    TOP_THR = 33.0

LABELS = ROOT/"artifacts/labels/labels.parquet"

print(f"[emb] {EMB.name}")
print(f"[cfg] PCA={PCA_N}  Ridge alpha={RIDGE_ALPHA}  HRD top20 threshold={TOP_THR}")

X = pd.read_parquet(EMB)
if "patient" in X.columns:
    X = X.set_index("patient")
X.index = X.index.astype(str).str.upper().str.slice(0, 12)

L = pd.read_parquet(LABELS)
if "patient" not in L.columns:
    L = L.rename(columns={"case_id":"patient"})
L["patient"] = L["patient"].astype(str).str.upper().str.slice(0, 12)
if "cancer" not in L.columns:
    L["cancer"] = "UNK"
L = L.set_index("patient")

common = X.index.intersection(L.index)
X = X.loc[common].sort_index()
L = L.loc[common].sort_index()

# ensure split exists; if missing, make a deterministic split (80/10/10) but keep your existing if present
if "split" not in L.columns:
    rng = np.random.RandomState(42)
    r = pd.Series(rng.rand(len(L)), index=L.index)
    L["split"] = pd.cut(r.rank(method="first"), bins=[0, 0.8, 0.9, 1.0], labels=["train","val","test"])

# Use HRD numeric if present; else HRD_genomic as fallback.
if "HRD" not in L.columns or L["HRD"].isna().all():
    if "HRD_genomic" in L.columns and L["HRD_genomic"].notna().any():
        L["HRD"] = L["HRD_genomic"].astype(float)
    else:
        raise RuntimeError("No numeric HRD column present (HRD/HRD_genomic).")

# make sure HRD is float
L["HRD"] = pd.to_numeric(L["HRD"], errors="coerce")

# create split index lists (not boolean masks!) on the ALIGNED index
idx_tr = L.index[(L["split"]=="train") & L["HRD"].notna()]
idx_va = L.index[(L["split"]=="val")   & L["HRD"].notna()]
idx_te = L.index[(L["split"]=="test")  & L["HRD"].notna()]

# define binary labels from threshold (top-20% over TRAIN); use locked TOP_THR
L["HRD_top20"] = (L["HRD"] >= TOP_THR).astype("Int64")

print(f"[aligned] patients={len(L):,}  splits={{'train': {len(idx_tr)}, 'val': {len(idx_va)}, 'test': {len(idx_te)}}}")

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=PCA_N, random_state=42)),
    ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=42)),
])

pipe.fit(X.loc[idx_tr], L.loc[idx_tr, "HRD"].astype(float))

# Platt on TRAIN ridge scores vs binary target (using TRAIN-only threshold)
z_tr = pipe.predict(X.loc[idx_tr]).reshape(-1, 1)
platt = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(
    z_tr, L.loc[idx_tr, "HRD_top20"].astype(int)
)

def predict_prob(index_like):
    z = pipe.predict(X.loc[index_like]).reshape(-1, 1)
    return pd.Series(platt.predict_proba(z)[:,1], index=index_like)

p_va = predict_prob(idx_va)
p_te = predict_prob(idx_te)

def safe_auc_ap(y_true, p):
    y = pd.Series(y_true).astype("Int64")
    m = y.notna() & pd.Series(p).notna()
    y = y[m].astype(int).values
    p = np.asarray(p)[m.values]
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return np.nan, np.nan, pos, neg, "one_class_eval"
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p)), pos, neg, None

val_auc, val_ap, val_pos, val_neg, val_reason = safe_auc_ap(L.loc[idx_va, "HRD_top20"], p_va)
tes_auc, tes_ap, tes_pos, tes_neg, tes_reason = safe_auc_ap(L.loc[idx_te, "HRD_top20"], p_te)

print("\n=== RESULTS (aligned, locked config) ===")
print(json.dumps({
    "embeddings_file": str(EMB),
    "pca_n": PCA_N,
    "ridge_alpha": RIDGE_ALPHA,
    "thr_train": TOP_THR,
    "counts": {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)},
    "val": {"auc": val_auc, "ap": val_ap, "pos": val_pos, "neg": val_neg, "reason": val_reason},
    "test": {"auc": tes_auc, "ap": tes_ap, "pos": tes_pos, "neg": tes_neg, "reason": tes_reason},
}, indent=2))


# SECTION 9: LEAVE-ONE-CANCER-OUT (LOCO) EVALUATION

# Script 8 = AUC Multi-Cancer (LOSO site-disjoint OOD) 
import json, re
from collections import Counter, defaultdict
from tqdm import tqdm

ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")

plocks = sorted((ROOT/"results/paper_lock").glob("paperlock_*"), key=lambda p: p.stat().st_mtime, reverse=True)
if plocks and (plocks[0]/"summary.json").exists():
    SUM = json.loads((plocks[0]/"summary.json").read_text(encoding="utf-8"))
    EMB = Path(SUM["embeddings_file"])
    PCA_N = int(SUM.get("pca_n", 384))
    RIDGE_ALPHA = float(SUM.get("ridge_alpha", 30.0))
    TOP_THR = float(SUM.get("thr_train", 33.0))
else:
    EMB = sorted((ROOT/"artifacts/embeddings").glob("patient_means_*.parquet"), key=lambda p:p.stat().st_mtime, reverse=True)[0]
    PCA_N, RIDGE_ALPHA, TOP_THR = 384, 30.0, 33.0

LABELS = ROOT/"artifacts/labels/labels.parquet"

def is_tcga(s): 
    s=str(s).upper(); return s.startswith("TCGA-")
def k12(s): 
    return str(s).upper()[:12]

def derive_site_from_path(p: str) -> str:
    p = str(p).lower()
    # look for tokens near scanner/site words
    toks = re.split(r"[\\/]", p)
    for t in toks[::-1]:
        if any(k in t for k in ["site","center","scanner","hospital","lab"]):
            return t
    # fallback: immediate parent or cancer folder
    return toks[-2] if len(toks) >= 2 else "unknown"

def latest_registry():
    cands=[]
    for d in (ROOT/"runs").glob("run_*"):
        reg = d/"registry.csv"
        if reg.exists(): cands.append((reg.stat().st_mtime, reg))
    return sorted(cands, reverse=True)[0][1] if cands else None

print(f"[emb] {EMB.name}  | [cfg] PCA={PCA_N}  Ridge={RIDGE_ALPHA}  thr(train)={TOP_THR}")

# --- load embeddings ---
X = pd.read_parquet(EMB)
if "patient" in X.columns: X = X.set_index("patient")
X.index = X.index.astype(str).map(k12)

# --- load labels ---
L = pd.read_parquet(LABELS)
if "patient" not in L.columns: L = L.rename(columns={"case_id":"patient"})
L["patient"] = L["patient"].astype(str).map(k12)
if "cancer" not in L.columns: L["cancer"] = "UNK"
if "HRD" not in L.columns or L["HRD"].isna().all():
    if "HRD_genomic" in L.columns and L["HRD_genomic"].notna().any():
        L["HRD"] = pd.to_numeric(L["HRD_genomic"], errors="coerce")
    else:
        raise RuntimeError("No numeric HRD field (HRD/HRD_genomic).")
L = L.set_index("patient")

# align
common = X.index.intersection(L.index)
X = X.loc[common].sort_index()
L = L.loc[common].sort_index()
L["HRD_top20"] = (L["HRD"] >= TOP_THR).astype("Int64")

# --- attach site from registry (majority per patient) ---
reg = latest_registry()
if reg is None:
    raise FileNotFoundError("No registry.csv found under runs/*/. Needed for site OOD.")
R = pd.read_csv(reg)
# standardize columns
col_map = {c.lower(): c for c in R.columns}
sid_col  = pick("slide_id","slide","slideid","case_id")
pat_col  = pick("patient","case_id","patient_id")
path_col = pick("slide_path","path","filepath","wsi_path")
if path_col is None or pat_col is None:
    raise RuntimeError("registry.csv lacks slide path/patient columns.")
R["_site"] = R[path_col].map(derive_site_from_path)
R["_patient12"] = R[pat_col].astype(str).map(k12)
# majority site per patient
site_by_patient = (R.groupby("_patient12")["_site"]
                     .agg(lambda s: Counter(s).most_common(1)[0][0]))
L["site"] = L.index.map(site_by_patient).fillna("unknown")

# --- train/val/test indices (non-null HRD) ---
idx_tr = L.index[(L["split"]=="train") & L["HRD"].notna()]
idx_va = L.index[(L["split"]=="val")   & L["HRD"].notna()]
idx_te = L.index[(L["split"]=="test")  & L["HRD"].notna()]

# locked base model on TRAIN
pipe = Pipeline([
    ("z",   StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=PCA_N, random_state=42)),
    ("rg",  Ridge(alpha=RIDGE_ALPHA, random_state=42)),
]).fit(X.loc[idx_tr], L.loc[idx_tr, "HRD"].astype(float))

z_tr = pipe.predict(X.loc[idx_tr]).reshape(-1,1)
platt = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(
    z_tr, L.loc[idx_tr, "HRD_top20"].astype(int)
)

def prob(ix):
    return pd.Series(platt.predict_proba(pipe.predict(X.loc[ix]).reshape(-1,1))[:,1], index=ix)

# --- LOSO across sites present in VAL+TEST (OOD) ---
def auc_ap(y, p):
    y = pd.Series(y).astype("Int64")
    m = y.notna() & pd.Series(p).notna()
    y = y[m].astype(int).values
    p = np.asarray(p)[m.values]
    pos, neg = int((y==1).sum()), int((y==0).sum())
    if pos==0 or neg==0:
        return np.nan, np.nan, pos, neg, "one_class_eval"
    return float(roc_auc_score(y,p)), float(average_precision_score(y,p)), pos, neg, None

eval_rows = []
sites = sorted(L.loc[idx_va.union(idx_te), "site"].unique())
for s in sites:
    te_idx = L.index[(L["site"]==s) & (L["HRD"].notna()) & (L["split"].isin(["val","test"]))]
    if len(te_idx)==0: continue
    # ensure both classes exist
    y = L.loc[te_idx, "HRD_top20"].astype("Int64")
    if (y==1).sum()==0 or (y==0).sum()==0:
        eval_rows.append({"site": s, "n": int(len(te_idx)), "pos": int((y==1).sum()),
                          "auc": np.nan, "ap": np.nan, "reason": "one_class_eval"})
        continue
    # train on TRAIN + other sites from VAL/TEST except s (keep training domain free of s)
    tr_idx = idx_tr.union(L.index[(L["site"]!=s) & (L["HRD"].notna()) & (L["split"].isin(["val","test"]))])
    # refit ridge+platt on this domain
    mdl = Pipeline([("z",StandardScaler()),("pca",PCA(n_components=PCA_N, random_state=42)),("rg",Ridge(alpha=RIDGE_ALPHA, random_state=42))])
    mdl.fit(X.loc[tr_idx], L.loc[tr_idx,"HRD"].astype(float))
    z = mdl.predict(X.loc[tr_idx]).reshape(-1,1)
    cal = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(z, (L.loc[tr_idx,"HRD"]>=TOP_THR).astype(int))
    p = pd.Series(cal.predict_proba(mdl.predict(X.loc[te_idx]).reshape(-1,1))[:,1], index=te_idx)
    auc, ap, pos, neg, reason = auc_ap(L.loc[te_idx,"HRD_top20"], p)
    eval_rows.append({"site": s, "n": int(len(te_idx)), "pos": pos, "neg": neg, "auc": auc, "ap": ap, "reason": reason})

OOD = pd.DataFrame(eval_rows).sort_values("n", ascending=False)
print("\n=== LOSO site-disjoint (VAL+TEST as OOD) ===")
display(OOD.head(20))

OUTDIR = ROOT/"results/paper_lock"
OUTDIR.mkdir(parents=True, exist_ok=True)
OOD.to_csv(OUTDIR/"supp_site_LOSO.csv", index=False)
print("WROTE →", OUTDIR/"supp_site_LOSO.csv")


# SECTION 10: LEAKGUARD REFIT & INTERNAL VALIDATION

# Script 9 = LeakGuard + Refit
import os, re, json, time, warnings
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)

warnings.filterwarnings("ignore")

ROOT        = Path(r"D:\个人文件夹\Sanwal\DL_V2")
LABELS_PQ   = ROOT / r"artifacts\labels\labels.parquet"
EMB_DIRS    = [ROOT / "results" / "S07_prep", ROOT / "artifacts" / "embeddings"]
EMB_GLOB    = "patient_means_clean_*emb_openclip_vitb16_turbo.parquet"

PCA_N       = 384
RIDGE_ALPHA = 30.0
TOP_FRAC    = 0.20
SEED        = 42
BOOT_N      = 200

LABEL_COLS  = {"HRD","HRD_top20","HRD_expr","HRD_scar","HRD_sig3","HRD_genomic","split","cancer","HRD_source"}
SUSPECT_PAT = re.compile(r"(hrd|label|target|split|cancer|site|scanner|fold|leak|_y$|_label$)", re.I)

def latest_embeddings():
    cands=[]
    for d in EMB_DIRS:
        if d.exists(): cands += list(d.glob(EMB_GLOB))
    if not cands: raise FileNotFoundError(f"No embeddings found under {EMB_DIRS} matching {EMB_GLOB}")
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def boot_ci_metric(y, p, metric_fn, n=BOOT_N, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2: return (np.nan,np.nan,np.nan,"one_class_eval")
    try: point = float(metric_fn(y,p))
    except Exception: return (np.nan,np.nan,np.nan,"metric_error")
    idx = np.arange(len(y)); vals=[]
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        try: vals.append(float(metric_fn(y[b], p[b])))
        except: pass
    if not vals: return (point,np.nan,np.nan,"ci_error")
    lo,hi = np.percentile(vals,[2.5,97.5])
    return (point,float(lo),float(hi),None)

def ensure_pack(root: Path, tag="naturepack_nolkg"):
    out = root / "results" / "paper_lock" / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    for sub in ["main_tables","supp_tables","main_figs","supp_figs"]: (out/sub).mkdir(parents=True, exist_ok=True)
    return out

def fig_save(fig, out_no_ext: Path):
    fig.tight_layout()
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_no_ext.with_suffix(".pdf"),            bbox_inches="tight")
    plt.close(fig)

def plot_roc(y,p,title):
    fpr,tpr,_ = roc_curve(y,p); A = roc_auc_score(y,p)
    fig = plt.figure(figsize=(4,4))
    plt.plot(fpr,tpr,label=f"AUC={A:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right"); return fig

def plot_pr(y,p,title):
    from sklearn.metrics import auc as _auc
    prec,rec,_ = precision_recall_curve(y,p); A=_auc(rec,prec)
    fig = plt.figure(figsize=(4,4))
    plt.plot(rec,prec,label=f"AP={A:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend(loc="lower left"); return fig

def plot_calibration(y, p, title, bins=10):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    q = np.quantile(p, np.linspace(0,1,bins+1)); q[0],q[-1]=0,1
    idx = np.digitize(p, q[1:-1], right=True)
    obs,exp=[],[]
    for b in range(bins):
        m = (idx==b)
        if m.sum()==0: continue
        obs.append(np.mean(y[m])); exp.append(np.mean(p[m]))
    obs=np.array(obs); exp=np.array(exp)
    fig = plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1],"--",label="Perfect")
    plt.scatter(exp,obs,s=25)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed positive rate")
    plt.title(title); plt.legend(loc="lower right"); return fig

def decision_curve(y, p, steps=101):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    ts = np.linspace(0.01,0.99,steps)
    N=len(y); curves=[]
    for t in ts:
        pred = (p>=t).astype(int)
        TP=int((pred & (y==1)).sum()); FP=int((pred & (y==0)).sum())
        NB = (TP/N) - (FP/N)*(t/(1-t))
        curves.append((t,NB))
    return np.array(curves)

def plot_decision(y,p,title):
    cur = decision_curve(y,p,steps=91)
    ts = cur[:,0]; NB = cur[:,1]
    prev = np.mean(y); NB_all = prev - (1-prev)*(ts/(1-ts)); NB_none = np.zeros_like(ts)
    fig = plt.figure(figsize=(5,4))
    plt.plot(ts, NB, label="Model")
    plt.plot(ts, NB_all, label="Treat All", linestyle="--")
    plt.plot(ts, NB_none, label="Treat None", linestyle=":")
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title(title); plt.legend(loc="best"); return fig

labels = pd.read_parquet(LABELS_PQ)
if "patient" not in labels.columns: labels = labels.rename(columns={"case_id":"patient"})
if "cancer"  not in labels.columns: labels["cancer"] = "UNK"
labels["patient"] = labels["patient"].astype(str)

EMB = latest_embeddings()
E = pd.read_parquet(EMB)
if "patient" in E.columns: E = E.set_index("patient")
E.index = E.index.astype(str)

df0 = labels.merge(E.reset_index(), on="patient", how="inner").set_index("patient")

num_cols = df0.select_dtypes(include=["number"]).columns.tolist()
suspect_by_name = [c for c in df0.columns if (c in LABEL_COLS) or SUSPECT_PAT.search(c or "")]
# Find strongly label-correlated numeric columns (|r|>0.98) on TRAIN where HRD notna
m_tr = (df0.get("split","train")== "train") if "split" in df0.columns else (pd.Series(index=df0.index, data="train")=="train")
tr_mask = m_tr & df0["HRD"].notna()
suspect_by_corr = []
if tr_mask.any():
    y_tr = df0.loc[tr_mask, "HRD"].astype(float).values
    for c in num_cols:
        x = df0.loc[tr_mask, c].astype(float).values
        if np.std(x)==0 or np.isnan(x).any(): continue
        r = np.corrcoef(x, y_tr)[0,1]
        if np.isfinite(r) and abs(r) > 0.98:
            suspect_by_corr.append(c)

# Identify canonical embedding dimension (try common sizes)
num_only = df0[num_cols]
common_dims = [256, 512, 768, 1024]
dim_choice = max([d for d in common_dims if num_only.shape[1] >= d] or [min(512, num_only.shape[1])])

# Preferred feature naming: columns like f0..f511 or dim_*
emb_like = [c for c in num_cols if re.match(r"^(f|feat|dim|embedding)[\._]?\d+$", str(c), re.I)]
if len(emb_like) >= 256:
    # keep the smallest-dim subset (sorted by trailing integer)
    def tailnum(s):
        m = re.search(r"(\d+)$", str(s)); return int(m.group(1)) if m else 10**9
    emb_like_sorted = sorted(emb_like, key=tailnum)
    keep_dims = emb_like_sorted[:dim_choice]
else:
    # fallback: take the 1st dim_choice numeric columns by stable order
    keep_dims = num_cols[:dim_choice]

# Build final keep set = embedding dims only, drop suspects
drop_set = set(suspect_by_name) | set(suspect_by_corr)
keep_dims = [c for c in keep_dims if c not in drop_set]

# If still >512, trim to 512; if <256, raise
if len(keep_dims) > 512: keep_dims = keep_dims[:512]
if len(keep_dims) < 128:
    raise RuntimeError(f"Too few clean features after leakage filter: {len(keep_dims)}")

Xnum = df0[keep_dims].astype("float32")
meta = df0[["cancer","split","HRD"]].copy()
if "split" not in meta.columns:
    # fallback split
    rng = np.random.default_rng(SEED)
    meta["split"] = "train"
    for c,g in meta.groupby("cancer"):
        idx = g.index.values.copy(); rng.shuffle(idx)
        n=len(idx); a=max(1,int(0.1*n)); b=max(a+1,int(0.2*n))
        meta.loc[idx[:a],"split"] = "val"
        meta.loc[idx[a:b],"split"] = "test"

m_tr = (meta["split"]=="train") & meta["HRD"].notna()
m_va = (meta["split"]=="val")   & meta["HRD"].notna()
m_te = (meta["split"]=="test")  & meta["HRD"].notna()

thr = np.nanpercentile(meta.loc[m_tr,"HRD"].values, 100*(1-TOP_FRAC))
meta["HRD_top20"] = (meta["HRD"] >= thr).astype(int)

Z = Xnum.to_numpy(copy=False)
y_tr_reg = meta.loc[m_tr,"HRD"].astype(float).values
y_tr_bin = meta.loc[m_tr,"HRD_top20"].astype(int).values

pipe = Pipeline([
    ("z",     StandardScaler(with_mean=True, with_std=True)),
    ("pca",   PCA(n_components=min(PCA_N, Z.shape[1]-1), random_state=SEED)),
    ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
])
pipe.fit(Z[m_tr.values], y_tr_reg)

platt = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
    pipe.predict(Z[m_tr.values]).reshape(-1,1),
    y_tr_bin
)

def predict(mask):
    s = pipe.predict(Z[mask.values]).reshape(-1,1)
    return platt.predict_proba(s)[:,1]

p_va = predict(m_va); y_va = meta.loc[m_va,"HRD_top20"].astype(int).values
p_te = predict(m_te); y_te = meta.loc[m_te,"HRD_top20"].astype(int).values

def metrics_block(y,p):
    auc,lo,hi,rsn = boot_ci_metric(y,p,roc_auc_score)
    ap ,lo2,hi2,_ = boot_ci_metric(y,p,average_precision_score)
    brier = float(brier_score_loss(y,p))
    return dict(auc=float(auc), auc_lo=(None if np.isnan(lo) else float(lo)),
                auc_hi=(None if np.isnan(hi) else float(hi)),
                ap=float(ap), ap_lo=(None if np.isnan(lo2) else float(lo2)),
                ap_hi=(None if np.isnan(hi2) else float(hi2)),
                brier=brier, n=int(len(y)), pos=int(np.sum(y)), neg=int(len(y)-np.sum(y)),
                reason_if_nan=rsn)

val_block = metrics_block(y_va, p_va)
te_block  = metrics_block(y_te, p_te)

rows=[]
for split_name, mask, p, y in [("val",m_va,p_va,y_va), ("test",m_te,p_te,y_te)]:
    part = meta.loc[mask, ["cancer"]].copy()
    part["y"] = y; part["p"] = p
    for c,g in part.groupby("cancer"):
        yy = g["y"].to_numpy().astype(int); pp = g["p"].to_numpy().astype(float)
        if len(np.unique(yy)) < 2:
            rows.append({"split":split_name,"cancer":c,"n":int(len(yy)),
                         "pos":int(yy.sum()),"neg":int(len(yy)-yy.sum()),
                         "auc":None,"ap":None,"reason":"one_class_eval"})
            continue
        rows.append({"split":split_name,"cancer":c,"n":int(len(yy)),
                     "pos":int(yy.sum()),"neg":int(len(yy)-yy.sum()),
                     "auc":float(roc_auc_score(yy,pp)), "ap":float(average_precision_score(yy,pp)),
                     "reason":None})
per_cancer = pd.DataFrame(rows).sort_values(["split","cancer"])

OUT = ensure_pack(ROOT, tag="naturepack_nolkg")

# main tables
pd.DataFrame([
    {"split":"val",  **val_block},
    {"split":"test", **te_block},
]).to_csv(OUT/"main_tables"/"overall_metrics.csv", index=False)

# supp tables
per_cancer.to_csv(OUT/"supp_tables"/"per_cancer_metrics.csv", index=False)

# figs
fig_save(plot_roc(y_va,p_va,"ROC (VAL) [no leakage]"),      OUT/"main_figs"/"roc_val")
fig_save(plot_pr (y_va,p_va,"PR (VAL) [no leakage]"),       OUT/"main_figs"/"pr_val")
fig_save(plot_roc(y_te,p_te,"ROC (TEST) [no leakage]"),     OUT/"main_figs"/"roc_test")
fig_save(plot_pr (y_te,p_te,"PR (TEST) [no leakage]"),      OUT/"main_figs"/"pr_test")
fig_save(plot_calibration(y_va,p_va,"Calibration (VAL)"),   OUT/"supp_figs"/"calibration_val")
fig_save(plot_calibration(y_te,p_te,"Calibration (TEST)"),  OUT/"supp_figs"/"calibration_test")
fig_save(plot_decision(y_va,p_va,"Decision Curve (VAL)"),   OUT/"supp_figs"/"decision_val")
fig_save(plot_decision(y_te,p_te,"Decision Curve (TEST)"),  OUT/"supp_figs"/"decision_test")

# leakage report
report = {
    "embeddings_file": str(EMB),
    "original_numeric_cols": len(num_cols),
    "suspect_by_name": sorted(list(set(suspect_by_name))),
    "suspect_by_corr_|r|>0.98": sorted(list(set(suspect_by_corr))),
    "kept_feature_count": int(len(keep_dims)),
    "kept_feature_examples": keep_dims[:10],
    "threshold_train": float(thr),
    "val": val_block,
    "test": te_block,
}
(OUT/"leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("=== LeakGuard summary ===")
print(json.dumps(report, indent=2))
print(f"\nSaved tables/figs under:\n- {OUT/'main_tables'}\n- {OUT/'supp_tables'}\n- {OUT/'main_figs'}\n- {OUT/'supp_figs'}")


# SECTION 11: FROZEN MODEL & UNIVERSAL SCORER

# Script 10. Final AUC Caliberation. Freeze model + universal scorer 
import os, re, json, time, warnings, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")

ROOT        = Path(r"D:\个人文件夹\Sanwal\DL_V2")
LABELS_PQ   = ROOT / r"artifacts\labels\labels.parquet"
EMB_DIRS    = [ROOT / "results" / "S07_prep", ROOT / "artifacts" / "embeddings"]
EMB_GLOB    = "patient_means_clean_*emb_openclip_vitb16_turbo.parquet"

# External cohorts (set to None or [] if you don't have them ready yet).
# If you have per-patient embeddings parquet for each cohort, point to them here.
EXTERNAL = {
    # "BRCA": r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\BRCA\patient_means.parquet",
    # "OV":   r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\OV\patient_means.parquet",
    # "CCRCC":r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\CCRCC\patient_means.parquet",
    # "UCEC": r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC\patient_means.parquet",
}

PCA_N       = 384
RIDGE_ALPHA = 30.0
TOP_FRAC    = 0.20
SEED        = 42
BOOT_N      = 200

# Use the leakage_report from your last run if present; else fall back to f000.. columns
leak_reports = sorted((ROOT/"results"/"paper_lock").glob("naturepack_nolkg_*/leakage_report.json"))
keep_dims = None
if leak_reports:
    rep = json.loads(leak_reports[-1].read_text(encoding="utf-8"))
    keep_dims = rep.get("kept_feature_examples", [])
    # If examples exist, reconstruct full set if they follow f000..fXXX naming
    if keep_dims and re.match(r"^f\d+$", keep_dims[0]):
        # assume 512 dims f000..f511
        keep_dims = [f"f{str(i).zfill(3)}" for i in range(rep.get("kept_feature_count", 512))]
else:
    # default to 512-d f000..f511
    keep_dims = [f"f{str(i).zfill(3)}" for i in range(512)]

labels = pd.read_parquet(LABELS_PQ)
if "patient" not in labels.columns: labels = labels.rename(columns={"case_id":"patient"})
if "cancer"  not in labels.columns: labels["cancer"] = "UNK"
labels["patient"] = labels["patient"].astype(str)

EMB = latest_embeddings()
E = pd.read_parquet(EMB)
if "patient" in E.columns: E = E.set_index("patient")
E.index = E.index.astype(str)

# If some keep_dims missing, restrict to intersection with available numeric columns
num_cols = E.select_dtypes(include=["number"]).columns.tolist()
keep_dims = [c for c in keep_dims if c in num_cols]
if len(keep_dims) < 128:
    raise RuntimeError(f"Too few clean features available: {len(keep_dims)}")

df = labels.merge(E[keep_dims].reset_index(), on="patient", how="inner").set_index("patient")
Z  = df[keep_dims].astype("float32").to_numpy(copy=False)
meta = df[["cancer","split","HRD"]].copy()

m_tr = (meta["split"]=="train") & meta["HRD"].notna()
m_va = (meta["split"]=="val")   & meta["HRD"].notna()
m_te = (meta["split"]=="test")  & meta["HRD"].notna()

thr = np.nanpercentile(meta.loc[m_tr,"HRD"].values, 100*(1-TOP_FRAC))
meta["HRD_top20"] = (meta["HRD"] >= thr).astype(int)

pipe = Pipeline([
    ("z",     StandardScaler(with_mean=True, with_std=True)),
    ("pca",   PCA(n_components=min(PCA_N, Z.shape[1]-1), random_state=SEED)),
    ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
]).fit(Z[m_tr.values], meta.loc[m_tr,"HRD"].astype(float).values)

platt = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
    pipe.predict(Z[m_tr.values]).reshape(-1,1),
    meta.loc[m_tr,"HRD_top20"].astype(int).values
)

def block(mask, name):
    s = pipe.predict(Z[mask.values]).reshape(-1,1)
    p = platt.predict_proba(s)[:,1]
    y = meta.loc[mask,"HRD_top20"].astype(int).values
    auc,lo,hi,rsn = boot_ci_metric(y,p,roc_auc_score)
    ap ,lo2,hi2,_ = boot_ci_metric(y,p,average_precision_score)
    brier = brier_score_loss(y,p)
    return {"split":name,"auc":float(auc),"auc_lo":None if np.isnan(lo) else float(lo),
            "auc_hi":None if np.isnan(hi) else float(hi),
            "ap":float(ap),"ap_lo":None if np.isnan(lo2) else float(lo2),
            "ap_hi":None if np.isnan(hi2) else float(hi2),
            "brier":float(brier),"n":int(len(y)),"pos":int(y.sum()),"neg":int(len(y)-y.sum()),
            "p":p,"y":y,"reason":rsn}

VAL = block(m_va, "val"); TEST = block(m_te, "test")

OUT = ensure_pack(ROOT, tag="naturepack_lock")
(OUT/"models").mkdir(exist_ok=True)

joblib.dump({"pipe":pipe, "platt":platt, "keep_dims":keep_dims, "thr_train":float(thr)},
            OUT/"models"/"frozen_model.joblib")
json.dump({
    "embeddings_file": str(EMB),
    "keep_dims": keep_dims,
    "thr_train": float(thr),
    "pca_n": int(min(PCA_N, Z.shape[1]-1)),
    "ridge_alpha": float(RIDGE_ALPHA),
}, open(OUT/"models"/"model_card.json","w"), indent=2)

pd.DataFrame([ {k:v for k,v in VAL.items() if k not in ("p","y")},
               {k:v for k,v in TEST.items() if k not in ("p","y")} ]).to_csv(OUT/"main_tables"/"overall_metrics.csv", index=False)

# figs
for name, blk in [("VAL",VAL), ("TEST",TEST)]:
    fig_save(plot_roc(blk["y"], blk["p"], f"ROC ({name})"), OUT/"main_figs"/f"roc_{name.lower()}")
    fig_save(plot_pr (blk["y"], blk["p"], f"PR ({name})"),  OUT/"main_figs"/f"pr_{name.lower()}")

def score_embeddings(embeddings_parquet: str, out_csv: str):
    """Scores any embeddings parquet (index or column named 'patient').
       Writes CSV with: patient, score."""
    data = pd.read_parquet(embeddings_parquet)
    if "patient" in data.columns: data = data.set_index("patient")
    data.index = data.index.astype(str)
    # load frozen model
    pack = joblib.load(OUT/"models"/"frozen_model.joblib")
    pipe_, platt_, keep_ = pack["pipe"], pack["platt"], pack["keep_dims"]
    # sanitize columns
    avail = [c for c in keep_ if c in data.columns]
    if len(avail) < 0.8*len(keep_):
        missing = [c for c in keep_ if c not in data.columns][:10]
        raise RuntimeError(f"Only {len(avail)}/{len(keep_)} required dims available. Missing (first): {missing}")
    X = data[avail].astype("float32").to_numpy(copy=False)
    s = pipe_.predict(X).reshape(-1,1)
    p = platt_.predict_proba(s)[:,1]
    out = pd.DataFrame({"patient": data.index, "HRD_prob": p})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out

ext_rows=[]
for cohort, emb_path in EXTERNAL.items():
    if not emb_path: continue
    emb_path = Path(emb_path)
    if not emb_path.exists():
        print(f"[EXT] {cohort}: embeddings parquet not found → {emb_path}")
        continue
    try:
        scores = score_embeddings(str(emb_path), str(OUT/"supp_tables"/f"{cohort}_scores.csv"))
        # try to pick up labels if shipped alongside (manifest.csv with patient + HRD_top20/target/label)
        man = emb_path.parent/"manifest.csv"
        if man.exists():
            m = pd.read_csv(man)
            # normalize columns
            cols = {c.lower():c for c in m.columns}
            pid_col = cols.get("patient") or cols.get("patient_id") or cols.get("case_id")
            y_col   = cols.get("hrd_top20") or cols.get("target") or cols.get("label")
            if pid_col and y_col:
                mm = m[[pid_col, y_col]].rename(columns={pid_col:"patient", y_col:"y"})
                mm["patient"] = mm["patient"].astype(str)
                joined = scores.merge(mm, on="patient", how="inner")
                if joined["y"].nunique() >= 2:
                    auc = roc_auc_score(joined["y"], joined["HRD_prob"])
                    ap  = average_precision_score(joined["y"], joined["HRD_prob"])
                    ext_rows.append({"cohort":cohort,"n":len(joined),
                                     "auc":float(auc),"ap":float(ap)})
                    # figs
                    fig_save(plot_roc(joined["y"].values, joined["HRD_prob"].values, f"ROC ({cohort})"),
                             OUT/"supp_figs"/f"roc_{cohort}")
                    fig_save(plot_pr (joined["y"].values, joined["HRD_prob"].values, f"PR ({cohort})"),
                             OUT/"supp_figs"/f"pr_{cohort}")
                else:
                    ext_rows.append({"cohort":cohort,"n":len(joined),
                                     "auc":None,"ap":None,"reason":"one_class_or_missing"})
            else:
                ext_rows.append({"cohort":cohort,"n":int(len(scores)),
                                 "auc":None,"ap":None,"reason":"no_labels_found"})
        else:
            ext_rows.append({"cohort":cohort,"n":int(len(scores)),
                             "auc":None,"ap":None,"reason":"no_manifest"})
    except Exception as e:
        ext_rows.append({"cohort":cohort,"n":0,"auc":None,"ap":None,"reason":str(e)})

if ext_rows:
    pd.DataFrame(ext_rows).to_csv(OUT/"supp_tables"/"external_eval_summary.csv", index=False)

summary = {
  "embeddings_file": str(EMB),
  "features_kept": int(len(keep_dims)),
  "thr_train": float(thr),
  "val": {k:v for k,v in VAL.items() if k not in ("p","y")},
  "test":{k:v for k,v in TEST.items() if k not in ("p","y")},
  "folders": {
      "main_tables": str(OUT/"main_tables"),
      "supp_tables": str(OUT/"supp_tables"),
      "main_figs":   str(OUT/"main_figs"),
      "supp_figs":   str(OUT/"supp_figs"),
      "models":      str(OUT/"models"),
  }
}
print("=== NATURE LOCK PACK ===")
print(json.dumps(summary, indent=2))


# SECTION 12: UNI FEATURE LOADING (BRCA)

#Script 11 Load UNI Features - BRCA
import h5py

def load_uni_h5_features(uni_dir: Path) -> pd.DataFrame:
    """
    Load UNI features from H5 files with proper dimension handling.
    UNI stores as (1, n_patches, 1536) - we need to aggregate patches per patient.
    """
    h5_files = list(uni_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files")
    
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {uni_dir}")
    
    features_dict = {}
    slide_to_patient = {}
    
    for h5_path in tqdm(h5_files, desc="Loading H5 files"):
        try:
            filename = h5_path.stem
            
            # Extract patient ID from filename (TCGA-XX-XXXX pattern)
            match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', filename, re.I)
            if not match:
                continue
            
            patient_id = match.group(1).upper()
            slide_to_patient[filename] = patient_id
            
            with h5py.File(h5_path, 'r') as f:
                # UNI stores features under 'features' key
                if 'features' not in f:
                    print(f"Warning: No 'features' key in {filename}")
                    continue
                
                features = f['features'][:]
                
                # Handle the 3D structure (1, n_patches, 1536)
                if features.ndim == 3:
                    # Remove the first dimension and aggregate patches
                    features = features[0]  # Now shape is (n_patches, 1536)
                    # Aggregate by mean across patches
                    patient_features = np.mean(features, axis=0)  # Shape: (1536,)
                elif features.ndim == 2:
                    # Already (n_patches, feature_dim)
                    patient_features = np.mean(features, axis=0)
                elif features.ndim == 1:
                    # Single feature vector
                    patient_features = features
                else:
                    print(f"Unexpected shape {features.shape} in {filename}")
                    continue
                
                # Store or average if patient already exists (multiple slides per patient)
                if patient_id in features_dict:
                    # Average multiple slides from same patient
                    features_dict[patient_id] = (features_dict[patient_id] + patient_features) / 2
                else:
                    features_dict[patient_id] = patient_features
                    
        except Exception as e:
            print(f"Error reading {h5_path.name}: {e}")
            continue
    
    if not features_dict:
        raise ValueError("No features successfully loaded from H5 files")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(features_dict, orient='index')
    
    # Verify dimension is 1536 (UNI feature size)
    if df.shape[1] != 1536:
        print(f"Warning: Expected 1536 features but got {df.shape[1]}")
    
    # Rename columns to standard format
    df.columns = [f"f{str(i).zfill(4)}" for i in range(df.shape[1])]
    
    print(f"\nLoaded features: {df.shape}")
    print(f"Unique patients: {len(df)}")
    print(f"Sample patient IDs: {list(df.index[:5])}")
    
    # Save slide mapping
    slide_mapping_df = pd.DataFrame.from_dict(
        slide_to_patient, orient='index', columns=['patient_id']
    )
    
    return df, slide_mapping_df

# Load the features
uni_dir = Path(r"D:\个人文件夹\Sanwal\UNI Features\TCGA-BRCA_IDC")
print("Loading UNI-BRCA features...")
uni_features, slide_mapping = load_uni_h5_features(uni_dir)

print(f"\nSuccessfully loaded:")
print(f"  - {len(uni_features)} patient features")
print(f"  - Feature dimensions: {uni_features.shape[1]}")
print(f"  - Total slides processed: {len(slide_mapping)}")

# Check data quality
print("\nData quality checks:")
print(f"  - NaN values: {uni_features.isna().sum().sum()}")
print(f"  - Feature mean: {uni_features.values.mean():.4f}")
print(f"  - Feature std: {uni_features.values.std():.4f}")

# Save for later use
output_dir = Path(r"D:\个人文件夹\Sanwal\DL_V2\results")
uni_features.to_parquet(output_dir / "uni_brca_features.parquet")
slide_mapping.to_csv(output_dir / "uni_slide_mapping.csv")

print(f"\nSaved to:")
print(f"  - {output_dir / 'uni_brca_features.parquet'}")
print(f"  - {output_dir / 'uni_slide_mapping.csv'}")

# Quick preview
print("\nPreview of features:")
print(uni_features.iloc[:5, :5])

# SECTION 13: UNI-BRCA HRD VALIDATION

# Script 12. Complete UNI-BRCA HRD validation
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
from scipy import stats

# Configuration
PROJECT_ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2")
LABELS_PATH = PROJECT_ROOT / "artifacts" / "labels" / "labels.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / f"UNI_BRCA_validation_{datetime.now():%Y%m%d_%H%M%S}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (matching your pipeline)
PCA_N = 384
RIDGE_ALPHA = 30.0
HRD_THRESHOLD = 33.0
SEED = 42
BOOTSTRAP_N = 1000

print("=" * 70)
print("UNI-BRCA HRD VALIDATION")
print("=" * 70)

# Load the UNI features we just extracted
uni_features = pd.read_parquet(PROJECT_ROOT / "results" / "uni_brca_features.parquet")
print(f"Loaded UNI features: {uni_features.shape}")

# Load labels
labels = pd.read_parquet(LABELS_PATH)
if 'patient' in labels.columns:
    labels = labels.set_index('patient')
labels.index = labels.index.str.upper().str.slice(0, 12)

# Filter to BRCA only
brca_mask = labels['cancer'].str.upper().isin(['BRCA', 'BREAST', 'IDC', 'ILC'])
labels_brca = labels[brca_mask].copy()

print(f"Total patients in labels: {len(labels)}")
print(f"BRCA patients in labels: {len(labels_brca)}")

# Find common patients
common_patients = uni_features.index.intersection(labels_brca.index)
print(f"Common BRCA patients: {len(common_patients)}")

if len(common_patients) == 0:
    print("\nNo overlap found. Checking patient ID formats...")
    print(f"UNI sample IDs: {list(uni_features.index[:5])}")
    print(f"BRCA label sample IDs: {list(labels_brca.index[:5])}")
else:
    # Align data
    X = uni_features.loc[common_patients].sort_index()
    y = labels_brca.loc[common_patients].sort_index()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Create binary target
    y['HRD_binary'] = (y['HRD'] >= HRD_THRESHOLD).astype(int)
    
    # Get splits
    train_mask = (y['split'] == 'train') & y['HRD'].notna()
    val_mask = (y['split'] == 'val') & y['HRD'].notna()
    test_mask = (y['split'] == 'test') & y['HRD'].notna()
    
    print(f"\n{'='*50}")
    print("DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Aligned patients: {len(common_patients)}")
    print(f"Train: {train_mask.sum()} ({(y.loc[train_mask, 'HRD_binary']==1).sum()} positive)")
    print(f"Val:   {val_mask.sum()} ({(y.loc[val_mask, 'HRD_binary']==1).sum()} positive)")
    print(f"Test:  {test_mask.sum()} ({(y.loc[test_mask, 'HRD_binary']==1).sum()} positive)")
    
    # HRD distribution
    print(f"\nHRD Score Statistics (BRCA):")
    print(f"  Mean: {y['HRD'].mean():.2f}")
    print(f"  Median: {y['HRD'].median():.2f}")
    print(f"  Std: {y['HRD'].std():.2f}")
    print(f"  % Above threshold ({HRD_THRESHOLD}): {(y['HRD'] >= HRD_THRESHOLD).mean()*100:.1f}%")
    
    if train_mask.sum() > 20:  # Need enough samples
        # Train model
        print(f"\n{'='*50}")
        print("TRAINING MODEL")
        print(f"{'='*50}")
        
        X_train = X[train_mask]
        y_train = y.loc[train_mask, 'HRD'].astype(float)
        y_train_binary = y.loc[train_mask, 'HRD_binary'].astype(int)
        
        # Build pipeline
        n_components = min(PCA_N, X_train.shape[1] - 1, X_train.shape[0] - 1)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('pca', PCA(n_components=n_components, random_state=SEED)),
            ('ridge', Ridge(alpha=RIDGE_ALPHA, random_state=SEED))
        ])
        
        # Train regression
        pipeline.fit(X_train, y_train)
        
        # Platt calibration
        train_scores = pipeline.predict(X_train).reshape(-1, 1)
        platt = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=SEED)
        platt.fit(train_scores, y_train_binary)
        
        print(f"Model trained with {n_components} PCA components")
        
        # Save model
        model_path = OUTPUT_DIR / "uni_brca_model.joblib"
        joblib.dump({
            'pipeline': pipeline,
            'platt': platt,
            'n_components': n_components,
            'threshold': HRD_THRESHOLD
        }, model_path)
        print(f"Model saved to: {model_path}")
        
        # Evaluate on all splits
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        
        results = []
        
        for split_name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
            if mask.sum() < 5:
                continue
                
            X_eval = X[mask]
            y_true = y.loc[mask, 'HRD_binary'].astype(int)
            
            # Predict
            scores = pipeline.predict(X_eval).reshape(-1, 1)
            probs = platt.predict_proba(scores)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_true, probs)
            ap = average_precision_score(y_true, probs)
            
            # Bootstrap CI for AUC
            rng = np.random.RandomState(SEED)
            n = len(y_true)
            auc_scores = []
            for _ in range(min(BOOTSTRAP_N, 500)):  # Faster for demo
                idx = rng.choice(n, size=n, replace=True)
                try:
                    auc_boot = roc_auc_score(y_true.iloc[idx], probs[idx])
                    auc_scores.append(auc_boot)
                except:
                    continue
            
            auc_ci_lower = np.percentile(auc_scores, 2.5) if auc_scores else auc
            auc_ci_upper = np.percentile(auc_scores, 97.5) if auc_scores else auc
            
            results.append({
                'Split': split_name,
                'N': mask.sum(),
                'Positive': int(y_true.sum()),
                'Negative': int(mask.sum() - y_true.sum()),
                'AUC': auc,
                'AUC_CI': f"[{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]",
                'AP': ap
            })
            
            print(f"\n{split_name}:")
            print(f"  N={mask.sum()} (Pos: {y_true.sum()}, Neg: {mask.sum()-y_true.sum()})")
            print(f"  AUC: {auc:.3f} {results[-1]['AUC_CI']}")
            print(f"  AP:  {ap:.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
        
        # Plot ROC curves
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        for ax, (split_name, mask) in zip(axes, [('Validation', val_mask), ('Test', test_mask)]):
            if mask.sum() < 5:
                continue
                
            X_eval = X[mask]
            y_true = y.loc[mask, 'HRD_binary'].astype(int)
            scores = pipeline.predict(X_eval).reshape(-1, 1)
            probs = platt.predict_proba(scores)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'UNI-BRCA (AUC={auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {split_name}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=1200, bbox_inches='tight')
        plt.show()
        
        # Summary comparison
        print(f"\n{'='*70}")
        print("COMPARISON WITH BASELINE")
        print(f"{'='*70}")
        print("Your OpenCLIP (pan-cancer): 0.77 AUC")
        
        test_result = results_df[results_df['Split'] == 'Test']
        if not test_result.empty:
            test_auc = test_result.iloc[0]['AUC']
            print(f"UNI-BRCA (cancer-specific): {test_auc:.3f} AUC")
            
            if test_auc > 0.77:
                print(f"✓ UNI-BRCA outperforms baseline by {test_auc-0.77:.3f}")
            else:
                print(f"→ Baseline is {0.77-test_auc:.3f} better")
        
        print("\nLiterature benchmarks for BRCA HRD:")
        print("  - TCGA-BRCA HRD (2024): 0.71-0.75 AUC")
        print("  - SuRe-Transformer (2024): 0.712 AUC")
        print("  - DeepSMILE (2023): 0.71 AUC")
        
        # Save summary report
        report = f"""
UNI-BRCA HRD Validation Report
Date: {datetime.now():%Y-%m-%d %H:%M}

Dataset:
- UNI features: 1536-dimensional
- Patients: {len(common_patients)} BRCA
- Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}

Model:
- PCA components: {n_components}
- Ridge alpha: {RIDGE_ALPHA}
- HRD threshold: {HRD_THRESHOLD}

Results:
{results_df.to_string(index=False)}

Comparison:
- OpenCLIP baseline: 0.77 AUC (pan-cancer)
- UNI-BRCA: {test_result.iloc[0]['AUC'] if not test_result.empty else 'N/A':.3f} AUC (BRCA-specific)

Output saved to: {OUTPUT_DIR}
"""
        
        with open(OUTPUT_DIR / "report.txt", 'w') as f:
            f.write(report)
        
        print(f"\n✓ All results saved to: {OUTPUT_DIR}")
        
    else:
        print("\n⚠ Insufficient training samples for model building")
        print(f"  Need at least 20 training samples, have {train_mask.sum()}")

# SECTION 14: CPTAC EXTERNAL LABEL PREPARATION

# === CPTAC ONLY: Reuse HRD (BMC Biology 2024) → Join to your WSIs → Clean out TCGA seg files ===
import os, re, sys

# CONFIG
ROOT = Path(r"D:\个人文件夹\Sanwal\IHGAMP Validation scarHRD")  # your project folder (as in screenshot)
DELETE_TCGA_SEGMENTS = True                                    # delete TCGA *.allelic_specific.seg under ROOT
HRD_THRESHOLD = 42                                             # scarHRD cut-off for binary label

# Filenames we expect (we'll auto-discover variants)
BMC_XLSX_CANDIDATES = [
    "CPTAC_HRD_from_BMCBiology2024.xlsx",       # your already-saved CPTAC-only sheet
    "BMCBiology2024_HRD_Table2.xlsx",           # original Additional File 9 (Sheet named 'CPTAC')
]
WSI_MANIFEST_CANDIDATES = [
    "cptac_all_slides_manifest.xlsx",
    "cptac_all_slides_manifest.csv",
]

# Helpers
def find_first(path: Path, names: list[str]) -> Path | None:
    for n in names:
        p = path / n
        if p.exists():
            return p
    # also try fuzzy search if exact name missing
    for p in path.glob("*.xlsx"):
        if "cptac" in p.stem.lower() and ("hrd" in p.stem.lower() or "table2" in p.stem.lower()):
            return p
    for p in path.glob("*.csv"):
        if "cptac" in p.stem.lower() and "manifest" in p.stem.lower():
            return p
    return None

def load_cptac_hrd_table(xlsx_path: Path) -> pd.DataFrame:
    """Load CPTAC sheet from BMC Biology 2024 HRD table (or your CPTAC-only export)."""
    # If it’s the full Additional File 9, pick the sheet that contains 'CPTAC'
    if "table2" in xlsx_path.stem.lower() or "hrd_table2" in xlsx_path.stem.lower():
        xls = pd.ExcelFile(xlsx_path)
        sheet_names = [s for s in xls.sheet_names if "CPTAC" in s.upper()]
        if not sheet_names:
            # fallback to sheet index 1 (Sheet 2 usually CPTAC)
            df = pd.read_excel(xlsx_path, sheet_name=1)
        else:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_names[0])
    else:
        # already CPTAC-only export
        df = pd.read_excel(xlsx_path)
    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]
    return df

def extract_submitter_id_from_row(row: pd.Series) -> str | None:
    pat = re.compile(r"\b(C3[LMN]-\d{5})\b", re.I)
    for v in row.astype(str).values:
        m = pat.search(v)
        if m:
            return m.group(1).upper()
    return None

def standardize_hrd_columns(df: pd.DataFrame, thr: int=42) -> pd.DataFrame:
    # Find continuous HRD column
    cand_sum = [c for c in df.columns if ("hrd" in c.lower() and "bin" not in c.lower())]
    if cand_sum:
        df = df.rename(columns={cand_sum[0]: "HRDsum"})
        df["HRDsum"] = pd.to_numeric(df["HRDsum"], errors="coerce")
    # Find/compute binary
    cand_bin = [c for c in df.columns if ("hrd" in c.lower() and "bin" in c.lower())]
    if cand_bin:
        df = df.rename(columns={cand_bin[0]: "HRD_Binary"})
        df["HRD_Binary"] = pd.to_numeric(df["HRD_Binary"], errors="coerce").fillna(0).astype(int)
    else:
        if "HRDsum" in df.columns:
            df["HRD_Binary"] = (df["HRDsum"] >= thr).astype(int)
        else:
            raise RuntimeError("No HRD columns found (neither continuous nor binary).")
    return df

def dedupe_to_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per submitter_id: prefer tumor row if a sample_type column exists; else max HRDsum."""
    df = df.copy()
    # detect a sample_type-ish column
    st_col = None
    for c in df.columns:
        cl = c.lower()
        if "sample" in cl and "type" in cl:
            st_col = c
            break

    def pick_group(g: pd.DataFrame) -> pd.Series:
        gg = g
        if st_col is not None:
            tum = gg[gg[st_col].astype(str).str.contains("tumor", case=False, na=False)]
            if len(tum):
                gg = tum
        if "HRDsum" in gg.columns and gg["HRDsum"].notna().any():
            gg = gg.sort_values(["HRD_Binary","HRDsum"], ascending=[False, False])
        return gg.iloc[0]

    out = df.groupby("submitter_id", as_index=False).apply(pick_group).reset_index(drop=True)
    return out[["submitter_id","HRDsum","HRD_Binary"] if "HRDsum" in out.columns else ["submitter_id","HRD_Binary"]]

def read_wsi_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    # Ensure submitter_id on slides
    if "submitter_id" not in df.columns:
        source_cols = []
        for c in ["case_id","case_id_guess","patient_id","slide_name","slide_path"]:
            if c in df.columns:
                source_cols.append(c)
        if not source_cols:
            # brute-force across all columns
            df["submitter_id"] = df.astype(str).apply(extract_submitter_id_from_row, axis=1)
        else:
            pat = re.compile(r"\b(C3[LMN]-\d{5})\b", re.I)
            def pull_from_first(x):
                for col in source_cols:
                    s = str(x.get(col, ""))
                    m = pat.search(s)
                    if m: return m.group(1).upper()
                return None
            df["submitter_id"] = df.apply(pull_from_first, axis=1)
    # keep only rows with valid C3 IDs
    df = df[~df["submitter_id"].isna()].copy()
    return df

# 1) Load CPTAC HRD table
hrd_file = find_first(ROOT, BMC_XLSX_CANDIDATES)
if not hrd_file:
    raise FileNotFoundError("Could not find the CPTAC HRD XLSX in the project folder.")
hrd_raw = load_cptac_hrd_table(hrd_file)
# Build submitter_id
hrd_raw["submitter_id"] = hrd_raw.apply(extract_submitter_id_from_row, axis=1)
hrd_raw = hrd_raw.dropna(subset=["submitter_id"]).copy()
# Standardize columns and dedupe
hrd_std = standardize_hrd_columns(hrd_raw, thr=HRD_THRESHOLD)
hrd_clean = dedupe_to_patient(hrd_std)

# 2) Load your CPTAC WSI manifest
wsi_file = find_first(ROOT, WSI_MANIFEST_CANDIDATES)
if not wsi_file:
    raise FileNotFoundError("Could not find cptac_all_slides_manifest (.xlsx/.csv) in the project folder.")
slides = read_wsi_manifest(wsi_file)
# Keep common useful columns if they exist
keep_cols = [c for c in ["submitter_id","cancer_type","slide_path","slide_name"] if c in slides.columns]
slides_slim = slides[keep_cols].drop_duplicates()

# 3) Join → WSI-level manifest + save
manifest = slides_slim.merge(hrd_clean, on="submitter_id", how="inner")
labels_path   = ROOT / "cptac_hrd_labels.cleaned.csv"
manifest_path = ROOT / "cptac_wsi_hrd_manifest.csv"
hrd_clean.to_csv(labels_path, index=False)
manifest.to_csv(manifest_path, index=False)

# 4) Optional: delete TCGA ASCAT segment files under ROOT (keep CPTAC only)
deleted = []
if DELETE_TCGA_SEGMENTS:
    for pat in ["**/TCGA-*.ascat*.allelic_specific.seg", "**/*TCGA*ascat*allelic_specific.seg"]:
        for fp in ROOT.rglob(pat):
            try:
                fp.unlink()
                deleted.append(str(fp))
            except Exception:
                pass  # ignore files locked by OS

# 5) Report
n_pat = hrd_clean["submitter_id"].nunique()
n_wsi = manifest["submitter_id"].nunique()
print("✅ CPTAC HRD labels (BMC Biology 2024) loaded and cleaned.")
print(f"   Patients with labels: {n_pat}")
print(f"   Patients matched to your WSIs: {n_wsi}" + ("" if n_pat==0 else f" ({n_wsi/n_pat*100:.1f}% of labels)"))
print(f"\nSaved:\n  - {labels_path}\n  - {manifest_path}")

if "cancer_type" in manifest.columns and not manifest.empty:
    by_ct = (manifest.groupby("cancer_type")["submitter_id"].nunique()
             .sort_values(ascending=False))
    print("\nBy cancer type (matched patients):")
    print(by_ct.to_string())

if DELETE_TCGA_SEGMENTS:
    print(f"\n🧹 TCGA *.allelic_specific.seg removed under {ROOT}: {len(deleted)} file(s)")
    if deleted:
        for x in deleted[:10]:
            print("   -", x)
        if len(deleted) > 10:
            print(f"   … (+{len(deleted)-10} more)")


# SECTION 15: OPENSLIDEFM INTERNAL TCGA EVALUATION

# IHGAMP-style INTERNAL AUC on TCGA using OpenSlideFM embeddings — HARDENED (leak-safe)
# - Uses your Script-10 protocol (TRAIN-only top20%, Z-score→PCA→Ridge→Platt)
# - STRICT feature whitelist + banlist + NaN handling + leakage correlation screen
# - Variants: 0.5µm, 2.0µm (if present), and concat(0.5+2.0) when both exist
# - Read-only on OpenSlideFM; writes only to D:\个人文件夹\Sanwal\IHGAMP_OpenslideFM\

from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
plt.switch_backend("agg")

DL_ROOT     = Path(r"D:\个人文件夹\Sanwal\DL_V2")
LABELS_PQ   = DL_ROOT / r"artifacts\labels\labels.parquet"  # must have: patient, split(train/val/test), HRD

OSFM_EMB    = Path(r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\tcga_openslidefm_patient_embeddings.parquet")

OUT_ROOT    = Path(r"D:\个人文件夹\Sanwal\IHGAMP_OpenslideFM")
OUT_MODELS  = OUT_ROOT / "models"
OUT_MAIN_T  = OUT_ROOT / "main_tables"
OUT_SUPP_T  = OUT_ROOT / "supp_tables"
OUT_MAIN_F  = OUT_ROOT / "main_figs"
OUT_SUPP_F  = OUT_ROOT / "supp_figs"
OUT_AUDIT   = OUT_ROOT / "audits"
for p in [OUT_ROOT, OUT_MODELS, OUT_MAIN_T, OUT_SUPP_T, OUT_MAIN_F, OUT_SUPP_F, OUT_AUDIT]:
    p.mkdir(parents=True, exist_ok=True)

# Protocol knobs
PCA_N           = 384
RIDGE_ALPHA     = 30.0
TOP_FRAC        = 0.20
SEED            = 42
DROP_NAN_THR    = 0.10   # drop feature columns with >10% NaN on TRAIN
DROP_CONST_THRESH = 1e-12  # drop near-constant columns on TRAIN (std < thresh)
CORR_BAN_TOP20  = 0.995  # absolute corr threshold vs HRD_top20 on TRAIN (leak guard)
CORR_BAN_CONT   = 0.995  # absolute corr threshold vs HRD (continuous) on TRAIN

# Suspicious feature name banlist (regex)
BAN_PATTERNS = re.compile(
    r"(?:^|[_\-])("
    r"oof|pred|prob|score|logit|calib|platt|ridge|lr|svm|xgb|gbm|rf|"
    r"head|cls|target|label|y|hrd|hrdsum|hrd_sum|binary|top20|auc|ap|brier"
    r")(?:$|[_\-])",
    re.IGNORECASE
)

def tcga12(x):
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(x), flags=re.I)
    return m.group(1).upper() if m else str(x)

def norm_scale(s):
    s = str(s).lower().replace("p",".")
    if s.startswith("0.5"): return "0.5"
    if s.startswith("2.0"): return "2.0"
    if "20x" in s: return "0.5"
    if "5x"  in s: return "2.0"
    return s

def metric_block(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    out = {"n": int(len(y)), "pos": int(y.sum()), "neg": int(len(y)-y.sum())}
    out["auc"]   = float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    out["ap"]    = float(average_precision_score(y,p))
    out["brier"] = float(brier_score_loss(y,p))
    return out

def detect_id_col(df: pd.DataFrame):
    # priority: explicit TCGA-like content, then common names, else first object col
    for c in df.columns:
        try:
            if df[c].astype(str).str.contains(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", regex=True).any():
                return c
        except Exception:
            pass
    for c in ["patient","patient_id","tcga_patient","submitter_id","case_id","id"]:
        if c in df.columns: return c
    obj_cols = [c for c in df.columns if df[c].dtype==object]
    if obj_cols: return obj_cols[0]
    raise RuntimeError("Could not detect patient identifier column in OpenSlideFM parquet.")

def strict_feature_matrix(df_feat: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Build a leak-safe numeric feature matrix.
    Priority:
      1) If columns that look like f000..f1535 exist in bulk, use only those.
      2) Else use numeric columns minus banned names.
    Return filtered df and audit info.
    """
    audit = {"initial_cols": df_feat.shape[1], "steps": []}

    cols = list(df_feat.columns)
    # Identify classic f000.. pattern
    fcols = [c for c in cols if re.fullmatch(r"[fF]\d{3,4}", str(c))]
    if len(fcols) >= 128:
        use = fcols
        audit["steps"].append({"rule":"whitelist_fNNN","kept":len(use)})
    else:
        # If numeric-ordinal columns named 0..1535 exist, prefer those
        idx_like = [c for c in cols if (isinstance(c,int) or (isinstance(c,str) and c.isdigit()))]
        if len(idx_like) >= 128:
            use = idx_like
            audit["steps"].append({"rule":"whitelist_indexlike","kept":len(use)})
        else:
            # fallback: numeric columns, ban suspicious names
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_feat[c])]
            safe = []
            banned = []
            for c in num_cols:
                if BAN_PATTERNS.search(str(c)):
                    banned.append(c)
                else:
                    safe.append(c)
            use = safe
            audit["steps"].append({"rule":"numeric_minus_banlist","kept":len(use),"banned":len(banned)})
    Xdf = df_feat[use].copy()
    audit["after_basic"] = Xdf.shape[1]
    return Xdf, audit

def leak_corr_filter(X_tr_df: pd.DataFrame, y_bin: np.ndarray, y_cont: np.ndarray, audit: dict):
    """
    Drop columns with high correlation to target(s) on TRAIN:
      - abs(Pearson corr with y_bin) >= CORR_BAN_TOP20
      - abs(Pearson corr with y_cont) >= CORR_BAN_CONT
    """
    drop = set()
    cols = X_tr_df.columns
    # Corr with binary target
    yb = y_bin.astype(float)
    for c in cols:
        x = X_tr_df[c].values
        if np.sum(np.isfinite(x)) < max(3, int(0.9*len(x))):  # skip very sparse
            continue
        xc = x[np.isfinite(x) & np.isfinite(yb)]
        yc = yb[np.isfinite(x) & np.isfinite(yb)]
        if len(xc) < 3: continue
        # std check to avoid zero division
        if np.std(xc) < 1e-12 or np.std(yc) < 1e-12: 
            continue
        r = np.corrcoef(xc, yc)[0,1]
        if np.isnan(r): continue
        if abs(r) >= CORR_BAN_TOP20:
            drop.add(c)
    # Corr with continuous HRD
    for c in cols:
        if c in drop: continue
        x = X_tr_df[c].values
        if np.sum(np.isfinite(x)) < max(3, int(0.9*len(x))):
            continue
        xc = x[np.isfinite(x) & np.isfinite(y_cont)]
        yc = y_cont[np.isfinite(x) & np.isfinite(y_cont)]
        if len(xc) < 3: continue
        if np.std(xc) < 1e-12 or np.std(yc) < 1e-12:
            continue
        r = np.corrcoef(xc, yc)[0,1]
        if np.isnan(r): continue
        if abs(r) >= CORR_BAN_CONT:
            drop.add(c)
    audit["corr_dropped"] = len(drop)
    return [c for c in cols if c not in drop], sorted(list(drop))

if not LABELS_PQ.exists():
    raise FileNotFoundError(f"labels parquet not found: {LABELS_PQ}")

labels = pd.read_parquet(LABELS_PQ)
labels.columns = [str(c).strip() for c in labels.columns]
if "patient" not in labels.columns:
    if "case_id" in labels.columns: labels = labels.rename(columns={"case_id":"patient"})
    else: raise RuntimeError("labels.parquet must contain 'patient' column")
if "split" not in labels.columns or "HRD" not in labels.columns:
    raise RuntimeError("labels.parquet must contain 'split' and continuous 'HRD' columns")

labels["patient"] = labels["patient"].astype(str).apply(tcga12)
labels["split"]   = labels["split"].astype(str).str.lower()
labels = labels[labels["split"].isin(["train","val","test"])].copy()
# ensure unique patient split
labels = labels.sort_values(["patient","split"]).drop_duplicates(subset=["patient"], keep="last")

# Define HRD_top20 on TRAIN only
m_tr_global = (labels["split"]=="train") & labels["HRD"].notna()
if m_tr_global.sum()==0:
    raise RuntimeError("No TRAIN rows with HRD available in labels.parquet to define top-20%.")
thr = np.nanpercentile(labels.loc[m_tr_global,"HRD"].values, 100*(1-TOP_FRAC))
labels["HRD_top20"] = (labels["HRD"] >= thr).astype(int)

if not OSFM_EMB.exists():
    raise FileNotFoundError(f"OpenSlideFM embeddings parquet not found: {OSFM_EMB}")

raw = pd.read_parquet(OSFM_EMB)
raw.columns = [str(c).strip() for c in raw.columns]

id_col = detect_id_col(raw)
raw["patient"] = raw[id_col].astype(str).apply(tcga12)

# scale column
scale_col = None
for c in raw.columns:
    if c.lower() in ["scale","mpp","mag","resolution","scale_label"]:
        scale_col = c; break
raw["scale"] = raw[scale_col].apply(norm_scale) if scale_col else "0.5"

# Build a basic numeric matrix FIRST so we can filter safely
numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
feat_df = raw[numeric_cols].copy()

# Strict feature selection (whitelist or numeric minus banlist)
X0_df, audit0 = strict_feature_matrix(feat_df)

# Re-attach ids and scale
E0 = pd.concat([raw[["patient","scale"]], X0_df], axis=1)

def make_dataset(scale_tag):
    sub = E0[E0["scale"]==scale_tag].copy()
    if sub.empty: return None
    common = sub.merge(labels, on="patient", how="inner")
    if common.empty: return None

    # isolate features (exclude obvious label columns if they slipped through)
    fcols = [c for c in common.columns
             if pd.api.types.is_numeric_dtype(common[c])
             and c not in ["HRD","HRD_top20"]]
    X = common[fcols].to_numpy(dtype=np.float32, copy=False)
    meta = common[["patient","split","HRD","HRD_top20"]].copy()
    return {"X":X, "meta":meta, "fcols":fcols, "frame":common[["patient"]+fcols]}

d05 = make_dataset("0.5")
d20 = make_dataset("2.0")
variants = {}
if d05: variants["OpenSlideFM_0p5"] = d05
if d20: variants["OpenSlideFM_2p0"] = d20

# concat requires overlap
if d05 and d20:
    m05 = d05["meta"].set_index("patient"); m20 = d20["meta"].set_index("patient")
    common_pat = m05.index.intersection(m20.index)
    if len(common_pat):
        # align order
        i05 = m05.loc[common_pat]; i20 = m20.loc[common_pat]
        # build frames to ensure matched feature sets
        F05 = d05["frame"].set_index("patient").loc[common_pat]
        F20 = d20["frame"].set_index("patient").loc[common_pat]
        # namespaced columns to avoid accidental same-name collisions
        Xc_df = pd.concat(
            [F05.add_prefix("s05_"), F20.add_prefix("s20_")],
            axis=1
        )
        Xc = Xc_df.to_numpy(dtype=np.float32, copy=False)
        meta = i05.reset_index()
        variants["OpenSlideFM_concat_0p5_2p0"] = {"X":Xc, "meta":meta, "fcols":list(Xc_df.columns), "frame":Xc_df.reset_index()}

if not variants:
    raise RuntimeError("No usable variant(s): could not align OpenSlideFM embeddings with labels/splits.")

def run_variant(name, pack):
    X = pack["X"].astype(np.float32, copy=True)
    meta = pack["meta"].copy()
    fcols = pack["fcols"]

    # masks
    m_tr = (meta["split"]=="train") & meta["HRD"].notna()
    m_va = (meta["split"]=="val")   & meta["HRD"].notna()
    m_te = (meta["split"]=="test")  & meta["HRD"].notna()
    if m_tr.sum()==0:
        raise RuntimeError(f"[{name}] No TRAIN rows with HRD.")

    # Define HRD_top20 using global TRAIN threshold
    meta["HRD_top20"] = (meta["HRD"] >= thr).astype(int)

    # Sanitize non-finite
    X[~np.isfinite(X)] = np.nan

    # TRAIN subset for fitting transforms & leak checks
    X_tr = X[m_tr.values]
    y_tr_bin  = meta.loc[m_tr, "HRD_top20"].astype(int).values
    y_tr_cont = meta.loc[m_tr, "HRD"].astype(float).values

    # Drop near-constant columns on TRAIN
    stds = np.nanstd(X_tr, axis=0)
    keep_mask = stds >= DROP_CONST_THRESH
    # Drop columns with too many NaNs on TRAIN
    miss_ratio = np.mean(np.isnan(X_tr), axis=0)
    keep_mask &= (miss_ratio <= DROP_NAN_THR)
    kept1 = int(keep_mask.sum())
    if kept1 < 2:
        raise RuntimeError(f"[{name}] Too few features after const/NaN filtering on TRAIN (kept {kept1}).")

    X = X[:, keep_mask]
    X_tr = X_tr[:, keep_mask]
    kept_fcols_step1 = [fcols[i] if fcols else f"c{i}" for i, k in enumerate(keep_mask) if k]

    # Impute (median) on TRAIN, then transform all splits
    imputer = SimpleImputer(strategy="median")
    X_tr_i  = imputer.fit_transform(X_tr)
    X_va_i  = imputer.transform(X[m_va.values])
    X_te_i  = imputer.transform(X[m_te.values])

    # Leakage correlation guard (TRAIN only), computed after imputation to avoid NaN effects
    X_tr_df = pd.DataFrame(X_tr_i, columns=kept_fcols_step1)
    kept_after_corr, dropped_corr = leak_corr_filter(X_tr_df, y_tr_bin, y_tr_cont, audit={})
    if len(kept_after_corr) < 2:
        raise RuntimeError(f"[{name}] Leak guard removed too many features (kept {len(kept_after_corr)}).")

    # Map kept columns to indices
    col_to_idx = {c:i for i,c in enumerate(kept_fcols_step1)}
    keep_idx2  = np.array([col_to_idx[c] for c in kept_after_corr], dtype=int)
    X_tr_i = X_tr_i[:, keep_idx2]
    X_va_i = X_va_i[:, keep_idx2]
    X_te_i = X_te_i[:, keep_idx2]

    # Pipeline (Script-10 parity): z-score -> PCA -> Ridge
    pca_n = int(min(PCA_N, X_tr_i.shape[1]-1)) if X_tr_i.shape[1] > 1 else 1
    pipe = Pipeline([
        ("z",   StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=pca_n, random_state=SEED)),
        ("rg",  Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
    ]).fit(X_tr_i, y_tr_cont)

    # Platt on TRAIN only
    tr_scores = pipe.predict(X_tr_i).reshape(-1,1)
    platt     = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
                    tr_scores, y_tr_bin)

    # score blocks
    def score_block(Xi, mask):
        s = pipe.predict(Xi).reshape(-1,1)
        p = platt.predict_proba(s)[:,1]
        y = meta.loc[mask,"HRD_top20"].astype(int).values
        return y, p

    yv, pv = score_block(X_va_i, m_va)
    yt, pt = score_block(X_te_i, m_te)

    VAL = metric_block(yv, pv)
    TEST= metric_block(yt, pt)

    # Warn if suspiciously perfect
    warn_flag = (VAL["auc"] is not None and TEST["auc"] is not None and VAL["auc"] >= 0.99 and TEST["auc"] >= 0.99)

    # Save per-variant artifacts
    vtag = name.replace("OpenSlideFM_", "").replace("concat_0p5_2p0","concat")
    (OUT_MODELS / vtag).mkdir(exist_ok=True)

    joblib.dump({
        "imputer": imputer,
        "pipe": pipe,
        "platt": platt,
        "pca_n": pca_n,
        "alpha": float(RIDGE_ALPHA),
        "drop_nan_threshold_train": float(DROP_NAN_THR),
        "drop_const_thresh": float(DROP_CONST_THRESH),
        "corr_ban_top20": float(CORR_BAN_TOP20),
        "corr_ban_cont": float(CORR_BAN_CONT),
        "kept_features": kept_after_corr,
        "warn_suspicious": warn_flag
    }, OUT_MODELS / vtag / "frozen_model.joblib")

    # tables
    pd.DataFrame([{"split":"val", **VAL},{"split":"test", **TEST}]).to_csv(
        OUT_MAIN_T / f"{vtag}_overall_metrics.csv", index=False)

    # predictions
    pd.DataFrame({"patient":meta.loc[m_va,"patient"].values, "p":pv, "y":yv}).to_csv(
        OUT_SUPP_T / f"{vtag}_val_predictions.csv", index=False)
    pd.DataFrame({"patient":meta.loc[m_te,"patient"].values, "p":pt, "y":yt}).to_csv(
        OUT_SUPP_T / f"{vtag}_test_predictions.csv", index=False)

    # figs
    fig_save(plot_roc(yv, pv, f"ROC (VAL) — {name}"),  OUT_MAIN_F / f"roc_val_{vtag}")
    fig_save(plot_pr (yv, pv, f"PR (VAL)  — {name}"),  OUT_MAIN_F / f"pr_val_{vtag}")
    fig_save(plot_roc(yt, pt, f"ROC (TEST) — {name}"), OUT_MAIN_F / f"roc_test_{vtag}")
    fig_save(plot_pr (yt, pt, f"PR (TEST)  — {name}"), OUT_MAIN_F / f"pr_test_{vtag}")

    # audit file for this variant
    audit_rows = []
    audit_rows.append({"stage":"base_selection", "kept_cols":len(kept_fcols_step1)})
    audit_rows.append({"stage":"corr_guard_dropped", "dropped_cols":len(dropped_corr)})
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT / f"{vtag}_feature_audit.csv", index=False)

    return {"name":name, "VAL":VAL, "TEST":TEST,
            "pca_n":pca_n, "n_tr":int(m_tr.sum()),
            "n_val":int(m_va.sum()), "n_te":int(m_te.sum()),
            "warn_suspicious": warn_flag}

print("="*100)
print("IHGAMP-style INTERNAL AUC (OpenSlideFM) — Hardened, leak-safe")
print("="*100)
print(f"Embeddings : {OSFM_EMB}")
print(f"Labels     : {LABELS_PQ}")
print(f"Output dir : {OUT_ROOT}\n")

results = []
for name, pack in variants.items():
    print(f"\n=== Running {name} ===")
    res = run_variant(name, pack)
    results.append(res)
    print(f"VAL  AUC={res['VAL']['auc']},  AP={res['VAL']['ap']},  Brier={res['VAL']['brier']}  | n={res['n_val']}")
    print(f"TEST AUC={res['TEST']['auc']}, AP={res['TEST']['ap']}, Brier={res['TEST']['brier']} | n={res['n_te']}")
    if res["warn_suspicious"]:
        print("⚠️  WARNING: AUC≥0.99 on both VAL/TEST — likely residual leakage. Inspect audits & inputs.")

card = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "labels_source": str(LABELS_PQ),
    "embeddings_source": str(OSFM_EMB),
    "protocol_target": "HRD_top20 (top 20% of TRAIN by continuous HRD)",
    "splits": ["train","val","test"],
    "pipeline": "Imputer(median; TRAIN) -> Z-score -> PCA -> Ridge -> Platt(LogReg on TRAIN)",
    "PCA_N_cap": int(PCA_N),
    "Ridge_alpha": float(RIDGE_ALPHA),
    "drop_nan_threshold_train": float(DROP_NAN_THR),
    "drop_const_thresh": float(DROP_CONST_THRESH),
    "corr_ban_top20": float(CORR_BAN_TOP20),
    "corr_ban_cont": float(CORR_BAN_CONT),
    "threshold_top20_train": float(thr),
    "variants": [
        {
            "name": r["name"],
            "pca_n": int(r["pca_n"]),
            "train_n": int(r["n_tr"]),
            "val_n": int(r["n_val"]),
            "test_n": int(r["n_te"]),
            "VAL_auc": r["VAL"]["auc"], "VAL_ap": r["VAL"]["ap"], "VAL_brier": r["VAL"]["brier"],
            "TEST_auc": r["TEST"]["auc"], "TEST_ap": r["TEST"]["ap"], "TEST_brier": r["TEST"]["brier"],
            "warn_suspicious": r["warn_suspicious"],
        } for r in results
    ]
}
with open(OUT_ROOT / "model_card.openslidefm_internal.json", "w", encoding="utf-8") as f:
    json.dump(card, f, indent=2)

for r in results:
    print(f"{r['name']}:  VAL AUC={r['VAL']['auc']}, TEST AUC={r['TEST']['auc']}  | PCA={r['pca_n']}")
print("All outputs saved under:", OUT_ROOT)


# SECTION 16: CPTAC-LUAD EXTERNAL VALIDATION (OpenCLIP)

# LUAD HRD evaluation (OpenCLIP patient embeddings)
# Inputs (edit if your paths differ)
PAT_EMB_PARQUET = r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_luad_v2\cptac_luad_patient_embeddings_fixed.parquet"
LABELS_CSV      = r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\LUAD\labels\cptac_luad_el_nahhas.csv"
OUT_DIR         = r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_luad_v2"

# HRD rules: prefer an explicit binary column; else derive from HRD_sum >= THRESH
HRD_THRESHOLD   = 42

import os, json, re, math, time, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

os.makedirs(OUT_DIR, exist_ok=True)

def _patient_token(x: str) -> str:
    """Normalize patient IDs like 'C3L-00001-21' -> 'C3L-00001'."""
    s = str(x)
    b = re.sub(r"\.svs$|\.tif$|\.tiff$|\.ndpi$|\.mrxs$|\.scn$|\.bif$", "", s, flags=re.I)
    # If 'AAA-00000-XX' style, keep first 2 dash parts
    parts = b.split("-")
    if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9]{3}", parts[0]) and re.fullmatch(r"\d{5}", parts[1]):
        return f"{parts[0]}-{parts[1]}"
    return b

def _find_label_cols(df: pd.DataFrame):
    cmap = {re.sub(r"[^a-z0-9]+","_", c.lower()): c for c in df.columns}
    # patient column
    for k in ["patient","patient_id","case_id","#patient_identifier","patient_identifier","subject","submitter_id"]:
        if k in cmap: pat_col = cmap[k]; break
    else:
        raise KeyError(f"No patient column found in labels. Have: {list(df.columns)[:12]}")
    # binary HRD column
    bin_col = None
    for k in ["hrd_binary","hrd_bin","hrd_status","hrd"]:
        if k in cmap:
            bin_col = cmap[k]; break
    # numeric HRD columns
    hsum = None
    for k in ["hrd_sum","hrdscore","hrd_score","hrdsum"]:
        if k in cmap:
            hsum = cmap[k]; break
    return pat_col, bin_col, hsum

def _bootstrap_ci(y_true, y_score, fn, n_boot=2000, seed=1337):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi, pi = y_true[idx], y_score[idx]
        try:
            vals.append(fn(yi, pi))
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(np.mean(vals)), float(lo), float(hi))

print("[LOAD] patient embeddings…")
E = pd.read_parquet(PAT_EMB_PARQUET)
zcols = [c for c in E.columns if isinstance(c,str) and re.fullmatch(r"z\d{4}", c)]
if "patient" not in E.columns or len(zcols) < 256:
    raise SystemExit(f"[FATAL] Embeddings parquet malformed. patient in cols? {'patient' in E.columns}, dim={len(zcols)}")
E["patient"] = E["patient"].astype(str).map(_patient_token)
print(f"[EMB] patients={E['patient'].nunique()}  dim={len(zcols)}")

print("[LOAD] labels…")
Lraw = pd.read_csv(LABELS_CSV)
pcol, bcol, hsum = _find_label_cols(Lraw)
L = Lraw.copy()
L["patient"] = L[pcol].astype(str).map(_patient_token)

if bcol and L[bcol].dropna().size > 0:
    ycol = "HRD_binary"
    # map common encodings
    def _to_bin(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if s in {"1","true","hrd","pos","positive","yes"}: return 1
        if s in {"0","false","hrp","neg","negative","no"}: return 0
        try: return int(float(s))
        except: return np.nan
    L[ycol] = L[bcol].map(_to_bin)
elif hsum and L[hsum].dropna().size > 0:
    ycol = "HRD_binary"
    L[ycol] = (pd.to_numeric(L[hsum], errors="coerce") >= HRD_THRESHOLD).astype("Int64")
else:
    raise SystemExit("[FATAL] No usable HRD column in labels (need HRD_binary or HRD_sum).")

# de-duplicate to 1 row/patient if needed (keep first)
L = L[["patient", ycol]].dropna().drop_duplicates(subset=["patient"])
print(f"[LAB] rows={len(L)}  pts={L['patient'].nunique()}  HRD+= {int(L[ycol].sum())}  HRD-= {int((1-L[ycol]).sum())}")

J = E.merge(L, on="patient", how="inner")
if len(J) == 0:
    raise SystemExit("[FATAL] no overlap between patient embeddings and labels.")

y = J[ycol].astype(int).to_numpy()
X = J[zcols].astype("float32").to_numpy()
pos = int(y.sum()); neg = int((1-y).sum())
print(f"[JOIN] matched={len(J)}  HRD+= {pos}  HRD-= {neg}  rate={pos/len(J):.3f}")
if pos == 0 or neg == 0:
    raise SystemExit("[FATAL] one-class labels after join.")

print("[CV] Stratified K-Fold with class_weight='balanced'")
n_splits = min(5, pos, neg) if min(pos,neg) < 5 else 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=1337))
])

oof = np.zeros(len(J), dtype=float)
fold_ids = np.full(len(J), -1, dtype=int)

for k, (tr, te) in enumerate(cv.split(X, y), 1):
    pipe.fit(X[tr], y[tr])
    oof[te] = pipe.predict_proba(X[te])[:,1]
    fold_ids[te] = k
    print(f"  fold {k}/{n_splits}  done")

auc  = roc_auc_score(y, oof)
ap   = average_precision_score(y, oof)
br   = brier_score_loss(y, oof)

auc_mean, auc_lo, auc_hi = _bootstrap_ci(y, oof, roc_auc_score)
ap_mean,  ap_lo,  ap_hi  = _bootstrap_ci(y, oof, average_precision_score)

print(f"[METRICS] AUC={auc:.3f} (95% {auc_lo:.3f}-{auc_hi:.3f})  "
      f"AP={ap:.3f} (95% {ap_lo:.3f}-{ap_hi:.3f})  Brier={br:.4f}")

preds = J[["patient"]].copy()
preds["fold"] = fold_ids
preds["y_true"] = y
preds["y_score"] = oof
preds_path = os.path.join(OUT_DIR, "cptac_luad_openclip_preds.csv")
preds.to_csv(preds_path, index=False, encoding="utf-8-sig")

metrics = {
    "n": int(len(y)),
    "pos": int(pos), "neg": int(neg),
    "auc": float(auc), "auc_ci": [float(auc_lo), float(auc_hi)],
    "ap": float(ap), "ap_ci": [float(ap_lo), float(ap_hi)],
    "brier": float(br),
    "splits": int(n_splits),
    "embed_dim": int(len(zcols)),
    "embeddings": os.path.basename(PAT_EMB_PARQUET),
    "labels": os.path.basename(LABELS_CSV)
}
with open(os.path.join(OUT_DIR, "cptac_luad_openclip_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

fpr, tpr, _ = roc_curve(y, oof)
prec, rec, _ = precision_recall_curve(y, oof)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, lw=2)
plt.plot([0,1],[0,1], ls="--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_cptac_luad_openclip.png"), dpi=200)
plt.close()

plt.figure(figsize=(5,4))
plt.plot(rec, prec, lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"PR (AP={ap:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_cptac_luad_openclip.png"), dpi=200)
plt.close()

print("[OK] Saved:")
print(" ", preds_path)
print(" ", os.path.join(OUT_DIR, "cptac_luad_openclip_metrics.json"))
print(" ", os.path.join(OUT_DIR, "roc_cptac_luad_openclip.png"))
print(" ", os.path.join(OUT_DIR, "pr_cptac_luad_openclip.png"))


# SECTION 17: OPENSLIDEFM EXTERNAL EXTRACTION & EVALUATION

# === Re-extract OpenSlideFM-like embeddings via projection head from openslidefm_student.pt ===
# No import of 'openslidefm'. No TorchScript. We: OpenCLIP tiles -> find 768-in Linear in the .pt ->
# project to OSFM token space -> slide mean+std -> patient mean -> evaluate vs El-Nahhas HRD.

import os, re, json, math, warnings
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_STUDENT = Path(r"D:\个人文件夹\Sanwal\OpenSlide\models\openslidefm_student.pt")

COHORTS = {
    "luad": {
        "manifest": r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\LUAD\preflight_luad\luad_matched_manifest_fixed.csv",
        "labels":   r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\LUAD\labels\cptac_luad_el_nahhas_thr42.csv",
        "out_dir":  r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_luad_v2_osfm",
    },
    "lusc": {
        "manifest": r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\LUSC\preflight_lusc\lusc_matched_manifest.csv",
        "labels":   r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\LUSC\labels\cptac_lusc_el_nahhas_thr42.csv",
        "out_dir":  r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_lusc_v2_osfm",
    },
    "hnscc": {
        "manifest": r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\HNSCC\preflight_hnscc\hnscc_matched_manifest.csv",
        "labels":   r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\HNSCC\labels\cptac_hnscc_el_nahhas_thr42.csv",
        "out_dir":  r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_hnscc_v2_osfm",
    },
    "ucec": {
        "manifest": r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC\preflight_ucec\ucec_matched_manifest.csv",
        "labels":   r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC\labels\cptac_ucec_el_nahhas.csv",
        "out_dir":  r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_ucec_v2_osfm",
    },
    # add PAAD when your manifest exists
}

# tiling
TARGET_MPP   = 0.5
TILE_PX      = 224
MAX_TILES    = 2048
BATCH_TOK    = 64

_PAT_RE = re.compile(r"(C3[NL]-\d{5})", re.I)
def norm_token(x: str|None) -> str|None:
    if x is None: return None
    s = str(x)
    m = _PAT_RE.search(s)
    if m: return m.group(1).upper()
    parts = Path(s).stem.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}".upper()
    return None

def read_manifest(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    slide_c = cols.get("slide") or cols.get("slide_path") or cols.get("path") or cols.get("file")
    pat_c   = cols.get("patient")
    if slide_c is None or pat_c is None:
        raise SystemExit(f"[FATAL] Manifest needs slide+patient. Got: {list(df.columns)}")
    df = df.rename(columns={slide_c:"slide", pat_c:"patient"})
    df["slide"] = df["slide"].astype(str)
    df["slide_key"] = df["slide"].map(lambda s: Path(s).name)
    df["patient"] = df["patient"].map(norm_token)
    df = df.dropna(subset=["patient"])
    return df[["slide","slide_key","patient"]]

def get_openclip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=DEVICE)
    model.eval()
    return model, preprocess

def load_osfm_projection(ckpt_path: Path, in_dim=768) -> nn.Module:
    if not ckpt_path.is_file():
        raise SystemExit(f"[FATAL] OSFM student checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj.get("state_dict", obj)
    # candidates = any 2D weight with shape (*, in_dim)
    cand = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == in_dim:
            cand.append((k, v.shape[0], v.shape[1]))
    if not cand:
        raise SystemExit("[FATAL] No 768-in Linear weight found in openslidefm_student.pt. "
                         "We need a projection head to map CLIP(768) -> OSFM(D).")
    # pick the largest out_dim
    cand.sort(key=lambda t: t[1], reverse=True)
    w_key, out_dim, _ = cand[0]
    b_key = w_key.replace("weight","bias")
    W = sd[w_key].float()
    b = sd.get(b_key, torch.zeros(out_dim, dtype=torch.float32))
    proj = nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        proj.weight.copy_(W)
        proj.bias.copy_(b)
    print(f"[OSFM-PROJ] using {w_key} → out_dim={out_dim}")
    return proj.to(DEVICE).eval()

def _get_mpps(sld):
    try:
        mx = float(sld.properties.get("openslide.mpp-x", "0") or "0")
        my = float(sld.properties.get("openslide.mpp-y", "0") or "0")
        if mx>0 and my>0: return (mx+my)/2.0
    except: pass
    # fallback: approximate by level downsample
    return None

def tile_coords(sld, target_mpp=0.5, tile_px=224, max_tiles=2048):
    base_w, base_h = sld.dimensions
    # choose level with closest mpp (or 0)
    level = 0
    mpp = _get_mpps(sld)
    if mpp is not None:
        # choose level whose downsample makes mpp closest to target
        best, best_diff = 0, 1e9
        for lv in range(sld.level_count):
            ds = sld.level_downsamples[lv]
            est = mpp * ds
            diff = abs(est - target_mpp)
            if diff < best_diff:
                best, best_diff = lv, diff
        level = best
    # dimensions at chosen level
    lw, lh = sld.level_dimensions[level]
    # stride = tile_px (no overlap)
    xs = list(range(0, lw - tile_px + 1, tile_px))
    ys = list(range(0, lh - tile_px + 1, tile_px))
    coords = [(x, y, level) for y in ys for x in xs]
    if len(coords) > max_tiles:
        # uniform subsample to max_tiles
        idx = np.linspace(0, len(coords)-1, num=max_tiles, dtype=int)
        coords = [coords[i] for i in idx]
    return coords

def encode_slide_to_osfm_feat(slide_path: Path, clip_model, preprocess, proj: nn.Module) -> np.ndarray:
    import openslide, PIL.Image as Image
    with openslide.OpenSlide(str(slide_path)) as sld:
        coords = tile_coords(sld, TARGET_MPP, TILE_PX, MAX_TILES)
        toks = []
        batch = []
        for (x, y, lv) in coords:
            # map level coords to level=0
            ds = sld.level_downsamples[lv]
            src_x = int(round(x * ds))
            src_y = int(round(y * ds))
            region = sld.read_region((src_x, src_y), 0, (TILE_PX, TILE_PX)).convert("RGB")
            batch.append(preprocess(region).unsqueeze(0))
            if len(batch) == BATCH_TOK:
                imgs = torch.cat(batch, dim=0).to(DEVICE)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                    z = clip_model.encode_image(imgs)            # [B,768]
                    z = z.float()
                    z = proj(z)                                  # [B,D]
                toks.append(z.detach().cpu().numpy())
                batch = []
        if batch:
            imgs = torch.cat(batch, dim=0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                z = clip_model.encode_image(imgs)
                z = proj(z.float())
            toks.append(z.detach().cpu().numpy())
    if not toks:
        return None
    Z = np.concatenate(toks, axis=0)           # [N,D]
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0, ddof=0)
    feat = np.concatenate([mu, sd], axis=0)    # [2D]
    return feat

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def evaluate(PAT: pd.DataFrame, LAB: pd.DataFrame, out_dir: Path, tag: str):
    J = PAT.merge(LAB, on="patient", how="inner")
    if J.empty or J["HRD_binary"].nunique() < 2:
        raise SystemExit("[FATAL] Only one class after join.")
    zcols = [c for c in J.columns if c.startswith("z")]
    X = J[zcols].values
    y = J["HRD_binary"].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:,1]
    auc = roc_auc_score(y, oof)
    ap  = average_precision_score(y, oof)
    br  = brier_score_loss(y, oof)
    out_dir.mkdir(parents=True, exist_ok=True)
    J.assign(pred=oof).to_csv(out_dir / f"{tag}_preds.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / f"{tag}_metrics.json","w",encoding="utf-8") as f:
        json.dump({"AUC":float(auc),"AP":float(ap),"Brier":float(br),"n":int(len(y))}, f, indent=2)
    print(f"[METRICS] {tag}: AUC={auc:.3f}  AP={ap:.3f}  Brier={br:.4f}  (n={len(y)})")

print(f"[ENV] device={DEVICE}")
# 1) OpenCLIP backbone
clip_model, preprocess = get_openclip()
# 2) Projection head from OSFM checkpoint
proj = load_osfm_projection(CKPT_STUDENT, in_dim=768)

summary = {}

for name, cfg in COHORTS.items():
    print("\n" + "="*8 + f" {name.upper()} (OSFM-proj) " + "="*8)
    M = read_manifest(cfg["manifest"])
    L = load_labels(cfg["labels"])
    out_dir = Path(cfg["out_dir"])
    # Encode per-slide → aggregate per-patient
    rows = []
    for slide, pat in tqdm(M[["slide","patient"]].itertuples(index=False, name=None), total=len(M)):
        feat = encode_slide_to_osfm_feat(Path(slide), clip_model, preprocess, proj)
        if feat is None: 
            continue
        rows.append({"patient": pat, **{f"z{idx:04d}": v for idx, v in enumerate(feat)}})
    if not rows:
        print(f"[SKIP] {name}: no slide features extracted.")
        summary[name] = {"error": "no slide features extracted"}
        continue
    SL = pd.DataFrame(rows)
    # patient mean over slides
    zcols = [c for c in SL.columns if c.startswith("z")]
    PAT = SL.groupby("patient")[zcols].mean().reset_index()
    PAT.to_parquet(out_dir / f"cptac_{name}_openslidefm_patient_embeddings.parquet", index=False)
    # Evaluate
    try:
        evaluate(PAT, L, out_dir, f"cptac_{name}_openslidefm")
        summary[name] = {"patients": int(len(PAT))}
    except SystemExit as e:
        print(f"[SKIP] {name}: {e}")
        summary[name] = {"error": str(e)}

print("\n[SUMMARY]")
print(json.dumps(summary, indent=2))


# SECTION 18: PTRC-HGSOC PLATINUM RESISTANCE PREDICTION

# === PTRC-HGSOC — robust end-to-end (WSI→manifest/labels→OSFM features→eval) ===
# Fixes: (1) parse WSI name into (Image Name, Image ID); (2) join clinical on those; (3) DO NOT TCGA-normalize patients.


import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.isotonic import IsotonicRegression

WSI_DIR       = Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\PKG - PTRC-HGSOC\data")
CLIN_PATH     = Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\PKG - PTRC-HGSOC\PTRC-HGSOC_List_clincal_data.xlsx")
CKPT_STUDENT  = Path(r"D:\个人文件夹\Sanwal\OpenSlide\models\openslidefm_student.pt")
OUT_DIR       = Path(r"D:\个人文件夹\Sanwal\OpenSlide\results\ptrc_hgsoc_osfm")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MPP  = 0.5
TILE_PX     = 224
MAX_TILES   = 2048
BATCH_TOK   = 64
SEED        = 1337
LABEL_COL   = "platinum_refractory"   # 1 = refractory/resistant, 0 = sensitive

np.random.seed(SEED); torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_int(x):
    try: 
        if pd.isna(x): return np.nan
        return int(str(x).strip())
    except: 
        return np.nan

def read_clinical(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in [".xlsx",".xls"]: C = pd.read_excel(p)
    elif p.suffix.lower()==".csv": C = pd.read_csv(p)
    else: raise SystemExit(f"[FATAL] Unsupported clinical file: {p}")
    # Normalize key columns
    cols = {c.lower(): c for c in C.columns}
    im_name = cols.get("image name") or cols.get("image_name") or cols.get("imagename")
    im_id   = cols.get("image id")   or cols.get("image_id")   or cols.get("imageid")
    pat_col = cols.get("patient")    or cols.get("patient id") or cols.get("patient_id")
    if not (im_name and im_id and pat_col):
        raise SystemExit("[FATAL] Clinical must have 'Image Name', 'Image ID', and 'Patient' columns.")
    C["ImageName_int"] = C[im_name].map(_safe_int)
    C["ImageID_int"]   = C[im_id].map(_safe_int)
    C["Patient_raw"]   = C[pat_col]
    # Primary label from Tumor response if present
    resp = cols.get("tumor response") or cols.get("tumor respons") or cols.get("response")
    if resp:
        v = C[resp].astype(str).str.lower().str.strip()
        C[LABEL_COL] = v.map(
            lambda s: 1 if s in {"refractory","resistant","pd","progressive disease"} 
            else (0 if s in {"sensitive","cr","complete response","pr","partial response"} else np.nan)
        ).astype("Int64")
    # Fallbacks
    pr = cols.get("platinum_refractory") or cols.get("platinum refractory")
    if (not resp) or C[LABEL_COL].isna().all():
        if pr:
            v = C[pr].astype(str).str.lower().str.strip()
            C[LABEL_COL] = v.map(lambda s: 1 if s in {"1","true","yes","y","refractory","resistant"} else (0 if s in {"0","false","no","n","sensitive"} else np.nan)).astype("Int64")
    pfi = cols.get("pfi") or cols.get("pfi_months") or cols.get("platinum free interval") or cols.get("platinum_free_interval")
    if (LABEL_COL not in C) or C[LABEL_COL].isna().all():
        if pfi:
            C[LABEL_COL] = (pd.to_numeric(C[pfi], errors="coerce") < 6).astype("Int64")
    return C

def build_manifest_from_wsi(wsi_dir: Path) -> pd.DataFrame:
    rx = re.compile(r"^(?P<img>\d+)_(?P<iid>\d+)_.*\.(svs|tif|tiff|ndpi|mrxs|scn|bif|qptiff)$", re.I)
    rows = []
    for p in wsi_dir.glob("*.svs"):
        m = rx.match(p.name)
        if not m: continue
        rows.append({
            "slide": str(p),
            "slide_key": p.name,
            "ImageName_int": int(m.group("img")),
            "ImageID_int":   int(m.group("iid"))
        })
    # Also search other formats just in case
    for ext in (".tif",".tiff",".ndpi",".mrxs",".scn",".bif",".qptiff"):
        for p in wsi_dir.glob(f"*{ext}"):
            m = rx.match(p.name)
            if not m: continue
            rows.append({
                "slide": str(p),
                "slide_key": p.name,
                "ImageName_int": int(m.group("img")),
                "ImageID_int":   int(m.group("iid"))
            })
    M = pd.DataFrame(rows)
    print(f"[MANIFEST] slides={len(M)}")
    if M.empty:
        raise SystemExit("[FATAL] No WSIs matched the expected 'ImageName_ImageID_*' pattern.")
    return M

def ece(y_true, y_prob, m=10):
    bins = np.linspace(0, 1, m+1); idx = np.digitize(y_prob, bins) - 1
    e, total = 0.0, len(y_true)
    for b in range(m):
        mask = (idx==b); nb = mask.sum()
        if nb == 0: continue
        conf, acc = y_prob[mask].mean(), y_true[mask].mean()
        e += (nb/total) * abs(acc - conf)
    return float(e)

def ci_boot(y, p, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    aucs, aps = [], []
    for _ in range(B):
        samp = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                               rng.choice(neg, size=len(neg), replace=True)])
        yy, pp = y[samp], p[samp]
        try: aucs.append(roc_auc_score(yy, pp))
        except: pass
        try: aps.append(average_precision_score(yy, pp))
        except: pass
    A = (np.percentile(aucs,2.5), np.percentile(aucs,97.5)) if aucs else (np.nan,np.nan)
    P = (np.percentile(aps, 2.5), np.percentile(aps, 97.5)) if aps  else (np.nan,np.nan)
    return tuple(map(float,A)), tuple(map(float,P))

def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p); return float(thr[np.argmax(tpr - fpr)])

def conf_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else np.nan
    spec = tn/(tn+fp) if (tn+fp) else np.nan
    ppv  = tp/(tp+fp) if (tp+fp) else np.nan
    npv  = tn/(tn+fn) if (tn+fn) else np.nan
    bal  = (sens+spec)/2 if np.isfinite(sens) and np.isfinite(spec) else np.nan
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, balanced_accuracy=bal, predicted_positives=int(yhat.sum()))

def thr_for(y, p, crit="NPV", target=0.95):
    uniq = np.unique(p)[::-1]; best_thr, best_val = None, -np.inf
    for thr in uniq:
        m = conf_metrics(y, p, thr); val = m["npv"] if crit=="NPV" else m["ppv"]
        if np.isfinite(val) and val >= target: return float(thr)
        if np.isfinite(val) and val > best_val: best_val, best_thr = val, float(thr)
    return best_thr

def _mpp(sld):
    try:
        mx = float(sld.properties.get("openslide.mpp-x","0") or "0")
        my = float(sld.properties.get("openslide.mpp-y","0") or "0")
        if mx>0 and my>0: return (mx+my)/2.0
    except: pass
    return None

def encode_slide(slide_path: Path, clip_model, preprocess, proj: nn.Module):
    with openslide.OpenSlide(str(slide_path)) as sld:
        coords = tile_coords(sld, TARGET_MPP, TILE_PX, MAX_TILES)
        toks, batch = [], []
        for (x,y,lv) in coords:
            ds = sld.level_downsamples[lv]
            src_x, src_y = int(round(x*ds)), int(round(y*ds))
            img = sld.read_region((src_x,src_y), 0, (TILE_PX, TILED_PX if False else TILE_PX)).convert("RGB")
            batch.append(preprocess(img).unsqueeze(0))
            if len(batch)==BATCH_TOK:
                imgs = torch.cat(batch,0).to(DEVICE)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                    z = clip_model.encode_image(imgs); z = proj(z.float())
                toks.append(z.detach().cpu().numpy()); batch=[]
        if batch:
            imgs = torch.cat(batch,0).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                z = clip_model.encode_image(imgs); z = proj(z.float())
            toks.append(z.detach().cpu().numpy())
    if not toks: return None
    Z = np.concatenate(toks,0); return np.concatenate([Z.mean(0), Z.std(0, ddof=0)], 0)

print(f"[CLIN] using: {CLIN_PATH}")
clin = read_clinical(CLIN_PATH)

M = build_manifest_from_wsi(WSI_DIR)

# Join manifest with clinical by (Image Name, Image ID)
Jm = M.merge(
    clin[["ImageName_int","ImageID_int","Patient_raw", LABEL_COL]],
    on=["ImageName_int","ImageID_int"], how="left"
)
# Patient string (stable ID for grouping)
Jm["patient"] = Jm["Patient_raw"].astype(str).str.strip()
n_before = len(Jm); Jm = Jm.dropna(subset=["patient"])
if Jm["patient"].isna().all():
    raise SystemExit("[FATAL] Could not assign patients from clinical (join keys mismatch).")
print(f"[JOIN] matched slides={Jm['patient'].notna().sum()} / {n_before}")

# Patient-level labels (from clinical)
L = clin.dropna(subset=[LABEL_COL]).copy()
L["patient"] = L["Patient_raw"].astype(str).str.strip()
L = L[["patient", LABEL_COL]].drop_duplicates(subset=["patient"])
if L.empty: 
    raise SystemExit("[FATAL] No labels derived from clinical.")

print(f"[DIAG] slides={len(Jm)}  patients(manifest)={Jm['patient'].nunique()}  patients(labels)={L['patient'].nunique()}  overlap={len(set(Jm['patient']) & set(L['patient']))}")
print("[DIAG] label balance (patients):", dict(pd.Series(L[LABEL_COL].astype(int)).value_counts().sort_index().to_dict()))

# Save manifest & labels for provenance
man_path = OUT_DIR / "ptrc_hgsoc_manifest.csv"; Jm[["slide","slide_key","patient"]].to_csv(man_path, index=False, encoding="utf-8-sig")
lab_path = OUT_DIR / "ptrc_hgsoc_patient_labels_platinum.csv"; L.to_csv(lab_path, index=False, encoding="utf-8-sig")
print(f"[OK] manifest → {man_path}")
print(f"[OK] labels   → {lab_path}")

emb_parquet = OUT_DIR / "ptrc_hgsoc_openslidefm_patient_embeddings.parquet"
emb_csv     = OUT_DIR / "ptrc_hgsoc_openslidefm_patient_embeddings.csv"

if emb_parquet.exists() or emb_csv.exists():
    print("[SKIP] Patient embeddings exist — loading.")
    PAT = pd.read_parquet(emb_parquet) if emb_parquet.exists() else pd.read_csv(emb_csv)
else:
    print(f"[ENV] device={DEVICE}")
    clip_model, preprocess = get_openclip()
    proj = load_osfm_projection(CKPT_STUDENT, in_dim=768)

    rows = []
    # Only slides with labeled patients
    pats_keep = set(L["patient"])
    Meff = Jm[Jm["patient"].isin(pats_keep)][["slide","patient"]].reset_index(drop=True)
    for slide, pat in tqdm(Meff.itertuples(index=False, name=None), total=len(Meff)):
        feat = encode_slide(Path(slide), clip_model, preprocess, proj)
        if feat is None: continue
        rows.append({"patient": pat, **{f"z{idx:04d}": v for idx,v in enumerate(feat)}})
    if not rows: raise SystemExit("[FATAL] No features extracted.")
    SL = pd.DataFrame(rows)
    zcols = [c for c in SL.columns if c.startswith("z")]
    PAT = SL.groupby("patient")[zcols].mean().reset_index()
    try:
        PAT.to_parquet(emb_parquet, index=False); print(f"[OK] patient embeddings (parquet) → {emb_parquet}")
    except Exception as e:
        PAT.to_csv(emb_csv, index=False, encoding="utf-8-sig"); print(f"[WARN] parquet failed ({e}); wrote CSV → {emb_csv}")

J = PAT.merge(L, on="patient", how="inner")
assert not J.empty and J[LABEL_COL].nunique()==2, "[FATAL] Only one class after join."
zcols = [c for c in J.columns if c.startswith("z")]
X = J[zcols].values; y = J[LABEL_COL].astype(int).values
n, pos, neg = len(y), int(y.sum()), int((y==0).sum()); prev = pos/n

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(n, float)
for tr, te in skf.split(X, y):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:,1]

auc_u = roc_auc_score(y, oof); ap_u = average_precision_score(y, oof); br_u = brier_score_loss(y, oof); ece_u = ece(y, oof)
auc_ci_u, ap_ci_u = ci_boot(y, oof)

# CV-on-scores calibration
p_platt = np.zeros_like(oof); p_iso = np.zeros_like(oof)
for tr, te in skf.split(oof.reshape(-1,1), y):
    lr = LogisticRegression(max_iter=1000).fit(oof[tr].reshape(-1,1), y[tr]); p_platt[te] = lr.predict_proba(oof[te].reshape(-1,1))[:,1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof[tr], y[tr]); p_iso[te] = iso.transform(oof[te])

def summarize(p):
    return (roc_auc_score(y,p), average_precision_score(y,p), brier_score_loss(y,p), ece(y,p), *ci_boot(y,p))
A_p, AP_p, Br_p, Ec_p, (Alo_p,Ahi_p), (APlo_p, Aphi_p) = *summarize(p_platt),
A_i, AP_i, Br_i, Ec_i, (Alo_i,Ahi_i), (APlo_i, Aphi_i) = *summarize(p_iso),

# Save preds + metrics
preds_csv = OUT_DIR / "ptrc_hgsoc_openslidefm_preds.csv"
pd.DataFrame({"patient": J["patient"], "pred": oof, LABEL_COL: y}).to_csv(preds_csv, index=False, encoding="utf-8-sig")
with open(OUT_DIR / "ptrc_hgsoc_openslidefm_metrics.json","w",encoding="utf-8") as f:
    json.dump({"AUC":float(auc_u),"AP":float(ap_u),"Brier":float(br_u),"n":int(n)}, f, indent=2)
print(f"[METRICS] PTRC-HGSOC (uncalibrated): AUC={auc_u:.3f}  AP={ap_u:.3f}  Brier={br_u:.4f}  (n={n})")

# Table 2
tbl2 = pd.DataFrame([
    dict(Cohort="PTRC-HGSOC", Variant="Uncalibrated",            n=n,pos=pos,neg=neg,prevalence=round(prev,4),
         AUC=round(auc_u,3), AUC_CI_low=round(auc_ci_u[0],3), AUC_CI_high=round(auc_ci_u[1],3),
         AP=round(ap_u,3),  AP_CI_low=round(ap_ci_u[0],3),   AP_CI_high=round(ap_ci_u[1],3),
         Brier=round(br_u,4), ECE_10bin=round(ece_u,4), Notes="5-fold OOF scores"),
    dict(Cohort="PTRC-HGSOC", Variant="Platt (CV-calibrated)",   n=n,pos=pos,neg=neg,prevalence=round(prev,4),
         AUC=round(A_p,3),  AUC_CI_low=round(Alo_p,3), AUC_CI_high=round(Ahi_p,3),
         AP=round(AP_p,3),  AP_CI_low=round(APlo_p,3), AP_CI_high=round(Aphi_p,3),
         Brier=round(Br_p,4), ECE_10bin=round(Ec_p,4), Notes="5-fold CV-on-scores"),
    dict(Cohort="PTRC-HGSOC", Variant="Isotonic (CV-calibrated)",n=n,pos=pos,neg=neg,prevalence=round(prev,4),
         AUC=round(A_i,3),  AUC_CI_low=round(Alo_i,3), AUC_CI_high=round(Ahi_i,3),
         AP=round(AP_i,3),  AP_CI_low=round(APlo_i,3), AP_CI_high=round(Aphi_i,3),
         Brier=round(Br_i,4), ECE_10bin=round(Ec_i,4), Notes="5-fold CV-on-scores"),
])
t2_path = OUT_DIR / "table2_primary_metrics_with_calibration.csv"; tbl2.to_csv(t2_path, index=False, encoding="utf-8-sig")
print("\n=== PTRC-HGSOC Table 2 ==="); print(tbl2.to_string(index=False))

# Table 3
def op_rows(name, p):
    yj = youden_thr(y,p); r1 = conf_metrics(y,p,yj)
    n95 = thr_for(y,p,"NPV",0.95); r2 = conf_metrics(y,p,n95) if n95 is not None else None
    p60 = thr_for(y,p,"PPV",0.60); r3 = conf_metrics(y,p,p60) if p60 is not None else None
    def row(rule, thr, r):
        return dict(Cohort="PTRC-HGSOC", Variant=name, Rule=rule, Threshold=(round(thr,4) if thr is not None else None),
                    Sensitivity=(round(r["sensitivity"],3) if r else None),
                    Specificity=(round(r["specificity"],3) if r else None),
                    PPV=(round(r["ppv"],3) if r else None), NPV=(round(r["npv"],3) if r else None),
                    BalancedAccuracy=(round(r["balanced_accuracy"],3) if r else None),
                    PredictedPositives=(r["predicted_positives"] if r else None), n=n,pos=pos,neg=neg)
    return [row("Youden-J", yj, r1), row("NPV≥0.95", n95, r2), row("PPV≥0.60", p60, r3)]

tbl3 = pd.DataFrame(op_rows("Uncalibrated", oof) + op_rows("Platt (CV-cal)", p_platt) + op_rows("Isotonic (CV-cal)", p_iso))
t3_path = OUT_DIR / "table3_operating_points_with_calibration.csv"; tbl3.to_csv(t3_path, index=False, encoding="utf-8-sig")
print("\n=== PTRC-HGSOC Table 3 ==="); print(tbl3.to_string(index=False))

# Curves
fpr,tpr,thr = roc_curve(y,oof); pd.DataFrame({"fpr":fpr,"tpr":tpr,"thr":np.append(thr,np.nan)[:len(fpr)]}).to_csv(OUT_DIR/"ptrc_hgsoc_roc_curve.csv", index=False)
P,R,_ = precision_recall_curve(y,oof); pd.DataFrame({"precision":P,"recall":R}).to_csv(OUT_DIR/"ptrc_hgsoc_pr_curve.csv", index=False)
with open(OUT_DIR / "ptrc_hgsoc_eval_summary_calibrated.json","w",encoding="utf-8") as f:
    json.dump({"cohort":"PTRC-HGSOC","n":int(n),"pos":int(pos),"neg":int(neg),"prevalence":float(prev),
               "uncalibrated":{"auc":float(auc_u),"ap":float(ap_u),"brier":float(br_u),"ece10":float(ece_u)}}, f, indent=2)

print("\n✓ Wrote:")
print("  -", OUT_DIR/"ptrc_hgsoc_manifest.csv")
print("  -", OUT_DIR/"ptrc_hgsoc_patient_labels_platinum.csv")
print("  -", OUT_DIR/"ptrc_hgsoc_openslidefm_preds.csv")
print("  -", t2_path)
print("  -", t3_path)
print("  -", OUT_DIR/"ptrc_hgsoc_roc_curve.csv")
print("  -", OUT_DIR/"ptrc_hgsoc_pr_curve.csv")
print("  -", OUT_DIR/"ptrc_hgsoc_eval_summary_calibrated.json")


# SECTION 19: SURGEN MMR OFF-TARGET EVALUATION

# SurGen (SR386 / SR1482) → IHGAMP validation via OpenSlideFM
# CPU-ONLY VERSION (no CUDA touches at all)
#   - Reads .czi WSIs (SR386 / SR1482)
#   - Builds manifest
#   - Extracts OSFM-projected features from ViT-L-14 (OpenCLIP)
#   - Trains 5-fold balanced LR on MMR-loss label
#   - Saves embeddings + preds + metrics + ROC/PR curves

import os, re, json, warnings


# 🔴 FORCE CPU ONLY – DO NOT CHANGE
DEVICE = "cpu"

CKPT_STUDENT = Path(r"D:\个人文件夹\Sanwal\OpenSlide\models\openslidefm_student.pt")

# Root that contains SR386, SR1482, etc.
WSI_ROOT = Path(r"D:\个人文件夹\Sanwal\Gen")

# Label paths – adjust extension if needed
SR386_LABELS = Path(r"D:\个人文件夹\Sanwal\Gen\Labels\SR386_labels.csv")
SR1482_LABELS = Path(r"D:\个人文件夹\Sanwal\Gen\Labels\SR1482_labels.csv")

OUT_DIR = Path(r"D:\个人文件夹\Sanwal\OpenSlide\results\surgen_osfm")

TARGET_MPP = 0.5
TILE_PX = 224
MAX_TILES = 2048
BATCH_TOK = 32  # safe for CPU

WSI_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".czi")


def read_any_table(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise SystemExit(f"[FATAL] Label file not found: {p}")
    suf = p.suffix.lower()
    if suf == ".xlsx":
        return pd.read_excel(p)
    return pd.read_csv(p, sep=None, engine="python")


def _load_czi_as_rgb(path: Path) -> np.ndarray:
    """Load CZI as H×W×3 uint8 using czifile."""
    import czifile

    with czifile.CziFile(str(path)) as czi:
        img = czi.asarray()

    while img.ndim > 3:
        img = img[0]

    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = np.moveaxis(img, 0, -1)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] > 3:
        img = img[..., :3]

    if img.dtype != np.uint8:
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = ((img - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    return img


def build_surgen_manifest(wsi_root: Path) -> pd.DataFrame:
    """
    Walk WSI_ROOT and collect all WSIs.
    Patient ID pattern:
      - File: SR386_40X_HE_T003_01.czi  → case_id=3, cohort='SR386'
      - patient = 'SR386_3'
    Same for SR1482_... files.
    """
    rows = []
    for root, dirs, files in os.walk(wsi_root):
        for fn in files:
            lf = fn.lower()
            if not lf.endswith(WSI_EXTS):
                continue

            slide_path = str(Path(root) / fn)

            if "sr386" in slide_path.lower():
                cohort = "SR386"
            elif "sr1482" in slide_path.lower():
                cohort = "SR1482"
            else:
                cohort = "SURGEN"

            m = re.search(r"_T(\d+)", fn, flags=re.IGNORECASE)
            if m:
                case_id = int(m.group(1))
            else:
                m2 = re.search(r"(\d+)", fn)
                if not m2:
                    continue
                case_id = int(m2.group(1))

            patient = f"{cohort}_{case_id}"

            rows.append(
                {
                    "slide": slide_path,
                    "slide_key": fn,
                    "cohort": cohort,
                    "case_id": case_id,
                    "patient": patient,
                }
            )

    if not rows:
        raise SystemExit(f"[FATAL] No WSI files {WSI_EXTS} found under {wsi_root}")

    mf = pd.DataFrame(rows).drop_duplicates(subset=["slide"])
    print(
        f"[MANIFEST] slides={len(mf)}  patients(unique)={mf['patient'].nunique()} "
        f"| cohorts={mf['cohort'].unique().tolist()}"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mf.to_csv(OUT_DIR / "surgen_manifest.csv", index=False)
    print(f"[OK] manifest → {OUT_DIR / 'surgen_manifest.csv'}")
    return mf


def _standardise_labels(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    case_col = cols.get("case_id") or cols.get("case") or cols.get("id")
    if case_col is None:
        raise SystemExit(f"[FATAL] No case_id column in {tag} labels. Got: {list(df.columns)}")
    df["case_id"] = pd.to_numeric(df[case_col], errors="coerce")

    mmr_col = None
    if "mmr_loss_binary" in cols:
        mmr_col = cols["mmr_loss_binary"]
        df["mmr_label"] = pd.to_numeric(df[mmr_col], errors="coerce")
    elif "mmr" in cols:
        mmr_col = cols["mmr"]
        s = df[mmr_col].astype(str).str.strip().str.lower()
        df["mmr_label"] = np.where(
            s.str.contains("loss"),
            1,
            np.where(s.str.contains("no"), 0, np.nan),
        )
    elif "mmr_ihc" in cols:
        mmr_col = cols["mmr_ihc"]
        s = df[mmr_col].astype(str).str.strip().str.lower()
        df["mmr_label"] = np.where(
            s.str.contains("loss"),
            1,
            np.where(s.str.contains("no"), 0, np.nan),
        )
    else:
        raise SystemExit(f"[FATAL] No MMR column in {tag} labels. Got: {list(df.columns)}")

    out = df[["case_id", "mmr_label"]].copy()
    out = out.dropna(subset=["case_id", "mmr_label"])
    out["case_id"] = out["case_id"].astype(int)
    out["mmr_label"] = out["mmr_label"].astype(int)
    out["patient"] = out["case_id"].map(lambda x: f"{tag}_{x}")
    return out


def load_surgen_labels(sr386_path: Path, sr1482_path: Path) -> pd.DataFrame:
    frames = []
    for path, tag in [(sr386_path, "SR386"), (sr1482_path, "SR1482")]:
        if path is None or not path.exists():
            print(f"[WARN] labels for {tag} not found at: {path} (skipping)")
            continue
        df_raw = read_any_table(path)
        print(f"[LABELS] {tag}: rows={len(df_raw)}  cols={list(df_raw.columns)}")
        frames.append(_standardise_labels(df_raw, tag))

    if not frames:
        raise SystemExit("[FATAL] No label tables loaded for SurGen.")

    lab = pd.concat(frames, ignore_index=True)
    lab = lab.drop_duplicates(subset=["patient"])
    print(f"[LABELS] total unique patients with MMR labels: {lab['patient'].nunique()}")
    print(f"[LABELS] class balance: {lab['mmr_label'].value_counts().to_dict()}")
    return lab[["patient", "mmr_label"]]


def evaluate_surgen(PAT: pd.DataFrame, LAB: pd.DataFrame, out_dir: Path, tag: str):
    J = PAT.merge(LAB, on="patient", how="inner")
    if J.empty or J["mmr_label"].nunique() < 2:
        raise SystemExit("[FATAL] Need at least two classes after join for evaluation.")

    zcols = [c for c in J.columns if c.startswith("z")]
    X = J[zcols].values
    y = J["mmr_label"].astype(int).values

    print(f"[JOIN] patients with features+labels = {len(J)}")
    print(f"[JOIN] class balance: {pd.Series(y).value_counts().to_dict()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    oof = np.zeros_like(y, dtype=float)

    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    br = brier_score_loss(y, oof)

    print(f"[METRICS] {tag}: AUC={auc:.3f}  AP={ap:.3f}  Brier={br:.4f}  (n={len(y)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df = J[["patient"]].copy()
    preds_df["label"] = y
    preds_df["pred"] = oof
    preds_path = out_dir / f"{tag}_preds.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[OK] preds CSV → {preds_path}")

    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"AUC": float(auc), "AP": float(ap), "Brier": float(br), "n": int(len(y))}, f, indent=2)
    print(f"[OK] metrics JSON → {metrics_path}")

    fpr, tpr, _ = roc_curve(y, oof)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(out_dir / f"{tag}_roc_curve.csv", index=False)

    prec, rec, _ = precision_recall_curve(y, oof)
    pr_df = pd.DataFrame({"recall": rec, "precision": prec})
    pr_df.to_csv(out_dir / f"{tag}_pr_curve.csv", index=False)


print(f"[ENV] device={DEVICE}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Build / load manifest
manifest = build_surgen_manifest(WSI_ROOT)

# 2) Load OSFM backbone + projection (CPU only)
clip_model, preprocess = get_openclip()
proj = load_osfm_projection(CKPT_STUDENT, in_dim=768)

# 3) Encode slides → patient embeddings (mean over slides)
embed_pq = OUT_DIR / "surgen_openslidefm_patient_embeddings.parquet"

if embed_pq.is_file():
    print(f"[LOAD] Using existing patient embeddings → {embed_pq}")
    PAT = pd.read_parquet(embed_pq)
else:
    rows = []
    for slide, patient in tqdm(
        manifest[["slide", "patient"]].itertuples(index=False, name=None),
        total=len(manifest),
        desc="Encoding SurGen slides (CPU)",
    ):
        try:
            feat = encode_slide_to_osfm_feat(Path(slide), clip_model, preprocess, proj)
        except Exception as e:
            warnings.warn(f"[WARN] Error on slide {slide}: {e}. Skipping.")
            continue

        if feat is None:
            continue
        row = {"patient": patient}
        row.update({f"z{idx:04d}": v for idx, v in enumerate(feat)})
        rows.append(row)

    if not rows:
        raise SystemExit("[FATAL] No slide features extracted for SurGen.")
    SL = pd.DataFrame(rows)
    zcols = [c for c in SL.columns if c.startswith("z")]
    PAT = SL.groupby("patient")[zcols].mean().reset_index()
    PAT.to_parquet(embed_pq, index=False)
    print(f"[OK] patient embeddings (parquet) → {embed_pq}")

# 4) Load labels and evaluate
LAB = load_surgen_labels(SR386_LABELS, SR1482_LABELS)
evaluate_surgen(PAT, LAB, OUT_DIR, tag="surgen_openslidefm")

print("\n[DONE] SurGen (MMR) validation via OSFM complete.")


# SECTION 20: TP53 PATHWAY SPECIFICITY ANALYSIS

# === Is TP53 signal independent of HRD, or mediated through it? ===
from scipy.stats import pearsonr

sub = preds[preds["TP53_mut"].notna()].copy()

# 1. How correlated are TP53 and HRD themselves?
r_tp53_hrd, p_tp53_hrd = pearsonr(sub["TP53_mut"].astype(float), sub["HRD_cont"])
print(f"TP53 vs HRD_continuous:  r={r_tp53_hrd:.3f} (p={p_tp53_hrd:.2e})")

# 2. IHGAMP AUC for TP53 among HRD-NEGATIVE patients only
hrd_neg = sub[sub["HRD_top20"] == 0]
tp53_pos_in_neg = int(hrd_neg["TP53_mut"].sum())
tp53_neg_in_neg = len(hrd_neg) - tp53_pos_in_neg
print(f"\nHRD-negative patients: n={len(hrd_neg)}, TP53+={tp53_pos_in_neg}, TP53-={tp53_neg_in_neg}")
if tp53_pos_in_neg >= 5 and tp53_neg_in_neg >= 5:
    auc_tp53_hrdneg = roc_auc_score(hrd_neg["TP53_mut"].astype(int), hrd_neg["ihgamp_prob"])
    print(f"IHGAMP AUC for TP53 (among HRD- only) = {auc_tp53_hrdneg:.3f}")
else:
    print("Too few events")

# 3. IHGAMP AUC for TP53 among HRD-POSITIVE patients only
hrd_pos = sub[sub["HRD_top20"] == 1]
tp53_pos_in_pos = int(hrd_pos["TP53_mut"].sum())
tp53_neg_in_pos = len(hrd_pos) - tp53_pos_in_pos
print(f"\nHRD-positive patients: n={len(hrd_pos)}, TP53+={tp53_pos_in_pos}, TP53-={tp53_neg_in_pos}")
if tp53_pos_in_pos >= 5 and tp53_neg_in_pos >= 5:
    auc_tp53_hrdpos = roc_auc_score(hrd_pos["TP53_mut"].astype(int), hrd_pos["ihgamp_prob"])
    print(f"IHGAMP AUC for TP53 (among HRD+ only) = {auc_tp53_hrdpos:.3f}")
else:
    print("Too few events")

# 4. Partial correlation: IHGAMP vs TP53, controlling for HRD
from sklearn.linear_model import LinearRegression
X_hrd = sub["HRD_cont"].values.reshape(-1,1)
resid_ihgamp = sub["ihgamp_prob"].values - LinearRegression().fit(X_hrd, sub["ihgamp_prob"].values).predict(X_hrd)
resid_tp53   = sub["TP53_mut"].astype(float).values - LinearRegression().fit(X_hrd, sub["TP53_mut"].astype(float).values).predict(X_hrd)
r_partial, p_partial = pearsonr(resid_ihgamp, resid_tp53)
print(f"\nPartial correlation (IHGAMP vs TP53, controlling for HRD): r={r_partial:.3f} (p={p_partial:.2e})")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print(f"  Raw IHGAMP-TP53 correlation:     r=0.298")
print(f"  Partial (controlling for HRD):   r={r_partial:.3f}")
if abs(r_partial) < 0.15:
    print(f"  → TP53 signal is MEDIATED through HRD (drops to near-zero)")
    print(f"  → Paper's HRD-specificity claim HOLDS")
elif abs(r_partial) < 0.25:
    print(f"  → Partial TP53 signal remains — shared morphological features")
    print(f"  → HRD is primary driver, TP53 is secondary/confounded")
else:
    print(f"  → Independent TP53 signal — model captures broader instability")

# SECTION 21: UCEC MMR OFF-TARGET & PTRC GENOMIC HRD CHECK

# REVIEWER RESPONSE — Major 4 + Major 7
# Major 4: PTRC-HGSOC genomic HRD evaluation (not just platinum resistance)
# Major 7: MMR-deficient endometrial carcinoma (CPTAC-UCEC) off-target
#
# Requires: FROZEN_PIPE, FROZEN_PLATT from the revision cell already in memory
#           OR rebuilds them from scratch if not available.

import os, re, json, warnings, time
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
warnings.filterwarnings("ignore")

DL_ROOT     = Path(r"D:\个人文件夹\Sanwal\DL_V2")
LABELS_PQ   = DL_ROOT / "artifacts" / "labels" / "labels.parquet"
TCGA_EMB    = DL_ROOT / "artifacts" / "embeddings" / "patient_means_clean_run_20250908_020405_emb_openclip_vitb16_turbo.parquet"
OUT         = Path(r"D:\个人文件夹\Sanwal\IHGAMP_Revision")
OUT.mkdir(parents=True, exist_ok=True)

# PTRC-HGSOC
PTRC_CLINICAL = Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\PKG - PTRC-HGSOC\PTRC-HGSOC_List_clincal_data.xlsx")
PTRC_EMB_DIR  = Path(r"D:\个人文件夹\Sanwal\OpenSlide\results\ptrc_hgsoc_osfm")

# CPTAC-UCEC
UCEC_EMB_DIR  = Path(r"D:\个人文件夹\Sanwal\OpenSlide\results\sscc\cptac_ucec_v2_osfm")
UCEC_LABEL_CANDIDATES = [
    Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC\labels\cptac_ucec_el_nahhas.csv"),
    Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC\labels\cptac_ucec_el_nahhas_thr42.csv"),
]

# CPTAC-UCEC clinical data (may contain MSI status)
UCEC_CLINICAL_CANDIDATES = [
    Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC\clinical"),
    Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC"),
    DL_ROOT / "results" / "cptac_FINAL_CORRECT_extraction" / "UCEC",
]

# Hyperparams
PCA_N = 384; RIDGE_ALPHA = 30.0; TOP_FRAC = 0.20; SEED = 42; BOOT_N = 2000

def cptac_token(x):
    s = str(x)
    m = re.search(r"(C3[NL]-\d{5})", s)
    if m: return m.group(1).upper()
    parts = Path(s).stem.split("-")
    if len(parts) >= 2: return f"{parts[0]}-{parts[1]}".upper()
    return s

def boot_ci(y, p, fn, B=BOOT_N, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, int), np.asarray(p, float)
    if len(np.unique(y)) < 2: return np.nan, np.nan, np.nan
    pt = float(fn(y, p))
    vals = []
    for _ in range(B):
        idx = rng.choice(len(y), len(y), replace=True)
        try: vals.append(float(fn(y[idx], p[idx])))
        except: pass
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return pt, float(lo), float(hi)

def find_emb_parquet(folder):
    if not folder.exists(): return None
    for pat in ["*patient*embed*.parquet", "*patient*.parquet", "*.parquet"]:
        cands = sorted(folder.glob(pat), key=lambda p: p.stat().st_size, reverse=True)
        if cands: return cands[0]
    return None

def detect_feat_cols(df):
    for prefix in ["feature_", "z", "f"]:
        cols = [c for c in df.columns if str(c).startswith(prefix) and c != "fold"]
        if len(cols) >= 64: return cols
    skip = {"patient","patient_id","cancer_type","site_code","slide","slide_key",
            "label","pred","fold","_pid","hrd_bin","HRD_binary"}
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in skip]

RESULTS = {}

if "FROZEN_PIPE" not in dir() or FROZEN_PIPE is None:
    print("Rebuilding frozen TCGA model from disk...")
    X_tcga = pd.read_parquet(TCGA_EMB)
    if "patient" in X_tcga.columns: X_tcga = X_tcga.set_index("patient")
    X_tcga.index = X_tcga.index.astype(str).str.upper().str.slice(0, 12)
    feat_cols_tcga = [c for c in X_tcga.columns if pd.api.types.is_numeric_dtype(X_tcga[c])]
    X_tcga = X_tcga[feat_cols_tcga]

    L = pd.read_parquet(LABELS_PQ)
    L["patient"] = L["patient"].astype(str).str.upper().str.slice(0, 12)
    L = L.drop_duplicates("patient").set_index("patient")
    common = sorted(set(X_tcga.index) & set(L.index))
    X_tcga = X_tcga.loc[common]; L = L.loc[common].copy()
    has_hrd = L["HRD"].notna()
    idx_tr = L.index[(L["split"] == "train") & has_hrd]
    thr_tcga = np.nanpercentile(L.loc[idx_tr, "HRD"].values, 100*(1-TOP_FRAC))
    L["HRD_top20"] = (L["HRD"] >= thr_tcga).astype(int)

    def get_X(idx): return X_tcga.loc[idx].values.astype(np.float32)
    pca_n = min(PCA_N, get_X(idx_tr).shape[1]-1, get_X(idx_tr).shape[0]-1)
    FROZEN_PIPE = Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=pca_n, random_state=SEED)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
    ]).fit(get_X(idx_tr), L.loc[idx_tr, "HRD"].astype(float).values)
    z_tr = FROZEN_PIPE.predict(get_X(idx_tr)).reshape(-1,1)
    FROZEN_PLATT = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
        z_tr, L.loc[idx_tr, "HRD_top20"].values)
    print(f"  Rebuilt: {pca_n} PCA, thr={thr_tcga:.0f}, train={len(idx_tr)}")
else:
    print("Using frozen model already in memory.")

def frozen_predict(X_arr):
    return FROZEN_PLATT.predict_proba(FROZEN_PIPE.predict(X_arr).reshape(-1,1))[:,1]


# ║  MAJOR 4 — PTRC-HGSOC: Genomic HRD evaluation                           ║
# ║  Reviewer: "Why not show AUROC for genomic HRD in HGSOC?"               ║
print("\n" + "█"*80)
print("MAJOR 4 — PTRC-HGSOC: GENOMIC HRD EVALUATION")
print("█"*80)

try:
    # Step 1: Load PTRC clinical data and inspect for HRD/scarHRD scores
    assert PTRC_CLINICAL.exists(), f"Clinical file not found: {PTRC_CLINICAL}"
    
    C = pd.read_excel(PTRC_CLINICAL)
    print(f"\n[CLINICAL] Shape: {C.shape}")
    print(f"[CLINICAL] Columns: {list(C.columns)}")
    
    # Show first few rows for inspection
    print(f"\n[CLINICAL] First 3 rows:")
    print(C.head(3).to_string())
    
    # Search for HRD-related columns
    hrd_cols = [c for c in C.columns if any(k in c.lower() for k in 
                ["hrd","scar","loh","tai","lst","gis","genomic instability",
                 "homologous","brca","mmr","msi","dna repair"])]
    print(f"\n[HRD-RELATED COLUMNS]: {hrd_cols}")
    
    for col in hrd_cols:
        vals = C[col].dropna()
        print(f"  '{col}': {len(vals)} non-null, dtype={vals.dtype}")
        if pd.api.types.is_numeric_dtype(vals):
            print(f"    range=[{vals.min()}, {vals.max()}], mean={vals.mean():.2f}")
        else:
            print(f"    unique values: {vals.unique()[:10]}")
    
    # Step 2: Check for BRCA mutation status (can serve as HRD proxy)
    brca_cols = [c for c in C.columns if "brca" in c.lower()]
    print(f"\n[BRCA COLUMNS]: {brca_cols}")
    for col in brca_cols:
        print(f"  '{col}': {C[col].value_counts().to_dict()}")
    
    # Step 3: Try to load PTRC embeddings
    ptrc_emb_path = find_emb_parquet(PTRC_EMB_DIR)
    ptrc_preds_path = None
    for p in PTRC_EMB_DIR.glob("*preds*.csv"):
        ptrc_preds_path = p; break
    
    print(f"\n[EMB] Parquet: {ptrc_emb_path}")
    print(f"[PREDS] Existing predictions: {ptrc_preds_path}")
    
    # Step 4: If we have BRCA status, evaluate IHGAMP for BRCA prediction in HGSOC
    has_genomic_hrd = False
    
    # Try HRD score columns
    for col in hrd_cols:
        if C[col].dtype in [np.float64, np.int64, float, int]:
            vals = pd.to_numeric(C[col], errors="coerce")
            if vals.notna().sum() >= 10:
                print(f"\n  ✓ Found numeric HRD-like column: '{col}' ({vals.notna().sum()} values)")
                has_genomic_hrd = True
                hrd_col_name = col
                break
    
    # Try BRCA mutation as HRD proxy
    brca_proxy = False
    brca_col_name = None
    for col in brca_cols:
        vals = C[col].dropna()
        if len(vals) >= 10:
            brca_proxy = True
            brca_col_name = col
            print(f"\n  ✓ Found BRCA column: '{col}' ({len(vals)} values)")
            break
    
    if ptrc_preds_path and (has_genomic_hrd or brca_proxy):
        # Load existing predictions (from within-cohort CV on platinum resistance)
        preds = pd.read_csv(ptrc_preds_path)
        print(f"\n[PREDS] Shape: {preds.shape}, columns: {list(preds.columns)}")
        
        # Detect patient column in predictions
        pred_pcol = None
        for c in preds.columns:
            if preds[c].astype(str).str.match(r"[A-Z0-9]").any() and "patient" in c.lower():
                pred_pcol = c; break
        if pred_pcol is None:
            pred_pcol = preds.columns[0]  # assume first column
        
        # Detect prediction column
        pred_col = None
        for c in ["pred", "prob_hrd", "prob", "prediction", "score"]:
            if c in preds.columns: pred_col = c; break
        if pred_col is None:
            num_cols = [c for c in preds.columns if pd.api.types.is_numeric_dtype(preds[c])]
            pred_col = num_cols[-1] if num_cols else None
        
        if pred_col:
            print(f"  Using patient='{pred_pcol}', prediction='{pred_col}'")
            
            # Detect patient column in clinical
            clin_pcol = None
            for c in C.columns:
                if "patient" in c.lower() or "case" in c.lower():
                    clin_pcol = c; break
            if clin_pcol is None:
                clin_pcol = C.columns[0]
            
            # Normalize patient IDs
            def norm_ptrc(x):
                s = str(x).strip()
                # Try numeric
                try:
                    return str(int(float(s)))
                except:
                    return s
            
            preds["_pid"] = preds[pred_pcol].astype(str).map(norm_ptrc)
            C["_pid"] = C[clin_pcol].astype(str).map(norm_ptrc)
            
            if has_genomic_hrd:
                C["_hrd_score"] = pd.to_numeric(C[hrd_col_name], errors="coerce")
                merged = preds.merge(C[["_pid", "_hrd_score"]].dropna(), on="_pid", how="inner")
                if len(merged) >= 10:
                    # Binary HRD at threshold 42 (clinical standard)
                    merged["hrd_bin"] = (merged["_hrd_score"] >= 42).astype(int)
                    pos = int(merged["hrd_bin"].sum())
                    neg = len(merged) - pos
                    print(f"\n  Genomic HRD eval: n={len(merged)}, HRD+={pos}, HRD-={neg}")
                    if pos >= 3 and neg >= 3:
                        auc, lo, hi = boot_ci(merged["hrd_bin"].values, merged[pred_col].values, roc_auc_score)
                        print(f"  IHGAMP AUC for genomic HRD in HGSOC = {auc:.3f} ({lo:.3f}–{hi:.3f})")
                        RESULTS["PTRC_HGSOC_genomic_HRD"] = {
                            "auc": auc, "ci": f"{lo:.3f}–{hi:.3f}",
                            "n": len(merged), "pos": pos, "threshold": 42,
                        }
            
            if brca_proxy:
                def _brca_bin(x):
                    if pd.isna(x): return np.nan
                    s = str(x).strip().lower()
                    if s in {"1","yes","true","mutated","mut","positive","pathogenic"}: return 1
                    if s in {"0","no","false","wildtype","wt","negative","wild-type","wild type"}: return 0
                    if "mut" in s or "pathogenic" in s or "deleterious" in s: return 1
                    if "wild" in s or "negative" in s or "none" in s: return 0
                    try: return int(float(s))
                    except: return np.nan
                
                C["_brca_bin"] = C[brca_col_name].map(_brca_bin)
                merged_b = preds.merge(C[["_pid", "_brca_bin"]].dropna(), on="_pid", how="inner")
                if len(merged_b) >= 10:
                    pos_b = int(merged_b["_brca_bin"].sum())
                    neg_b = len(merged_b) - pos_b
                    print(f"\n  BRCA mutation eval: n={len(merged_b)}, BRCA+={pos_b}, BRCA-={neg_b}")
                    if pos_b >= 3 and neg_b >= 3:
                        auc_b, lo_b, hi_b = boot_ci(merged_b["_brca_bin"].values, 
                                                     merged_b[pred_col].values, roc_auc_score)
                        print(f"  IHGAMP AUC for BRCA in HGSOC = {auc_b:.3f} ({lo_b:.3f}–{hi_b:.3f})")
                        RESULTS["PTRC_HGSOC_BRCA_prediction"] = {
                            "auc": auc_b, "ci": f"{lo_b:.3f}–{hi_b:.3f}",
                            "n": len(merged_b), "pos": pos_b,
                        }
    
    if not has_genomic_hrd and not brca_proxy:
        print("\n  ⚠ No genomic HRD scores or BRCA mutation status found in PTRC clinical data.")
        print("  Available columns for reference:")
        for c in C.columns:
            print(f"    - {c}: {C[c].dtype}, non-null={C[c].notna().sum()}")
        print("\n  → Response letter should explain: PTRC-HGSOC was collected for platinum")
        print("    response analysis and does not include genomic scarHRD scores.")
        print("    HGSOC has ~71.5% HRD prevalence (TCGA), making binary classification")
        print("    challenging due to extreme class imbalance in the HRD-negative direction.")
        RESULTS["PTRC_HGSOC_genomic_HRD"] = {
            "status": "NOT_AVAILABLE",
            "reason": "No genomic HRD scores in PTRC clinical data",
        }

except Exception as ex:
    print(f"\nMAJOR 4 FAILED: {ex}")
    import traceback; traceback.print_exc()


# ║  MAJOR 7 — MMR-deficient Endometrial Carcinoma (CPTAC-UCEC)              ║
# ║  Reviewer: "Off-target should use GYN cancer like MMR-deficient UCEC"    ║
print("\n" + "█"*80)
print("MAJOR 7 — MMR-DEFICIENT ENDOMETRIAL CARCINOMA (CPTAC-UCEC)")
print("█"*80)

try:
    # Step 1: Find UCEC embeddings
    ucec_emb_path = find_emb_parquet(UCEC_EMB_DIR)
    print(f"\n[EMB] UCEC parquet: {ucec_emb_path}")
    
    # Also check for existing preds
    ucec_preds_path = None
    for p in UCEC_EMB_DIR.glob("*preds*.csv"):
        ucec_preds_path = p; break
    print(f"[PREDS] Existing: {ucec_preds_path}")
    
    # Step 2: Search for MSI/MMR labels in CPTAC-UCEC
    print(f"\n[SEARCH] Looking for MSI/MMR labels...")
    
    mmr_labels = None
    
    # Strategy A: Check existing UCEC label CSVs
    for lab_path in UCEC_LABEL_CANDIDATES:
        if lab_path.exists():
            df = pd.read_csv(lab_path)
            mmr_cols = [c for c in df.columns if any(k in c.lower() for k in 
                       ["mmr","msi","microsatellite","mismatch","mlh1","msh2","msh6","pms2",
                        "pole","molecular_subtype","subtype"])]
            if mmr_cols:
                print(f"  ✓ Found MMR-related columns in {lab_path.name}: {mmr_cols}")
                mmr_labels = df
                break
            else:
                print(f"  ✗ {lab_path.name}: no MMR columns. Cols: {list(df.columns)}")
    
    # Strategy B: Search UCEC clinical directories for any file with MSI/MMR info
    if mmr_labels is None:
        print(f"\n[SEARCH] Scanning for clinical files with MSI/MMR data...")
        for base in UCEC_CLINICAL_CANDIDATES:
            if not base.exists(): continue
            for ext in ["*.csv", "*.xlsx", "*.tsv", "*.txt"]:
                for fp in base.rglob(ext):
                    try:
                        if fp.suffix == ".xlsx":
                            df = pd.read_excel(fp, nrows=5)
                        elif fp.suffix == ".tsv":
                            df = pd.read_csv(fp, sep="\t", nrows=5)
                        else:
                            df = pd.read_csv(fp, nrows=5)
                        mmr_cols = [c for c in df.columns if any(k in c.lower() for k in
                                   ["mmr","msi","microsatellite","mismatch","mlh1","msh2",
                                    "msh6","pms2","pole","molecular","subtype"])]
                        if mmr_cols:
                            print(f"  ✓ Found in {fp.name}: {mmr_cols}")
                            if fp.suffix == ".xlsx":
                                mmr_labels = pd.read_excel(fp)
                            elif fp.suffix == ".tsv":
                                mmr_labels = pd.read_csv(fp, sep="\t")
                            else:
                                mmr_labels = pd.read_csv(fp)
                            break
                    except:
                        continue
            if mmr_labels is not None: break
    
    # Strategy C: Try CPTAC Proteogenomic data portal format
    if mmr_labels is None:
        print(f"\n[SEARCH] Looking for CPTAC UCEC proteogenomic clinical data...")
        cptac_clin_candidates = [
            Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets\UCEC"),
            Path(r"D:\个人文件夹\Sanwal\R2_V1\Validation Datasets"),
        ]
        for base in cptac_clin_candidates:
            if not base.exists(): continue
            for fp in base.rglob("*clinical*"):
                if fp.is_file() and fp.suffix in [".csv", ".xlsx", ".tsv"]:
                    try:
                        if fp.suffix == ".xlsx":
                            df = pd.read_excel(fp, nrows=3)
                        elif fp.suffix == ".tsv":
                            df = pd.read_csv(fp, sep="\t", nrows=3)
                        else:
                            df = pd.read_csv(fp, nrows=3)
                        mmr_cols = [c for c in df.columns if any(k in c.lower() for k in
                                   ["mmr","msi","microsatellite","mismatch","subtype","molecular"])]
                        if mmr_cols:
                            print(f"  ✓ Found in {fp}: {mmr_cols}")
                            if fp.suffix == ".xlsx":
                                mmr_labels = pd.read_excel(fp)
                            elif fp.suffix == ".tsv":
                                mmr_labels = pd.read_csv(fp, sep="\t")
                            else:
                                mmr_labels = pd.read_csv(fp)
                            break
                    except:
                        continue
    
    # Strategy D: Use TCGA-UCEC data from our own labels (TCGA has MSI status)
    if mmr_labels is None:
        print(f"\n[SEARCH] Trying TCGA-UCEC MSI status from mc3/TCGA annotations...")
        mc3_path = DL_ROOT / "artifacts" / "mc3.v0.2.8.PUBLIC.maf"
        if mc3_path.exists():
            print(f"  Loading mc3 for MLH1/MSH2/MSH6/PMS2 mutations in UCEC...")
            maf = pd.read_csv(mc3_path, sep="\t",
                              usecols=["Hugo_Symbol", "Tumor_Sample_Barcode"],
                              dtype=str, comment="#", low_memory=False)
            maf["patient"] = maf["Tumor_Sample_Barcode"].str.slice(0, 12)
            
            # Get UCEC patients from our labels
            L_all = pd.read_parquet(LABELS_PQ)
            ucec_patients = L_all[L_all["cancer"].str.upper() == "UCEC"]["patient"].str.upper().str.slice(0,12).unique()
            
            # MMR genes
            mmr_genes = {"MLH1", "MSH2", "MSH6", "PMS2"}
            mmr_muts = maf[(maf["Hugo_Symbol"].isin(mmr_genes)) & 
                           (maf["patient"].isin(ucec_patients))]
            mmr_patients = set(mmr_muts["patient"].unique())
            
            mmr_labels = pd.DataFrame({"patient": ucec_patients})
            mmr_labels["MMR_deficient"] = mmr_labels["patient"].isin(mmr_patients).astype(int)
            mmr_labels["_source"] = "mc3_mutation"
            
            n_def = int(mmr_labels["MMR_deficient"].sum())
            print(f"  TCGA-UCEC: {n_def} MMR-deficient (by mutation) / {len(mmr_labels)} total")
            
            if n_def < 5:
                print(f"  ⚠ Very few MMR-deficient by mutation alone.")
                print(f"    Note: MMR deficiency in UCEC is often from MLH1 promoter methylation,")
                print(f"    not somatic mutation. Mutation-based detection underestimates prevalence.")
    
    # Step 3: If we have MMR labels, run the evaluation
    if mmr_labels is not None:
        print(f"\n[EVAL] Running MMR-deficiency off-target evaluation...")
        
        # Find MMR status column
        mmr_col = None
        for c in mmr_labels.columns:
            cl = c.lower()
            if any(k in cl for k in ["mmr_deficient", "mmr_status", "mmr", "msi_status",
                                      "msi", "microsatellite"]):
                mmr_col = c
                break
        
        if mmr_col is None and "MMR_deficient" in mmr_labels.columns:
            mmr_col = "MMR_deficient"
        
        if mmr_col is None:
            # Try molecular subtype — MSI-H is typically MMR-deficient
            for c in mmr_labels.columns:
                if "subtype" in c.lower() or "molecular" in c.lower():
                    mmr_col = c
                    print(f"  Using molecular subtype column: '{c}'")
                    print(f"  Values: {mmr_labels[c].value_counts().to_dict()}")
                    break
        
        if mmr_col:
            # Binarize MMR status
            def mmr_binary(x):
                if pd.isna(x): return np.nan
                s = str(x).strip().lower()
                if s in {"1", "deficient", "mmr-d", "mmrd", "dmmr", "msi-h", "msih",
                          "msi", "high", "unstable", "positive", "yes", "true"}: return 1
                if s in {"0", "proficient", "mmr-p", "mmrp", "pmmr", "mss", "msi-l",
                          "stable", "negative", "no", "false", "low"}: return 0
                if "msi-h" in s or "deficient" in s or "unstable" in s: return 1
                if "mss" in s or "proficient" in s or "stable" in s: return 0
                try: return int(float(s))
                except: return np.nan
            
            mmr_labels["mmr_bin"] = mmr_labels[mmr_col].map(mmr_binary)
            mmr_valid = mmr_labels.dropna(subset=["mmr_bin"])
            n_def = int(mmr_valid["mmr_bin"].sum())
            n_prof = len(mmr_valid) - n_def
            print(f"  MMR-deficient: {n_def}, MMR-proficient: {n_prof}")
            
            # Find patient column in mmr_labels
            mmr_pcol = None
            for c in mmr_labels.columns:
                if "patient" in c.lower() or "case" in c.lower() or "submitter" in c.lower():
                    mmr_pcol = c; break
            if mmr_pcol is None: mmr_pcol = mmr_labels.columns[0]
            
            # Check if this is TCGA data (use frozen model) or CPTAC (use within-cohort CV)
            sample_ids = mmr_valid[mmr_pcol].astype(str).head(5).tolist()
            is_tcga = any("TCGA" in s.upper() for s in sample_ids)
            
            if is_tcga:
                print(f"\n  [TCGA-UCEC] Using frozen TCGA model (off-target evaluation)")
                mmr_valid["_pid"] = mmr_valid[mmr_pcol].astype(str).str.upper().str.slice(0,12)
                
                # Load TCGA embeddings for UCEC patients
                X_tcga_full = pd.read_parquet(TCGA_EMB)
                if "patient" in X_tcga_full.columns: X_tcga_full = X_tcga_full.set_index("patient")
                X_tcga_full.index = X_tcga_full.index.astype(str).str.upper().str.slice(0,12)
                feat_cols = [c for c in X_tcga_full.columns if pd.api.types.is_numeric_dtype(X_tcga_full[c])]
                
                common = sorted(set(X_tcga_full.index) & set(mmr_valid.set_index("_pid").index))
                print(f"  Overlap: {len(common)} patients")
                
                if len(common) >= 10:
                    X_eval = X_tcga_full.loc[common, feat_cols].values.astype(np.float32)
                    y_eval = mmr_valid.set_index("_pid").loc[common, "mmr_bin"].astype(int).values
                    pos_e = int(y_eval.sum())
                    neg_e = len(y_eval) - pos_e
                    print(f"  Aligned: n={len(common)}, MMRd={pos_e}, MMRp={neg_e}")
                    
                    if pos_e >= 3 and neg_e >= 3:
                        p_mmr = frozen_predict(X_eval)
                        auc_mmr, lo_m, hi_m = boot_ci(y_eval, p_mmr, roc_auc_score)
                        ap_mmr, _, _ = boot_ci(y_eval, p_mmr, average_precision_score)
                        print(f"\n  {'='*50}")
                        print(f"  HRD MODEL → MMR-DEFICIENT UCEC (OFF-TARGET)")
                        print(f"  {'='*50}")
                        print(f"  AUC = {auc_mmr:.3f} ({lo_m:.3f}–{hi_m:.3f})")
                        print(f"  AP  = {ap_mmr:.3f}")
                        print(f"  n={len(common)}, MMRd={pos_e}, MMRp={neg_e}")
                        
                        RESULTS["UCEC_MMR_off_target"] = {
                            "auc": auc_mmr, "ci": f"{lo_m:.3f}–{hi_m:.3f}",
                            "ap": ap_mmr, "n": len(common), "pos": pos_e,
                            "source": "TCGA-UCEC", "method": "frozen_TCGA_model_off_target",
                        }
                    else:
                        print(f"  Too few events for AUC (MMRd={pos_e}, MMRp={neg_e})")
                        RESULTS["UCEC_MMR_off_target"] = {
                            "status": "INSUFFICIENT_EVENTS", "n": len(common), "pos": pos_e
                        }
            
            else:
                print(f"\n  [CPTAC-UCEC] Using within-cohort CV (like SurGen evaluation)")
                mmr_valid["_pid"] = mmr_valid[mmr_pcol].astype(str).map(cptac_token)
                
                if ucec_emb_path:
                    E_ucec = pd.read_parquet(str(ucec_emb_path))
                    if "patient_id" in E_ucec.columns and "patient" not in E_ucec.columns:
                        E_ucec = E_ucec.rename(columns={"patient_id": "patient"})
                    
                    ucec_pcol = None
                    for c in E_ucec.columns:
                        if E_ucec[c].astype(str).str.contains(r"C3[NL]-\d{5}", regex=True).any():
                            ucec_pcol = c; break
                    if ucec_pcol is None:
                        for c in ["patient","patient_id","case_id"]: 
                            if c in E_ucec.columns: ucec_pcol = c; break
                    
                    if ucec_pcol:
                        E_ucec["_pid"] = E_ucec[ucec_pcol].astype(str).map(cptac_token)
                        ucec_feat = detect_feat_cols(E_ucec)
                        X_ucec = E_ucec.groupby("_pid")[ucec_feat].mean()
                        
                        common = sorted(set(X_ucec.index) & set(mmr_valid.set_index("_pid").index))
                        print(f"  Overlap: {len(common)}")
                        
                        if len(common) >= 10:
                            X_eval = X_ucec.loc[common].values.astype(np.float32)
                            y_eval = mmr_valid.set_index("_pid").loc[common, "mmr_bin"].astype(int).values
                            pos_e = int(y_eval.sum()); neg_e = len(y_eval) - pos_e
                            print(f"  Aligned: n={len(common)}, MMRd={pos_e}, MMRp={neg_e}")
                            
                            if pos_e >= 3 and neg_e >= 3:
                                # 5-fold CV
                                p_oof = np.full(len(y_eval), np.nan)
                                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                                for tr_ix, te_ix in cv.split(X_eval, y_eval):
                                    clf = LogisticRegression(class_weight="balanced", max_iter=1000, 
                                                             random_state=SEED)
                                    clf.fit(X_eval[tr_ix], y_eval[tr_ix])
                                    p_oof[te_ix] = clf.predict_proba(X_eval[te_ix])[:,1]
                                
                                auc_mmr = roc_auc_score(y_eval, p_oof)
                                auc_mmr_pt, lo_m, hi_m = boot_ci(y_eval, p_oof, roc_auc_score)
                                ap_mmr, _, _ = boot_ci(y_eval, p_oof, average_precision_score)
                                
                                print(f"\n  {'='*50}")
                                print(f"  HRD EMBEDDINGS → MMR-DEFICIENT UCEC (OFF-TARGET)")
                                print(f"  {'='*50}")
                                print(f"  AUC = {auc_mmr:.3f} ({lo_m:.3f}–{hi_m:.3f})")
                                print(f"  AP  = {ap_mmr:.3f}")
                                print(f"  n={len(common)}, MMRd={pos_e}, MMRp={neg_e}")
                                
                                RESULTS["UCEC_MMR_off_target"] = {
                                    "auc": auc_mmr, "ci": f"{lo_m:.3f}–{hi_m:.3f}",
                                    "ap": ap_mmr, "n": len(common), "pos": pos_e,
                                    "source": "CPTAC-UCEC",
                                    "method": "within_cohort_5fold_balanced_LR",
                                }
    
    else:
        print("\n  ⚠ No MMR/MSI labels found for UCEC.")
        print("  Searched: UCEC label CSVs, clinical directories, mc3 MAF")
        print("\n  → To get UCEC MSI status, you can:")
        print("    1. Download CPTAC-UCEC clinical data from PDC (pdc.cancer.gov)")
        print("    2. Or use cBioPortal: search 'UCEC CPTAC' → download clinical data")
        print("    3. The 'Molecular Subtype' column typically has: MSI-H, MSS, POLE, CN-high, CN-low")

except Exception as ex:
    print(f"\nMAJOR 7 FAILED: {ex}")


# ║  SUMMARY                                                                  ║
print("\n" + "█"*80)
print("SUMMARY — MAJOR 4 + MAJOR 7")
print("█"*80)

print(f"\n{'='*80}")
print(f"  {'Analysis':<40s} {'AUC':>7s} {'95% CI':>18s} {'n':>6s} {'Events':>8s}")
print(f"{'='*80}")
for k, v in RESULTS.items():
    if v.get("status") in ["NOT_AVAILABLE", "INSUFFICIENT_EVENTS"]:
        print(f"  {k:<40s} {'N/A':>7s} {'':>18s} {str(v.get('n','')):>6s} {v.get('status','')}")
    else:
        auc_s = f"{v['auc']:.3f}" if v.get('auc') is not None else "N/A"
        ci_s = v.get('ci', '')
        n_s = str(v.get('n', ''))
        pos_s = str(v.get('pos', ''))
        print(f"  {k:<40s} {auc_s:>7s} {ci_s:>18s} {n_s:>6s} {pos_s:>8s}")
print(f"{'='*80}")

# Save
out_json = OUT / f"reviewer_major4_7_{time.strftime('%Y%m%d_%H%M%S')}.json"
def js(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj): return None
    return obj
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({k: {kk: js(vv) for kk,vv in v.items()} for k,v in RESULTS.items()}, f, indent=2)
print(f"\nSaved → {out_json}")

# SECTION 22: CONSENSUS HRD LABELS & BRCA-TRAINED MODEL

# REVIEWER RESPONSE — Major 2 + Major 8
# Major 2: Multi-algorithm HRD definition (consensus: scarHRD + BRCA1/2 mutation)
# Major 8: Train model on BRCA1/2 mutations instead of scarHRD
#
# Uses: mc3 MAF (on disk), TCGA OpenCLIP embeddings (on disk), labels (on disk)
# Runtime: ~2 minutes, no GPU

warnings.filterwarnings("ignore")

DL_ROOT   = Path(r"D:\个人文件夹\Sanwal\DL_V2")
LABELS_PQ = DL_ROOT / "artifacts" / "labels" / "labels.parquet"
TCGA_EMB  = DL_ROOT / "artifacts" / "embeddings" / "patient_means_clean_run_20250908_020405_emb_openclip_vitb16_turbo.parquet"
MC3_MAF   = DL_ROOT / "artifacts" / "mc3.v0.2.8.PUBLIC.maf"
OUT       = Path(r"D:\个人文件夹\Sanwal\IHGAMP_Revision")
OUT.mkdir(parents=True, exist_ok=True)

PCA_N = 384; RIDGE_ALPHA = 30.0; SEED = 42; BOOT_N = 2000

RESULTS = {}

print("█"*80)
print("LOADING DATA")
print("█"*80)

# Labels
L = pd.read_parquet(LABELS_PQ)
L["patient"] = L["patient"].astype(str).str.upper().str.slice(0, 12)
L = L.drop_duplicates("patient").set_index("patient")
print(f"Labels: {len(L)} patients, HRD non-null: {L['HRD'].notna().sum()}")

# Embeddings
X = pd.read_parquet(TCGA_EMB)
if "patient" in X.columns: X = X.set_index("patient")
X.index = X.index.astype(str).str.upper().str.slice(0, 12)
feat_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
X = X[feat_cols]
print(f"Embeddings: {len(X)} patients × {len(feat_cols)}d")

# mc3 MAF → BRCA1/2 mutations
assert MC3_MAF.exists(), f"mc3 MAF not found: {MC3_MAF}"
print(f"Loading mc3 MAF: {MC3_MAF.name}...")
maf = pd.read_csv(MC3_MAF, sep="\t",
                   usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"],
                   dtype=str, comment="#", low_memory=False)
maf["patient"] = maf["Tumor_Sample_Barcode"].str.slice(0, 12)

# BRCA1/2 mutations (nonsilent only)
silent = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR", "RNA"}
maf_ns = maf[~maf["Variant_Classification"].isin(silent)].copy()

brca1_pts = set(maf_ns[maf_ns["Hugo_Symbol"] == "BRCA1"]["patient"])
brca2_pts = set(maf_ns[maf_ns["Hugo_Symbol"] == "BRCA2"]["patient"])
brca12_pts = brca1_pts | brca2_pts
all_maf_pts = set(maf["patient"].unique())

print(f"mc3: {len(all_maf_pts)} patients total")
print(f"  BRCA1 mutated: {len(brca1_pts)}")
print(f"  BRCA2 mutated: {len(brca2_pts)}")
print(f"  BRCA1/2 union: {len(brca12_pts)}")

# Align all three
common = sorted(set(X.index) & set(L.index) & all_maf_pts)
X_al = X.loc[common]
L_al = L.loc[common].copy()
L_al["BRCA12_mut"] = L_al.index.isin(brca12_pts).astype(int)
L_al["BRCA1_mut"] = L_al.index.isin(brca1_pts).astype(int)
L_al["BRCA2_mut"] = L_al.index.isin(brca2_pts).astype(int)

has_hrd = L_al["HRD"].notna()
print(f"\nAligned: {len(common)} patients with embeddings + labels + mutation data")
print(f"  HRD non-null: {has_hrd.sum()}")
print(f"  BRCA1/2 mutated: {L_al['BRCA12_mut'].sum()}")

# Splits
idx_tr = L_al.index[(L_al["split"] == "train") & has_hrd]
idx_va = L_al.index[(L_al["split"] == "val") & has_hrd]
idx_te = L_al.index[(L_al["split"] == "test") & has_hrd]
print(f"  Splits: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

# ║  MAJOR 2 — MULTI-ALGORITHM CONSENSUS HRD LABELS                         ║
# ║  Reviewer: "Study would be strengthened by using multiple HRD            ║
# ║  prediction algorithms to define high-confidence HRD+ tumors"            ║
print("\n" + "█"*80)
print("MAJOR 2 — MULTI-ALGORITHM CONSENSUS HRD LABELS")
print("█"*80)

# Define multiple HRD label strategies:
# A) Original: scarHRD ≥ 33 (current paper)
# B) Conservative: scarHRD ≥ 42 (clinical standard, what reviewer prefers)
# C) BRCA-confirmed: scarHRD ≥ 33 AND BRCA1/2 mutated (highest confidence)
# D) Broad consensus: scarHRD ≥ 33 OR BRCA1/2 mutated
# E) Strict consensus: scarHRD ≥ 42 AND BRCA1/2 mutated

label_strategies = {}

# Strategy A: Original (scarHRD ≥ 33)
label_strategies["scarHRD≥33 (original)"] = {
    "y": (L_al.loc[has_hrd, "HRD"] >= 33).astype(int),
    "threshold": 33,
}

# Strategy B: Conservative (scarHRD ≥ 42)
label_strategies["scarHRD≥42 (clinical)"] = {
    "y": (L_al.loc[has_hrd, "HRD"] >= 42).astype(int),
    "threshold": 42,
}

# Strategy C: High-confidence (scarHRD ≥ 33 AND BRCA1/2 mutated)
y_c = ((L_al.loc[has_hrd, "HRD"] >= 33) & (L_al.loc[has_hrd, "BRCA12_mut"] == 1)).astype(int)
label_strategies["scarHRD≥33 + BRCA1/2 (high-conf)"] = {"y": y_c, "threshold": "33+BRCA"}

# Strategy D: Broad (scarHRD ≥ 33 OR BRCA1/2 mutated)
y_d = ((L_al.loc[has_hrd, "HRD"] >= 33) | (L_al.loc[has_hrd, "BRCA12_mut"] == 1)).astype(int)
label_strategies["scarHRD≥33 | BRCA1/2 (broad)"] = {"y": y_d, "threshold": "33|BRCA"}

# Strategy E: Strict (scarHRD ≥ 42 AND BRCA1/2 mutated)
y_e = ((L_al.loc[has_hrd, "HRD"] >= 42) & (L_al.loc[has_hrd, "BRCA12_mut"] == 1)).astype(int)
label_strategies["scarHRD≥42 + BRCA1/2 (strict)"] = {"y": y_e, "threshold": "42+BRCA"}

print(f"\n{'='*90}")
print(f"{'Strategy':<40s} {'HRD+':<8s} {'HRD-':<8s} {'%HRD+':<8s} {'Train AUC':<12s} {'Test AUC':<12s} {'Test 95% CI'}")
print(f"{'='*90}")

for name, info in label_strategies.items():
    y_full = info["y"]
    
    # Get train/test labels
    y_tr = y_full.loc[y_full.index.isin(idx_tr)]
    y_te = y_full.loc[y_full.index.isin(idx_te)]
    y_va = y_full.loc[y_full.index.isin(idx_va)]
    
    pos_total = int(y_full.sum())
    neg_total = len(y_full) - pos_total
    pct = 100 * pos_total / len(y_full) if len(y_full) else 0
    
    pos_tr = int(y_tr.sum())
    pos_te = int(y_te.sum())
    
    if pos_tr < 5 or (len(y_tr) - pos_tr) < 5:
        print(f"  {name:<40s} {pos_total:<8d} {neg_total:<8d} {pct:<8.1f} {'skip (too few)':>12s}")
        continue
    
    # Train PCA→Ridge→Platt on TRAIN with these labels
    # For regression target: use continuous HRD for strategies A/B, binary for C/D/E
    X_tr = get_X(y_tr.index)
    X_te = get_X(y_te.index)
    X_va = get_X(y_va.index)
    
    pca_n = min(PCA_N, X_tr.shape[1]-1, X_tr.shape[0]-1)
    
    # Use continuous HRD as regression target for strategies A and B
    if info["threshold"] in [33, 42]:
        reg_target = L_al.loc[y_tr.index, "HRD"].astype(float).values
    else:
        # For consensus strategies, use binary directly
        reg_target = y_tr.values.astype(float)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_n, random_state=SEED)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
    ]).fit(X_tr, reg_target)
    
    z_tr = pipe.predict(X_tr).reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
        z_tr, y_tr.values)
    
    # Predict
    p_te = platt.predict_proba(pipe.predict(X_te).reshape(-1, 1))[:, 1]
    p_tr = platt.predict_proba(z_tr)[:, 1]
    
    if pos_te >= 3 and (len(y_te) - pos_te) >= 3:
        auc_te, lo, hi = boot_ci(y_te.values, p_te, roc_auc_score)
        auc_tr = roc_auc_score(y_tr.values, p_tr)
        ci_str = f"{lo:.3f}–{hi:.3f}"
    else:
        auc_te, auc_tr, ci_str = np.nan, np.nan, "N/A (few events)"
    
    print(f"  {name:<40s} {pos_total:<8d} {neg_total:<8d} {pct:<8.1f} {auc_tr:<12.3f} {auc_te:<12.3f} {ci_str}")
    
    RESULTS[f"M2_{name[:30]}"] = {
        "auc_test": auc_te, "auc_train": auc_tr, "ci": ci_str,
        "n": len(y_full), "pos": pos_total, "pct": round(pct, 1),
        "pos_test": pos_te,
    }

print(f"{'='*90}")

# Cross-tabulation: scarHRD vs BRCA1/2
print(f"\n--- Cross-tabulation: scarHRD status × BRCA1/2 mutation ---")
sub = L_al[has_hrd].copy()
sub["scarHRD≥33"] = (sub["HRD"] >= 33).astype(int)
sub["scarHRD≥42"] = (sub["HRD"] >= 42).astype(int)
ct33 = pd.crosstab(sub["scarHRD≥33"], sub["BRCA12_mut"], margins=True)
ct33.index = ["scarHRD<33", "scarHRD≥33", "Total"]
ct33.columns = ["BRCA-WT", "BRCA-Mut", "Total"]
print(ct33)

print()
ct42 = pd.crosstab(sub["scarHRD≥42"], sub["BRCA12_mut"], margins=True)
ct42.index = ["scarHRD<42", "scarHRD≥42", "Total"]
ct42.columns = ["BRCA-WT", "BRCA-Mut", "Total"]
print(ct42)

# Save cross-tabs
ct33.to_csv(OUT / "crosstab_scarHRD33_vs_BRCA12.csv")
ct42.to_csv(OUT / "crosstab_scarHRD42_vs_BRCA12.csv")


# ║  MAJOR 8 — TRAIN ON BRCA1/2 MUTATIONS INSTEAD OF scarHRD                ║
# ║  Reviewer: "Train a model on BRCA1/2 mutation presence rather than       ║
# ║  scarHRD score. See if it has better predictive power for actual HRD."   ║
print("\n" + "█"*80)
print("MAJOR 8 — TRAIN ON BRCA1/2 MUTATIONS")
print("█"*80)

# 8A: Train model to predict BRCA1/2 mutation → then evaluate on scarHRD
# 8B: Train model to predict BRCA1/2 → evaluate on BRCA1/2 (self-prediction)
# 8C: Compare scarHRD-trained model vs BRCA-trained model for HRD prediction

# Use ALL patients with mutation data (not just those with scarHRD)
common_all = sorted(set(X.index) & all_maf_pts)
X_all = X.loc[common_all]
L_all = L.loc[L.index.isin(common_all)].copy()
L_all["BRCA12_mut"] = L_all.index.isin(brca12_pts).astype(int)

# Need splits for these patients
idx_tr_all = L_all.index[L_all["split"] == "train"]
idx_te_all = L_all.index[L_all["split"] == "test"]
idx_va_all = L_all.index[L_all["split"] == "val"]

brca_tr = L_all.loc[idx_tr_all, "BRCA12_mut"]
brca_te = L_all.loc[idx_te_all, "BRCA12_mut"]
brca_va = L_all.loc[idx_va_all, "BRCA12_mut"]

print(f"\nAll patients with mutations + embeddings: {len(common_all)}")
print(f"  Train: {len(idx_tr_all)} (BRCA+={brca_tr.sum()})")
print(f"  Val:   {len(idx_va_all)} (BRCA+={brca_va.sum()})")
print(f"  Test:  {len(idx_te_all)} (BRCA+={brca_te.sum()})")

def get_X_all(idx): return X_all.loc[idx, feat_cols].values.astype(np.float32)

# --- 8A: Train on BRCA1/2 mutations ---
print(f"\n--- 8A: Train PCA→Ridge→Platt on BRCA1/2 mutations ---")

X_tr_b = get_X_all(idx_tr_all)
y_tr_b = brca_tr.values

pca_n_b = min(PCA_N, X_tr_b.shape[1]-1, X_tr_b.shape[0]-1)
pipe_brca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=pca_n_b, random_state=SEED)),
    ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
]).fit(X_tr_b, y_tr_b.astype(float))

z_tr_b = pipe_brca.predict(X_tr_b).reshape(-1, 1)
platt_brca = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
    z_tr_b, y_tr_b)

def brca_predict(idx):
    return platt_brca.predict_proba(pipe_brca.predict(get_X_all(idx)).reshape(-1,1))[:,1]

# Evaluate BRCA-trained model on:
# (a) BRCA1/2 prediction (self-task)
p_brca_te = brca_predict(idx_te_all)
y_brca_te = brca_te.values
auc_brca_self, lo_bs, hi_bs = boot_ci(y_brca_te, p_brca_te, roc_auc_score)
print(f"  BRCA-trained → BRCA1/2 prediction (TEST): AUC = {auc_brca_self:.3f} ({lo_bs:.3f}–{hi_bs:.3f})")
print(f"    (n={len(y_brca_te)}, BRCA+={y_brca_te.sum()})")

RESULTS["M8_BRCA_trained_self"] = {
    "auc": auc_brca_self, "ci": f"{lo_bs:.3f}–{hi_bs:.3f}",
    "n": len(y_brca_te), "pos": int(y_brca_te.sum()),
    "task": "BRCA1/2 mutation prediction",
}

# (b) scarHRD prediction (cross-task: does BRCA model predict HRD?)
idx_te_hrd = L_all.index[(L_all["split"]=="test") & L_all["HRD"].notna()]
p_brca_hrd = brca_predict(idx_te_hrd)
y_hrd_33 = (L_al.loc[idx_te_hrd, "HRD"] >= 33).astype(int).values
y_hrd_42 = (L_al.loc[idx_te_hrd, "HRD"] >= 42).astype(int).values

if y_hrd_33.sum() >= 5 and (len(y_hrd_33) - y_hrd_33.sum()) >= 5:
    auc_brca_hrd33, lo_33, hi_33 = boot_ci(y_hrd_33, p_brca_hrd, roc_auc_score)
    print(f"  BRCA-trained → scarHRD≥33 prediction (TEST): AUC = {auc_brca_hrd33:.3f} ({lo_33:.3f}–{hi_33:.3f})")
    RESULTS["M8_BRCA_trained_vs_scarHRD33"] = {
        "auc": auc_brca_hrd33, "ci": f"{lo_33:.3f}–{hi_33:.3f}",
        "n": len(y_hrd_33), "pos": int(y_hrd_33.sum()),
        "task": "BRCA-trained model → scarHRD≥33",
    }

if y_hrd_42.sum() >= 5 and (len(y_hrd_42) - y_hrd_42.sum()) >= 5:
    auc_brca_hrd42, lo_42, hi_42 = boot_ci(y_hrd_42, p_brca_hrd, roc_auc_score)
    print(f"  BRCA-trained → scarHRD≥42 prediction (TEST): AUC = {auc_brca_hrd42:.3f} ({lo_42:.3f}–{hi_42:.3f})")
    RESULTS["M8_BRCA_trained_vs_scarHRD42"] = {
        "auc": auc_brca_hrd42, "ci": f"{lo_42:.3f}–{hi_42:.3f}",
        "n": len(y_hrd_42), "pos": int(y_hrd_42.sum()),
        "task": "BRCA-trained model → scarHRD≥42",
    }

# --- 8B: Compare head-to-head: scarHRD-trained vs BRCA-trained ---
print(f"\n--- 8B: Head-to-head comparison on TEST set ---")

# Rebuild scarHRD-trained model (original pipeline)
y_tr_scar = (L_al.loc[idx_tr, "HRD"] >= 33).astype(int)
pipe_scar = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=min(PCA_N, get_X(idx_tr).shape[1]-1, get_X(idx_tr).shape[0]-1), random_state=SEED)),
    ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
]).fit(get_X(idx_tr), L_al.loc[idx_tr, "HRD"].astype(float).values)
z_tr_s = pipe_scar.predict(get_X(idx_tr)).reshape(-1,1)
platt_scar = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED).fit(
    z_tr_s, y_tr_scar.values)

def scar_predict(idx):
    return platt_scar.predict_proba(pipe_scar.predict(get_X(idx)).reshape(-1,1))[:,1]

# On TEST set with HRD labels
p_scar_te = scar_predict(idx_te)
y_te_33 = (L_al.loc[idx_te, "HRD"] >= 33).astype(int).values
y_te_42 = (L_al.loc[idx_te, "HRD"] >= 42).astype(int).values
y_te_brca = L_al.loc[idx_te, "BRCA12_mut"].values

# scarHRD-trained model predictions
auc_scar_33 = roc_auc_score(y_te_33, p_scar_te)
auc_scar_brca = roc_auc_score(y_te_brca, p_scar_te) if y_te_brca.sum() >= 5 else np.nan

# BRCA-trained model predictions (need to align to idx_te which requires HRD)
p_brca_te2 = brca_predict(idx_te)
auc_brca_33 = roc_auc_score(y_te_33, p_brca_te2)
auc_brca_brca2 = roc_auc_score(y_te_brca, p_brca_te2) if y_te_brca.sum() >= 5 else np.nan

if y_te_42.sum() >= 5:
    auc_scar_42 = roc_auc_score(y_te_42, p_scar_te)
    auc_brca_42 = roc_auc_score(y_te_42, p_brca_te2)
else:
    auc_scar_42, auc_brca_42 = np.nan, np.nan

print(f"\n{'='*75}")
print(f"{'Target':<25s} {'scarHRD-trained':>18s} {'BRCA-trained':>18s} {'Δ':>10s}")
print(f"{'='*75}")
print(f"{'scarHRD ≥ 33':<25s} {auc_scar_33:>18.3f} {auc_brca_33:>18.3f} {auc_brca_33-auc_scar_33:>+10.3f}")
if np.isfinite(auc_scar_42):
    print(f"{'scarHRD ≥ 42':<25s} {auc_scar_42:>18.3f} {auc_brca_42:>18.3f} {auc_brca_42-auc_scar_42:>+10.3f}")
if np.isfinite(auc_scar_brca):
    print(f"{'BRCA1/2 mutation':<25s} {auc_scar_brca:>18.3f} {auc_brca_brca2:>18.3f} {auc_brca_brca2-auc_scar_brca:>+10.3f}")
print(f"{'='*75}")

RESULTS["M8_head_to_head"] = {
    "scarHRD_trained_vs_scarHRD33": auc_scar_33,
    "scarHRD_trained_vs_scarHRD42": auc_scar_42 if np.isfinite(auc_scar_42) else None,
    "scarHRD_trained_vs_BRCA12": auc_scar_brca if np.isfinite(auc_scar_brca) else None,
    "BRCA_trained_vs_scarHRD33": auc_brca_33,
    "BRCA_trained_vs_scarHRD42": auc_brca_42 if np.isfinite(auc_brca_42) else None,
    "BRCA_trained_vs_BRCA12": auc_brca_brca2 if np.isfinite(auc_brca_brca2) else None,
}

# --- 8C: BRCA-trained model on non-BRCA HRD patients ---
# The reviewer specifically asked: "if a model trained on BRCA1/2 mutation presence
# has better predictive power for actual HRD tumors without BRCA1/2 mutations"
print(f"\n--- 8C: Predicting HRD in non-BRCA-mutant patients ---")
print(f"  (Does HRD morphology exist beyond BRCA1/2 mutations?)")

non_brca_te = L_al.loc[idx_te][(L_al.loc[idx_te, "BRCA12_mut"] == 0) & L_al.loc[idx_te, "HRD"].notna()]
y_nb_33 = (non_brca_te["HRD"] >= 33).astype(int)
pos_nb = int(y_nb_33.sum())
neg_nb = len(y_nb_33) - pos_nb
print(f"  Non-BRCA TEST patients: n={len(non_brca_te)}, scarHRD≥33: {pos_nb}")

if pos_nb >= 5 and neg_nb >= 5:
    p_scar_nb = scar_predict(non_brca_te.index)
    p_brca_nb = brca_predict(non_brca_te.index)
    
    auc_scar_nb = roc_auc_score(y_nb_33.values, p_scar_nb)
    auc_brca_nb = roc_auc_score(y_nb_33.values, p_brca_nb)
    
    auc_scar_nb_pt, lo_s, hi_s = boot_ci(y_nb_33.values, p_scar_nb, roc_auc_score)
    auc_brca_nb_pt, lo_b, hi_b = boot_ci(y_nb_33.values, p_brca_nb, roc_auc_score)
    
    print(f"  scarHRD-trained → HRD in non-BRCA: AUC = {auc_scar_nb:.3f} ({lo_s:.3f}–{hi_s:.3f})")
    print(f"  BRCA-trained   → HRD in non-BRCA: AUC = {auc_brca_nb:.3f} ({lo_b:.3f}–{hi_b:.3f})")
    print(f"  Δ = {auc_brca_nb - auc_scar_nb:+.3f}")
    
    if auc_scar_nb > auc_brca_nb + 0.02:
        print(f"\n  → scarHRD-trained model is BETTER at detecting non-BRCA HRD")
        print(f"    This supports that scarHRD captures additional HRD mechanisms beyond BRCA1/2")
    elif auc_brca_nb > auc_scar_nb + 0.02:
        print(f"\n  → BRCA-trained model generalizes to non-BRCA HRD")
        print(f"    BRCA morphology may overlap with other HRD mechanisms")
    else:
        print(f"\n  → Both models perform similarly on non-BRCA HRD patients")
    
    RESULTS["M8_non_BRCA_HRD"] = {
        "scarHRD_trained_auc": auc_scar_nb, "scarHRD_ci": f"{lo_s:.3f}–{hi_s:.3f}",
        "BRCA_trained_auc": auc_brca_nb, "BRCA_ci": f"{lo_b:.3f}–{hi_b:.3f}",
        "n": len(non_brca_te), "pos": pos_nb,
        "interpretation": "Comparison on BRCA-wildtype patients only",
    }


# ║  SUMMARY                                                                  ║
print("\n" + "█"*80)
print("COMPLETE SUMMARY — MAJOR 2 + MAJOR 8")
print("█"*80)

print(f"\n{'='*90}")
print(f"  {'Analysis':<45s} {'AUC':>7s} {'95% CI':>18s} {'n':>6s} {'Events':>8s}")
print(f"{'='*90}")
for k, v in RESULTS.items():
    if "auc" in v:
        auc_s = f"{v['auc']:.3f}" if v.get('auc') is not None and np.isfinite(v.get('auc', np.nan)) else "N/A"
        ci_s = v.get('ci', '')
        n_s = str(v.get('n', ''))
        pos_s = str(v.get('pos', ''))
        print(f"  {k:<45s} {auc_s:>7s} {ci_s:>18s} {n_s:>6s} {pos_s:>8s}")
    elif "auc_test" in v:
        auc_s = f"{v['auc_test']:.3f}" if np.isfinite(v.get('auc_test', np.nan)) else "N/A"
        ci_s = v.get('ci', '')
        n_s = str(v.get('n', ''))
        pos_s = str(v.get('pos', ''))
        pct_s = f"({v.get('pct','')}%)"
        print(f"  {k:<45s} {auc_s:>7s} {ci_s:>18s} {n_s:>6s} {pos_s:>8s} {pct_s}")
print(f"{'='*90}")

# Save
out_json = OUT / f"reviewer_major2_8_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({k: {kk: js(vv) for kk, vv in v.items()} for k, v in RESULTS.items()}, f, indent=2)
print(f"\nSaved → {out_json}")
print(f"Cross-tabs → {OUT / 'crosstab_scarHRD33_vs_BRCA12.csv'}")
print(f"           → {OUT / 'crosstab_scarHRD42_vs_BRCA12.csv'}")
