"""
Preprocessing module for WSI tiling and normalization.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    warnings.warn("OpenSlide not available. WSI processing will be limited.")


@dataclass
class TileInfo:
    """Information about an extracted tile."""
    x: int
    y: int
    level: int
    size: int
    tissue_ratio: float


class TileExtractor:
    """Extract tiles from whole slide images."""

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 256,
        level: int = 0,
        tissue_threshold: float = 0.60,
        patch_cap: int = 3000,
        target_mpp: Optional[float] = 0.5,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.level = level
        self.tissue_threshold = tissue_threshold
        self.patch_cap = patch_cap
        self.target_mpp = target_mpp

        if not OPENSLIDE_AVAILABLE:
            raise RuntimeError("OpenSlide is required for tile extraction")

    def get_slide_mpp(self, slide: "openslide.OpenSlide") -> float:
        """Get microns per pixel from slide metadata."""
        try:
            mpp_x = float(slide.properties.get(
                openslide.PROPERTY_NAME_MPP_X, 0.25
            ))
            return mpp_x
        except (ValueError, KeyError):
            return 0.25  # Default assumption

    def compute_tissue_mask(
        self, 
        slide: "openslide.OpenSlide",
        downsample: int = 32
    ) -> Tuple[np.ndarray, float]:
        """Compute tissue mask from slide thumbnail."""
        # Get thumbnail
        dims = slide.dimensions
        thumb_size = (dims[0] // downsample, dims[1] // downsample)
        thumbnail = slide.get_thumbnail(thumb_size)
        thumb_array = np.array(thumbnail)

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(thumb_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask > 0, downsample

    def extract_tiles(
        self,
        slide_path: str,
        output_dir: Optional[str] = None,
        return_arrays: bool = True,
    ) -> List[Tuple[np.ndarray, TileInfo]]:
        """Extract tiles from a single slide."""
        slide = openslide.OpenSlide(slide_path)

        # Get tissue mask
        tissue_mask, ds = self.compute_tissue_mask(slide)

        # Calculate tile positions
        dims = slide.level_dimensions[self.level]
        tiles = []

        for y in range(0, dims[1] - self.patch_size + 1, self.stride):
            for x in range(0, dims[0] - self.patch_size + 1, self.stride):
                # Check tissue content
                mask_x = x // ds
                mask_y = y // ds
                mask_size = self.patch_size // ds

                if mask_x + mask_size > tissue_mask.shape[1]:
                    continue
                if mask_y + mask_size > tissue_mask.shape[0]:
                    continue

                region = tissue_mask[
                    mask_y:mask_y + mask_size,
                    mask_x:mask_x + mask_size
                ]
                tissue_ratio = region.mean()

                if tissue_ratio >= self.tissue_threshold:
                    tiles.append(TileInfo(
                        x=x, y=y, 
                        level=self.level,
                        size=self.patch_size,
                        tissue_ratio=float(tissue_ratio)
                    ))

        # Apply patch cap
        if len(tiles) > self.patch_cap:
            # Prioritize high tissue content
            tiles = sorted(tiles, key=lambda t: -t.tissue_ratio)
            tiles = tiles[:self.patch_cap]

        # Extract tile images
        results = []
        for tile_info in tiles:
            region = slide.read_region(
                (tile_info.x, tile_info.y),
                tile_info.level,
                (tile_info.size, tile_info.size)
            ).convert('RGB')

            if return_arrays:
                results.append((np.array(region), tile_info))

            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                fname = f"tile_x{tile_info.x}_y{tile_info.y}.jpg"
                region.save(out_path / fname, quality=90)

        slide.close()
        return results

    def process_slide(
        self,
        slide_path: str,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Process a single slide and return summary."""
        tiles = self.extract_tiles(slide_path, output_dir, return_arrays=False)

        return {
            "slide": str(slide_path),
            "n_tiles": len(tiles) if output_dir else 0,
            "status": "OK"
        }
