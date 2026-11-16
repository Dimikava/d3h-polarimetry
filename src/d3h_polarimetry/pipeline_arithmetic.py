"""POI arithmetic and mask selection for polarization workflows."""

from __future__ import annotations
import logging
from typing import List, Optional, Sequence, Tuple
import numpy as np
from .pipeline_config import AnalysisConfig, LocalStats
from .poi_mask_prototype import (
    PrototypeConfig as MaskPrototypeConfig,
    grow_intensity_masks,
)

logger = logging.getLogger(__name__)


class ArithmeticProcessor:
    """
    Handles POI-based arithmetic operations with error propagation.

    Performs the transformation Z = 2 * (Mean2 - X) for pixels within
    the selection mask, with proper error propagation.

    Attributes:
        config: Analysis configuration.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        """
        Initialize the arithmetic processor.

        Args:
            config: Analysis configuration. Uses defaults if not provided.
        """
        self.config = config or AnalysisConfig()
        logger.debug("ArithmeticProcessor initialized")

    def build_intensity_masks(
        self, roi_mean: np.ndarray, poi_points: Sequence[Tuple[int, int]]
    ) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """Build connected intensity masks for POIs on the ROI mean image."""
        if not poi_points:
            return (np.full(roi_mean.shape, -1, dtype=int), [], [])
        seed_levels = [
            float(
                np.mean(
                    roi_mean[
                        max(0, r - self.config.twist_mask_patch_radius) : min(
                            roi_mean.shape[0],
                            r + self.config.twist_mask_patch_radius + 1,
                        ),
                        max(0, c - self.config.twist_mask_patch_radius) : min(
                            roi_mean.shape[1],
                            c + self.config.twist_mask_patch_radius + 1,
                        ),
                    ]
                )
            )
            for r, c in poi_points
        ]
        positive = [v for v in seed_levels if v > 0]
        if positive:
            low = min(positive)
            high = max(positive)
        else:
            low = 0.0
            high = 0.0
        thresholds: List[float] = []
        for value in seed_levels:
            if high <= low + 1e-12:
                norm = 0.0
            else:
                norm = (value - low) / (high - low)
            thresholds.append(
                self.config.twist_mask_threshold_dim
                + norm
                * (
                    self.config.twist_mask_threshold_bright
                    - self.config.twist_mask_threshold_dim
                )
            )
        mask_config = MaskPrototypeConfig(
            method="intensity",
            reference_mode="mean",
            patch_radius=self.config.twist_mask_patch_radius,
            background_fraction=self.config.twist_mask_background_fraction,
            intensity_thresholds=tuple(thresholds),
        )
        assignment_map, _best, _margin, masks, _templates = (
            grow_intensity_masks(roi_mean, poi_points, mask_config)
        )
        return (assignment_map, masks, thresholds)

    def get_masked_stats(
        self, angle_map: np.ndarray, r2_map: np.ndarray, mask: np.ndarray
    ) -> Optional[LocalStats]:
        """Collect angle statistics inside a mask after R² thresholding."""
        vals: List[float] = []
        rows, cols = angle_map.shape
        for r in range(rows):
            for c in range(cols):
                if (
                    mask[r, c]
                    and r2_map[r, c] > self.config.r2_threshold
                    and (not np.isnan(angle_map[r, c]))
                ):
                    vals.append(float(angle_map[r, c]))
        return LocalStats.from_data(vals)

    def compute_arithmetic(
        self,
        angle_map: np.ndarray,
        r2_map: np.ndarray,
        poi1_stats: LocalStats,
        poi2_stats: LocalStats,
        selection_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float], float]:
        """Return abs(2*(POI2_mean - angle)) % 60 on selected pixels with R² above threshold, and 2*sqrt(std1² + std2²)."""
        mean1, std1 = (poi1_stats.mean, poi1_stats.std)
        mean2, std2 = (poi2_stats.mean, poi2_stats.std)
        sigma_z = 2 * np.sqrt(std1**2 + std2**2)
        logger.debug(
            f"Arithmetic: POI1={mean1:.2f}±{std1:.2f}, POI2={mean2:.2f}±{std2:.2f}"
        )
        logger.debug(f"Using intensity mask selection, sigma_z={sigma_z:.2f}")
        result_map = np.full_like(angle_map, np.nan)
        valid_pixels: List[float] = []
        rows, cols = angle_map.shape
        for r in range(rows):
            for c in range(cols):
                val = angle_map[r, c]
                r2 = r2_map[r, c]
                if r2 > self.config.r2_threshold and (not np.isnan(val)):
                    if selection_mask is None or selection_mask[r, c]:
                        z = np.abs(2 * (mean2 - val)) % 60
                        result_map[r, c] = z
                        valid_pixels.append(z)
        logger.debug(f"Arithmetic: {len(valid_pixels)} valid pixels")
        return (result_map, valid_pixels, sigma_z)
