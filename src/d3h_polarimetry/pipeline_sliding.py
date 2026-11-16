"""Sliding-window analysis implementation for polarization workflows."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .pipeline_config import (
    AnalysisConfig,
    LocalStats,
    SlidingWindowMaps,
    SlidingWindowStats,
)
from .polarization import PolarizationAnalyzer
from .pipeline_helpers import ProgressTracker

logger = logging.getLogger(__name__)


class SlidingWindowAnalyzer:
    """
    Performs sliding window orientation analysis on ROI stacks.

    Analyzes each window position using the PolarizationAnalyzer and
    collects results into maps and statistics.

    Attributes:
        analyzer: The polarization analyzer instance.
        config: Analysis configuration.
    """

    def __init__(
        self,
        analyzer: PolarizationAnalyzer,
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        """
        Initialize the sliding window analyzer.

        Args:
            analyzer: PolarizationAnalyzer instance.
            config: Analysis configuration. Uses defaults if not provided.
        """
        self.analyzer = analyzer
        self.config = config or AnalysisConfig()
        logger.debug("SlidingWindowAnalyzer initialized")

    @staticmethod
    def _mask_rejected_fft_fields(row: Dict[str, Any]) -> None:
        """Blank exported FFT analysis fields for windows rejected by the A0 gate."""
        for key in (
            "phase_fft",
            "phase_deg_fft",
            "D_fft_fft",
            "D_fft_positive_fft",
            "a_0_fft",
            "a_2_fft",
            "a_4_fft",
            "a_6_fft",
            "a_2_iso_fft",
            "a_4_iso_fft",
            "a_6_iso_fft",
            "F0_fft",
            "F2_fft",
            "F4_fft",
            "F6_fft",
            "F0_signed_fft",
            "F2_signed_fft",
            "F4_signed_fft",
            "F6_signed_fft",
            "T_fft",
            "Q_fft",
            "B_formula_fft",
            "C_fft",
            "T_signed_fft",
            "Q_signed_fft",
            "B_signed_formula_fft",
            "C_signed_fft",
            "C_signed_stable_fft",
            "C_signed_stable_unbounded_fft",
            "C_signed_stable_source_fft",
            "C_signed_condition_fft",
            "phi_2_fft",
            "phi_2_deg_fft",
            "phi_4_fft",
            "phi_4_deg_fft",
            "phi_6_fft",
            "phi_6_deg_fft",
            "model_phi_2_fft",
            "model_phi_2_deg_fft",
            "model_phi_4_fft",
            "model_phi_4_deg_fft",
            "dominant_phi_fft",
            "dominant_phi_deg_fft",
            "dominant_phi_on_2_fft",
            "dominant_phi_on_2_deg_fft",
            "dominant_phi_on_2_by_divide_fft",
            "dominant_phi_on_2_by_divide_deg_fft",
            "R_fft",
            "R_signed_fft",
            "B_fft",
            "theta_fft",
            "theta_deg_fft",
            "best_formula_fft",
            "best_formula_index_fft",
            "shared_phi_fft",
            "shared_phi_deg_fft",
            "shared_best_formula_fft",
            "shared_best_formula_index_fft",
        ):
            if key in row:
                row[key] = (
                    ""
                    if key in {"best_formula_fft", "shared_best_formula_fft"}
                    else np.nan
                )

    @staticmethod
    def _mask_rejected_curve_fields(row: Dict[str, Any]) -> None:
        """Blank exported curve-fit analysis fields for windows rejected by the A0 gate."""
        for key in (
            "angle_output_deg_curve",
            "phi_curve",
            "phi_deg_curve",
            "D_curve",
            "A_curve",
            "B_curve",
        ):
            if key in row:
                row[key] = np.nan

    def analyze(
        self,
        roi_stack: np.ndarray,
        angles: np.ndarray,
        apply_threshold: bool = True,
        progress_label: Optional[str] = None,
        roi_origin: Tuple[int, int] = (0, 0),
    ) -> Tuple[SlidingWindowMaps, SlidingWindowStats, List[Dict[str, Any]]]:
        """
        Perform sliding window analysis on an ROI stack.

        Args:
            roi_stack: 3D array of shape (n_frames, height, width).
            angles: 1D array of polarization angles in radians.
            apply_threshold: Whether to filter by R² threshold for stats.

        Returns:
            Tuple of (SlidingWindowMaps, SlidingWindowStats, raw_window_data).

        Raises:
            ValueError: If ROI is too small for sliding window.
        """
        n_frames, roi_height, roi_width = roi_stack.shape
        ws = self.config.window_size
        map_h = roi_height - ws + 1
        map_w = roi_width - ws + 1
        if map_h < 1 or map_w < 1:
            raise ValueError(
                f"ROI too small for window size {ws}. ROI: {roi_height}x{roi_width}, need at least {ws}x{ws}"
            )
        mode = getattr(self.analyzer, "fft_mode", None)
        is_thgmm2 = False
        is_myelin = False
        is_myosin = False
        map_fft_ang = np.full((map_h, map_w), np.nan)
        map_fft_r2 = np.full((map_h, map_w), 0.0)
        map_curve_ang = np.full((map_h, map_w), np.nan)
        map_curve_r2 = np.full((map_h, map_w), 0.0)
        roi_mean_for_a0 = np.mean(roi_stack, axis=0)
        center_offset = ws // 2
        map_a0_intensity = roi_mean_for_a0[
            center_offset : center_offset + map_h,
            center_offset : center_offset + map_w,
        ]
        a0_cutoff = self.config.a0_mask_cutoff
        global_a0_map = self.config.a0_mask_map
        global_a0_pass_map = self.config.a0_mask_pass_map
        map_a0_intensity_mask: Optional[np.ndarray] = None
        if a0_cutoff is not None and global_a0_map is not None:
            origin_r, origin_c = roi_origin
            center_r0 = origin_r + center_offset
            center_c0 = origin_c + center_offset
            candidate = global_a0_map[
                center_r0 : center_r0 + map_h, center_c0 : center_c0 + map_w
            ]
            if candidate.shape == (map_h, map_w):
                map_a0_intensity = np.asarray(candidate, dtype=float)
            if global_a0_pass_map is not None:
                candidate_mask = global_a0_pass_map[
                    center_r0 : center_r0 + map_h,
                    center_c0 : center_c0 + map_w,
                ]
                if candidate_mask.shape == (map_h, map_w):
                    map_a0_intensity_mask = np.asarray(
                        candidate_mask, dtype=bool
                    )
        elif self.config.a0_mask_percentile is not None and a0_cutoff is None:
            finite_a0 = map_a0_intensity[np.isfinite(map_a0_intensity)]
            if finite_a0.size:
                a0_cutoff = float(
                    np.nanpercentile(
                        finite_a0, float(self.config.a0_mask_percentile)
                    )
                )
        if map_a0_intensity_mask is None and a0_cutoff is not None:
            cutoff_value = float(a0_cutoff)
            cutoff_eps = max(abs(cutoff_value), 1.0) * 1e-12
            map_a0_intensity_mask = np.isfinite(map_a0_intensity) & (
                map_a0_intensity >= cutoff_value - cutoff_eps
            )
        elif map_a0_intensity_mask is None:
            map_a0_intensity_mask = np.ones((map_h, map_w), dtype=bool)
        fft_vals: List[float] = []
        curve_vals: List[float] = []
        count_total = 0
        count_fft = 0
        count_curve = 0
        raw_window_data = []
        threshold = self.config.r2_threshold if apply_threshold else 0.0
        progress = ProgressTracker(
            total=map_h * map_w, label=progress_label or "Sliding window"
        )
        for r in range(map_h):
            for c in range(map_w):
                sub_stack = roi_stack[:, r : r + ws, c : c + ws]
                sub_signal = sub_stack.mean(axis=(1, 2))
                res = self.analyzer.analyze_signal(
                    sub_signal, angles, patch_data=self.config.patch_data
                )
                count_total += 1
                a0_intensity = float(map_a0_intensity[r, c])
                a0_intensity_passed = bool(map_a0_intensity_mask[r, c])
                row = {
                    "Window R": r,
                    "Window C": c,
                    "a0_intensity": a0_intensity,
                    "a0_intensity_cutoff": a0_cutoff
                    if a0_cutoff is not None
                    else np.nan,
                    "a0_intensity_passed": a0_intensity_passed,
                }
                if "fft" in res and isinstance(res["fft"], dict):
                    r2 = res["fft"].get("r_squared", 0)
                    threshold_passed = r2 > threshold
                    threshold_passed = bool(
                        threshold_passed and a0_intensity_passed
                    )
                    ang = (
                        res["fft"].get("phase_deg", np.nan)
                        if a0_intensity_passed
                        else np.nan
                    )
                    map_fft_r2[r, c] = r2
                    map_fft_ang[r, c] = ang
                    if threshold_passed:
                        fft_vals.append(ang)
                        count_fft += 1
                    for k, v in res["fft"].items():
                        if k != "prediction":
                            row[f"{k}_fft"] = v
                    if not a0_intensity_passed:
                        self._mask_rejected_fft_fields(row)
                if "fit" in res and isinstance(res["fit"], dict):
                    r2 = res["fit"].get("r_squared", 0)
                    threshold_passed = r2 > threshold
                    threshold_passed = bool(
                        threshold_passed and a0_intensity_passed
                    )
                    ang = (
                        res["fit"].get("angle_output_deg", np.nan)
                        if a0_intensity_passed
                        else np.nan
                    )
                    map_curve_r2[r, c] = r2
                    map_curve_ang[r, c] = ang
                    if threshold_passed:
                        curve_vals.append(ang)
                        count_curve += 1
                    for k, v in res["fit"].items():
                        if k != "prediction":
                            row[f"{k}_curve"] = v
                    if not a0_intensity_passed:
                        self._mask_rejected_curve_fields(row)
                raw_window_data.append(row)
                progress.update(count_total)
        maps = SlidingWindowMaps(
            fft_angle=map_fft_ang,
            fft_r2=map_fft_r2,
            curve_angle=map_curve_ang,
            curve_r2=map_curve_r2,
            a0_intensity_map=map_a0_intensity,
            a0_intensity_mask=map_a0_intensity_mask,
        )
        stats = SlidingWindowStats(
            total_windows=count_total,
            fft_passed=count_fft,
            curve_passed=count_curve,
            fft_orientations=fft_vals,
            curve_orientations=curve_vals,
        )
        logger.debug(
            f"Sliding window: {count_total} windows, FFT passed: {count_fft}, Curve passed: {count_curve}"
        )
        progress.finish()
        return (maps, stats, raw_window_data)

    def get_local_stats(
        self,
        poi_r: int,
        poi_c: int,
        angle_map: np.ndarray,
        r2_map: np.ndarray,
        region_size: int = 20,
    ) -> Optional[LocalStats]:
        """
        Get statistics for a local region around a POI.

        Args:
            poi_r: POI row coordinate (in ROI space).
            poi_c: POI column coordinate (in ROI space).
            angle_map: 2D orientation angle map.
            r2_map: 2D R² value map.
            region_size: Size of region to analyze (region_size x region_size).

        Returns:
            LocalStats object or None if no valid data.
        """
        map_h, map_w = angle_map.shape
        half = region_size // 2
        r_start = max(0, poi_r - half)
        r_end = min(map_h, poi_r + half)
        c_start = max(0, poi_c - half)
        c_end = min(map_w, poi_c + half)
        vals: List[float] = []
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                r2 = r2_map[r, c]
                ang = angle_map[r, c]
                if r2 > self.config.r2_threshold and (not np.isnan(ang)):
                    vals.append(ang)
        return LocalStats.from_data(vals)
