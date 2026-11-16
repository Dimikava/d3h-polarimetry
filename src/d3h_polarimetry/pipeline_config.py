"""Configuration and result container types for analysis pipelines."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from .cache import AnalysisCache
from .models import FFTMode, PhysicalModel
from .visualization import FigureStyle


@dataclass
class AnalysisConfig:
    """D3h analysis settings. Window size 10, R² threshold 0.8; frame patching is disabled."""

    r2_threshold: float = 0.8
    window_size: int = 10
    bin_fraction: float = 0.1
    bin_fraction_roimpois: float = 0.05
    roi_histogram_bin_count: Optional[int] = 4
    poi_histogram_bin_count: Optional[int] = None
    bin_fraction_local_roimpois: float = 0.03
    local_hist_tick_count: int = 30
    sigma_threshold: float = 20.0
    patch_data: bool = False
    enable_curve_fit: bool = True
    use_global_optimizer: bool = False
    phi_vmax: Optional[float] = None
    phi_vmin: Optional[float] = None
    a0_mask_percentile: Optional[float] = 50.0
    a0_mask_cutoff: Optional[float] = None
    a0_mask_map: Optional[np.ndarray] = None
    a0_mask_pass_map: Optional[np.ndarray] = None
    temp_ignore_phi_ge_58: bool = False
    temp_fix_twist_harmonization: bool = False
    twist_mask_background_fraction: float = 0.15
    twist_mask_patch_radius: int = 1
    twist_mask_threshold_bright: float = 0.85
    twist_mask_threshold_dim: float = 0.65

    def __post_init__(self) -> None:
        if self.a0_mask_percentile is not None and (
            not 0.0 <= float(self.a0_mask_percentile) <= 100.0
        ):
            raise ValueError(
                "a0_mask_percentile must be between 0 and 100, or None to disable it."
            )


@dataclass
class SessionConfig:
    """
    Complete configuration for an analysis session.

    Acts as a Single Source of Truth for the current run, aggregating input/output
    paths, the chosen physical model, the FFT mode, and algorithmic hyperparameters.

    Attributes:
        image_dir: Parent directory containing TIFF files.
        save_dir: Destination directory for all results.
        model: The PhysicalModel instance used for analysis.
        fft_mode: The chosen FFTMode.
        analysis: AnalysisConfig for algorithmic parameters.
        style: FigureStyle configuration for visual output.
    """

    image_dir: Path
    save_dir: Path
    model: "PhysicalModel"
    fft_mode: "FFTMode"
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    style: FigureStyle = field(default_factory=FigureStyle)
    figure_format: str = "mixed"

    def __post_init__(self) -> None:
        """Normalize and validate figure output profile."""
        self.figure_format = self.figure_format.lower()
        if self.figure_format not in {"png", "svg", "mixed"}:
            raise ValueError(
                f"Unsupported figure format: {self.figure_format}. Use 'png', 'svg', or 'mixed'."
            )

    def figure_extension(self, kind: str = "raster") -> str:
        """
        Resolve an output extension for a figure category.

        Kinds:
            vector: line/scatter/text-dominant plots such as polar plots
                and histograms.
            overlay: grayscale image plus vector annotations such as ROI
                boxes, POI markers, labels, and scale bars.
            raster: heatmaps/colormaps and image-like outputs.
        """
        if kind not in {"vector", "overlay", "raster"}:
            raise ValueError(f"Unsupported figure kind: {kind}")
        if self.figure_format in {"png", "svg"}:
            return self.figure_format
        if kind in {"vector", "overlay"}:
            return "svg"
        return "png"

    def figure_path(self, stem: str, kind: str = "raster") -> Path:
        """Build an output path for matplotlib-generated figures."""
        directory = self.figure_dir(stem, kind=kind)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{stem}.{self.figure_extension(kind)}"

    def extra_figure_paths(
        self, stem: str, kind: str = "raster"
    ) -> List[Path]:
        """
        Return any companion export paths for the figure.

        In mixed mode, raster-heavy figures keep their PNG primary output and
        also get an SVG companion for post-processing.
        """
        if kind == "raster" and self.figure_format == "mixed":
            primary = self.figure_path(stem, kind=kind)
            svg_path = primary.with_suffix(".svg")
            if svg_path != primary:
                return [svg_path]
        return []

    def figure_dir(self, stem: str, kind: str = "raster") -> Path:
        """Route figure outputs into subfolders by artifact type."""
        if kind == "vector":
            if "histogram" in stem:
                return self.save_dir / "histograms"
            return self.save_dir / "polar_plots"
        if kind == "overlay":
            return self.save_dir / "mean_image"
        if stem.startswith("twist_angle_"):
            return self.save_dir / "twist_angle"
        if stem.endswith("dominant_harmonic_map"):
            return self.save_dir / "misc"
        if "new_r2" in stem:
            return self.save_dir / "new r2"
        if "thgmm2" in stem:
            return self.save_dir / "maps" / "thgmm2"
        if stem.startswith("whole_image_"):
            return self.save_dir / "maps" / "whole_image"
        return self.save_dir / "maps"

    def table_dir(self, category: str = "summaries") -> Path:
        """Return a subdirectory for CSV-like analysis tables."""
        base = self.save_dir / "tables"
        mapping = {
            "summaries": base / "summaries",
            "window_data": base / "window_data",
            "curve_fit_cache": base / "curve_fit_cache",
            "thgmm2": base / "thgmm2",
        }
        path = mapping.get(category, base / category)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mean_image_path(self, filename: str) -> Path:
        """Return a path under the mean-image artifact folder."""
        path = self.save_dir / "mean_image"
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def cache_path(
        self, filename: str = AnalysisCache.DEFAULT_FILENAME
    ) -> Path:
        """Return a path under the cache folder."""
        path = self.save_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path / filename


@dataclass
class LocalStats:
    """
    Statistics for a local region around a POI.

    Attributes:
        mean: Mean orientation in the region.
        std: Standard deviation of orientations.
        min: Minimum orientation value.
        max: Maximum orientation value.
        data: Raw orientation values that passed R² threshold.
    """

    mean: float
    std: float
    min: float
    max: float
    data: List[float]

    @classmethod
    def from_data(cls, data: List[float]) -> Optional["LocalStats"]:
        """Create LocalStats from data, or None if data is empty."""
        if not data:
            return None
        return cls(
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            data=data,
        )


@dataclass
class SlidingWindowMaps:
    """FFT and curve-fit orientation/R² maps with A0 intensity and validity masks."""

    fft_angle: np.ndarray
    fft_r2: np.ndarray
    curve_angle: np.ndarray
    curve_r2: np.ndarray
    a0_intensity_map: Optional[np.ndarray] = None
    a0_intensity_mask: Optional[np.ndarray] = None


@dataclass
class SlidingWindowStats:
    """
    Statistics from sliding window analysis.

    Attributes:
        total_windows: Total number of windows analyzed.
        fft_passed: Number of windows passing R² threshold (FFT).
        curve_passed: Number of windows passing R² threshold (curve).
        fft_orientations: List of FFT orientations passing threshold.
        curve_orientations: List of curve orientations passing threshold.
    """

    total_windows: int
    fft_passed: int
    curve_passed: int
    fft_orientations: List[float]
    curve_orientations: List[float]

    def get_stats(
        self, method: str = "fft"
    ) -> Tuple[float, float, float, float]:
        """
        Get min, max, mean, std for a method.

        Returns:
            Tuple of (min, max, mean, std) or (0, 0, 0, 0) if no data.
        """
        data = (
            self.fft_orientations
            if method == "fft"
            else self.curve_orientations
        )
        if not data:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(np.min(data)),
            float(np.max(data)),
            float(np.mean(data)),
            float(np.std(data)),
        )
