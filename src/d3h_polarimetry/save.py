from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FFTResult:
    """Results from FFT-based orientation analysis."""

    a_0: float
    a_n: float
    phase: float
    phase_deg: float
    r_squared: float
    prediction: np.ndarray
    harmonic_index: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for legacy compatibility."""
        return {
            "a_0": self.a_0,
            "a_4": self.a_n,
            "a_n": self.a_n,
            "phase": self.phase,
            "phase_deg": self.phase_deg,
            "r_squared": self.r_squared,
            "prediction": self.prediction,
            "harmonic_index": self.harmonic_index,
        }


@dataclass
class CurveFitResult:
    """Results from curve fitting analysis."""

    A: float
    B: float
    phi: float
    angle_output_deg: float
    r_squared: float
    prediction: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for legacy compatibility."""
        return {
            "A": self.A,
            "B": self.B,
            "phi": self.phi,
            "angle_output_deg": self.angle_output_deg,
            "r_squared": self.r_squared,
            "prediction": self.prediction,
        }


class ResultExporter:
    """
    Exports analysis results to CSV files.

    Attributes:
        save_dir: Directory for output files.
    """

    def __init__(self, save_dir: Path) -> None:
        """
        Initialize the exporter.

        Args:
            save_dir: Directory for output files.
        """
        self.save_dir = save_dir
        logger.debug(f"ResultExporter initialized: {save_dir}")

    def _table_dir(self, category: str) -> Path:
        path = self.save_dir / "tables" / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_roi_results(
        self,
        results: List[Dict[str, Any]],
        filename: str = "analysis_results.csv",
    ) -> Path:
        """Export ROI analysis results to CSV."""
        if not results:
            logger.warning("No ROI results to export")
            return self._table_dir("summaries") / filename
        df = pd.DataFrame(results)
        path = self._table_dir("summaries") / filename
        df.to_csv(path, index=False)
        logger.info(f"Exported {len(results)} ROI results to {path}")
        return path

    def export_poi_results(
        self, results: List[Dict[str, Any]], filename: str = "poi_results.csv"
    ) -> Path:
        """Export POI analysis results to CSV."""
        if not results:
            logger.warning("No POI results to export")
            return self._table_dir("summaries") / filename
        df = pd.DataFrame(results)
        path = self._table_dir("summaries") / filename
        df.to_csv(path, index=False)
        logger.info(f"Exported {len(results)} POI results to {path}")
        return path

    def export_sliding_stats(
        self,
        results: List[Dict[str, Any]],
        filename: str = "sliding_window_stats.csv",
    ) -> Path:
        """Export sliding window statistics to CSV."""
        if not results:
            logger.warning("No sliding window stats to export")
            return self._table_dir("summaries") / filename
        df = pd.DataFrame(results)
        path = self._table_dir("summaries") / filename
        df.to_csv(path, index=False)
        logger.info(f"Exported {len(results)} sliding window stats to {path}")
        return path

    def export_window_data(
        self,
        results: List[Dict[str, Any]],
        roi_id: int,
        prefix: str = "window_data",
    ) -> Path:
        """Export raw sliding window maps for replotting offline."""
        if not results:
            logger.warning(f"No window data to export for ROI {roi_id}")
            return (
                self._table_dir("window_data") / f"{prefix}_roi_{roi_id}.csv"
            )
        df = pd.DataFrame(results)
        path = self._table_dir("window_data") / f"{prefix}_roi_{roi_id}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Exported window data for ROI {roi_id} to {path}")
        return path

    def export_curve_fit_window_data(
        self,
        results: List[Dict[str, Any]],
        roi_id: Optional[int] = None,
        filename: Optional[str] = None,
        prefix: str = "curve_fit_window_data",
    ) -> Path:
        """Export only per-window curve-fit parameters for offline reuse."""
        if not results:
            if filename:
                return self._table_dir("curve_fit_cache") / filename
            if roi_id is None:
                return self._table_dir("curve_fit_cache") / f"{prefix}.csv"
            return (
                self._table_dir("curve_fit_cache")
                / f"{prefix}_roi_{roi_id}.csv"
            )
        df = pd.DataFrame(results)
        keep_cols = [
            col
            for col in df.columns
            if col in {"ROI ID", "Window R", "Window C"}
            or col.endswith("_curve")
            or col == "error_curve"
        ]
        filtered = df[keep_cols].copy()
        if filename:
            path = self._table_dir("curve_fit_cache") / filename
        elif roi_id is None:
            path = self._table_dir("curve_fit_cache") / f"{prefix}.csv"
        else:
            path = (
                self._table_dir("curve_fit_cache")
                / f"{prefix}_roi_{roi_id}.csv"
            )
        filtered.to_csv(path, index=False)
        logger.info(f"Exported curve-fit window data to {path}")
        return path
