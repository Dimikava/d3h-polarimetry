"""
TIFF Analyzer - Polarization Analysis Module

This module provides polarization-resolved signal analysis for second harmonic
generation (SHG) microscopy data. It implements FFT-based and curve fitting
methods to extract molecular orientation from angular-dependent intensity signals.

Physics Background:
    In SHG polarimetry, the signal intensity I(α) varies with excitation
    polarization angle α according to the molecular hyperpolarizability tensor.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
from .models import (
    PhysicalModel,
    D3hFixedModel,
    FFTMode,
    d3h_fixed_fft,
    d3h_rotating_fft,
)
from .polarization_shg import SHGPolarizationMixin

logger = logging.getLogger(__name__)


class FitError(Exception):
    """Raised when curve fitting fails."""

    pass


class InsufficientDataError(Exception):
    """Raised when signal data is too short for analysis."""

    pass


class PolarizationAnalyzer(SHGPolarizationMixin):
    """
    Analyzer for polarization-resolved microscopy signals.

    Performs both FFT-based and curve fitting analysis to extract
    molecular orientation from angular intensity data.

    Attributes:
        model: The physical model used for curve fitting.

    Example:
        >>> analyzer = PolarizationAnalyzer()
        >>> angles = np.linspace(0, 2*np.pi, 360)
        >>> signal = np.random.rand(360)  # Replace with real data
        >>> results = analyzer.analyze_signal(signal, angles)
        >>> print(results['fft']['phase_deg'])
    """

    _FFT_CONFIG: Dict[FFTMode, Tuple[int, Any, float]] = {
        FFTMode.D3HFIXED: (4, d3h_fixed_fft, 0.0),
        FFTMode.D3HROTATING: (6, d3h_rotating_fft, 0.0),
    }

    def __init__(
        self,
        model: Optional[PhysicalModel] = None,
        fft_mode: FFTMode = FFTMode.D3HFIXED,
        use_global_optimizer: bool = False,
        enable_curve_fit: bool = True,
    ) -> None:
        """Initialize D3h FFT/curve fitting, optionally with global optimizer fallback."""
        self.model = model or D3hFixedModel()
        self.fft_mode = fft_mode
        self.use_global_optimizer = use_global_optimizer
        self.enable_curve_fit = enable_curve_fit
        logger.debug(
            "Initialized PolarizationAnalyzer with %s, FFT mode: %s, global_optimizer: %s, curve_fit: %s",
            self.model.name,
            fft_mode.value,
            use_global_optimizer,
            enable_curve_fit,
        )

    @staticmethod
    def r_squared(signal: np.ndarray, predicted: np.ndarray) -> float:
        """
        Compute the coefficient of determination (R²).
        """
        ss_res = np.sum((signal - predicted) ** 2)
        ss_tot = np.sum((signal - np.mean(signal)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - ss_res / ss_tot

    def analyze_signal(
        self, signal: np.ndarray, angles: np.ndarray, patch_data: bool = False
    ) -> Dict[str, Any]:
        """Analyze one angular signal with the selected D3h model. Input arrays are copied."""
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal, got shape {signal.shape}")
        if angles.ndim != 1:
            raise ValueError(f"Expected 1D angles, got shape {angles.shape}")
        if len(signal) == 0:
            raise InsufficientDataError("Empty signal array")
        if len(signal) != len(angles):
            raise InsufficientDataError(
                f"Signal length ({len(signal)}) must match angles length ({len(angles)})"
            )
        results: Dict[str, Any] = {}
        signal = signal.copy()
        results["signal"] = signal
        results["fft"] = self._analyze_fft(signal, angles)
        results["fit"] = (
            self._analyze_curve_fit(signal, angles, fft_result=results["fft"])
            if self.enable_curve_fit
            else {}
        )
        return results
