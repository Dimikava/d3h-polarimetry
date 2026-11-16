"""SHG/common FFT and curve-fit methods for PolarizationAnalyzer."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.fft import rfft
from scipy.optimize import OptimizeWarning, curve_fit, differential_evolution
from .save import CurveFitResult, FFTResult

logger = logging.getLogger(__name__)


class SHGPolarizationMixin:
    def _extract_phase(
        self,
        raw_phase: float,
        phase_offset: float,
        a_0: float,
        a_n: float,
        angles: np.ndarray,
        signal: np.ndarray,
        theoretical_func,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Extract phase from FFT and resolve sign ambiguity.

        Applies the mode-specific phase offset (0 for cosine, −π/2 for sine),
        then compares +phase and −phase predictions via R² to pick the best.
        """
        precise_phase = np.mod(raw_phase + phase_offset, 2 * np.pi) / 6.0
        pred_pos = theoretical_func(a_0, a_n, precise_phase, angles)
        r_pos = self.r_squared(signal, pred_pos)
        pred_neg = theoretical_func(a_0, a_n, -precise_phase, angles)
        r_neg = self.r_squared(signal, pred_neg)
        if r_neg > r_pos:
            precise_phase = (
                np.abs(np.mod(raw_phase + phase_offset, 2 * np.pi) - 2 * np.pi)
                / 6.0
            )
            return (precise_phase, r_neg, pred_neg)
        return (precise_phase, r_pos, pred_pos)

    def _analyze_fft(
        self, signal: np.ndarray, angles: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform FFT-based orientation analysis.

        Uses _FFT_CONFIG to determine harmonic index, theoretical function,
        and phase offset per FFT mode.
        """
        fft_result = rfft(signal)
        fft_phases = np.angle(fft_result)
        harmonic_idx, theoretical_func, phase_offset = self._FFT_CONFIG[
            self.fft_mode
        ]
        if len(fft_result) <= harmonic_idx:
            logger.warning(
                f"Signal too short for FFT analysis (need index {harmonic_idx})"
            )
            return {"error": f"Signal too short for FFT index {harmonic_idx}"}
        a_n = 2 * np.abs(fft_result[harmonic_idx]) / len(angles)
        a_0 = np.abs(fft_result[0]) / len(angles)
        raw_phase = fft_phases[harmonic_idx]
        precise_phase, r_fft, best_pred = self._extract_phase(
            raw_phase, phase_offset, a_0, a_n, angles, signal, theoretical_func
        )
        result = FFTResult(
            a_0=a_0,
            a_n=a_n,
            phase=precise_phase,
            phase_deg=np.degrees(precise_phase),
            r_squared=r_fft,
            prediction=best_pred,
            harmonic_index=harmonic_idx,
        )
        logger.debug(
            f"FFT ({self.fft_mode.value}): phase={result.phase_deg:.2f}°, R²={result.r_squared:.4f}"
        )
        return result.to_dict()

    def _build_fit_result(
        self,
        A_norm: float,
        B_norm: float,
        phi_fit: float,
        sig_range: float,
        sig_min: float,
        angles: np.ndarray,
        signal: np.ndarray,
    ) -> CurveFitResult:
        """
        Rescale normalized fit parameters and build a CurveFitResult.
        """
        A_fit = A_norm * sig_range
        B_fit = B_norm * sig_range + sig_min
        fit_pred = self.model.model_function(angles, A_fit, B_fit, phi_fit)
        r_sq = self.r_squared(signal, fit_pred)
        return CurveFitResult(
            A=A_fit,
            B=B_fit,
            phi=phi_fit,
            angle_output_deg=np.degrees(phi_fit),
            r_squared=r_sq,
            prediction=fit_pred,
        )

    def _analyze_curve_fit(
        self,
        signal: np.ndarray,
        angles: np.ndarray,
        fft_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform nonlinear curve fitting to extract orientation.

        Pipeline:
            1. Try curve_fit with FFT phase only as an initial guess (A, B, phi free)
            2. If local fitting fails, fall back to a global optimizer (if enabled)
        """
        sig_min = signal.min()
        sig_max = signal.max()
        sig_range = sig_max - sig_min
        if sig_range == 0:
            return {"error": "Flat signal (zero range)", "r_squared": 0.0}
        signal_norm = (signal - sig_min) / sig_range
        amplitude_guess = max(
            float(np.max(signal_norm) - np.min(signal_norm)), 0.001
        )
        offset_guess = float(np.min(signal_norm))
        has_fft_phase = fft_result and "phase" in fft_result
        if has_fft_phase:
            phi_init = fft_result["phase"]
        else:
            phi_init = self.model.get_initial_guess()[2]
        phi_guesses = np.linspace(0.0, np.pi / 3.0, 18, endpoint=False)
        if has_fft_phase:
            phi_guesses = np.unique(np.concatenate(([phi_init], phi_guesses)))
        best_result = None
        for guess in phi_guesses:
            try:
                params, _ = curve_fit(
                    self.model.model_function,
                    angles,
                    signal_norm,
                    p0=[amplitude_guess, offset_guess, float(guess)],
                    bounds=([0, 0, 0], [10, 10, np.pi / 3]),
                    maxfev=2000,
                )
                result = self._build_fit_result(
                    *params, sig_range, sig_min, angles, signal
                )
                if (
                    best_result is None
                    or result.r_squared > best_result.r_squared
                ):
                    best_result = result
            except (RuntimeError, ValueError, OptimizeWarning):
                continue
        if best_result is not None:
            logger.debug(
                f"Curve fit: angle={best_result.angle_output_deg:.2f}°, R²={best_result.r_squared:.4f}"
            )
            return best_result.to_dict()
        logger.warning("Curve fitting failed for all starting values")
        if self.use_global_optimizer:
            try:
                logger.info(
                    "Attempting global optimizer (differential_evolution)..."
                )
                bounds_full = self.model.get_bounds()
                de_bounds = list(zip(bounds_full[0], bounds_full[1]))
                de_bounds_norm = [(0, 10), (0, 10), de_bounds[2]]

                def objective(params):
                    pred = self.model.model_function(angles, *params)
                    return np.sum((signal_norm - pred) ** 2)

                de_result = differential_evolution(
                    objective,
                    bounds=de_bounds_norm,
                    maxiter=100,
                    tol=1e-06,
                    seed=42,
                )
                result = self._build_fit_result(
                    *de_result.x, sig_range, sig_min, angles, signal
                )
                logger.info(
                    f"Global optimizer succeeded: angle={result.angle_output_deg:.2f}°, R²={result.r_squared:.4f}"
                )
                return result.to_dict()
            except Exception as e3:
                logger.warning(f"Global optimizer also failed: {e3}")
        return {"error": "All fitting strategies failed", "r_squared": 0.0}
