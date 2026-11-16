"""Numerical behavior and independent expected values."""

import numpy as np
import pytest

from d3h_polarimetry.models import D3hFixedModel, D3hRotatingModel, FFTMode
from d3h_polarimetry.polarization import (
    PolarizationAnalyzer,
    InsufficientDataError,
)
from d3h_polarimetry.pipeline_config import AnalysisConfig, LocalStats
from d3h_polarimetry.pipeline_sliding import SlidingWindowAnalyzer
from d3h_polarimetry.pipeline_arithmetic import ArithmeticProcessor


@pytest.mark.parametrize(
    "model,mode,harmonic",
    [
        (D3hFixedModel(), FFTMode.D3HFIXED, 4),
        (D3hRotatingModel(), FFTMode.D3HROTATING, 6),
    ],
)
@pytest.mark.parametrize("phi", [0.0, 12.0, 37.0, 59.0])
def test_recovers_model_and_coefficients(model, mode, harmonic, phi):
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    signal = model.model_function(angles, 500, 120, np.deg2rad(phi))
    result = PolarizationAnalyzer(model, mode).analyze_signal(signal, angles)
    fft, fit = result["fft"], result["fit"]
    assert fft["harmonic_index"] == harmonic
    assert fft["a_0"] == pytest.approx(370)
    assert fft["a_n"] == pytest.approx(250)
    for actual in (fft["phase_deg"], fit["angle_output_deg"]):
        assert abs((actual - phi + 30) % 60 - 30) < 1e-5
    assert fft["r_squared"] == pytest.approx(1)
    np.testing.assert_allclose(fft["prediction"], signal, atol=1e-8)
    # The inherited fit normalizes by sampled extrema with a nonnegative
    # normalized background. Its clean-signal amplitude bias is preserved.
    assert fit["r_squared"] > 0.9999
    np.testing.assert_allclose(fit["prediction"], signal, atol=2)
    assert fit["A"] == pytest.approx(500, abs=2)
    assert fit["B"] == pytest.approx(120, abs=2)


def test_noisy_rotating_fit():
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    model = D3hRotatingModel()
    signal = model.model_function(angles, 500, 120, np.deg2rad(37))
    signal += np.random.default_rng(0).normal(0, 120, signal.shape)
    fit = PolarizationAnalyzer(model, FFTMode.D3HROTATING).analyze_signal(
        signal, angles
    )["fit"]
    assert abs((fit["angle_output_deg"] - 37 + 30) % 60 - 30) <= 2
    assert fit["r_squared"] > 0.6


@pytest.mark.parametrize(
    "mode,model",
    [
        (FFTMode.D3HFIXED, D3hFixedModel()),
        (FFTMode.D3HROTATING, D3hRotatingModel()),
    ],
)
def test_window_maps_and_fft_only(mode, model):
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    signal = model.model_function(angles, 500, 120, np.deg2rad(23))
    stack = np.broadcast_to(signal[:, None, None], (180, 5, 6)).copy()
    analyzer = PolarizationAnalyzer(model, mode, enable_curve_fit=False)
    maps, stats, rows = SlidingWindowAnalyzer(
        analyzer, AnalysisConfig(window_size=3, a0_mask_percentile=None)
    ).analyze(stack, angles)
    assert maps.fft_angle.shape == (3, 4)
    np.testing.assert_allclose(maps.fft_angle, 23, atol=1e-8)
    np.testing.assert_allclose(maps.fft_r2, 1)
    assert stats.total_windows == stats.fft_passed == len(rows) == 12
    assert stats.curve_passed == 0
    assert np.isnan(maps.curve_angle).all()
    with pytest.raises(ValueError, match="too small"):
        SlidingWindowAnalyzer(analyzer, AnalysisConfig(window_size=7)).analyze(
            stack, angles
        )


def test_twist_arithmetic_mask_threshold_and_uncertainty():
    processor = ArithmeticProcessor(AnalysisConfig(r2_threshold=0.8))
    first = LocalStats.from_data([9, 11])
    second = LocalStats.from_data([28, 32])
    angles = np.array([[10.0, 20.0, 55.0], [15.0, 25.0, 35.0]])
    quality = np.array([[1.0, 0.8, 1.0], [1.0, 1.0, 1.0]])
    mask = np.array([[True, True, True], [False, True, True]])
    result, values, sigma = processor.compute_arithmetic(
        angles, quality, first, second, mask
    )
    np.testing.assert_allclose(
        result, [[40, np.nan, 50], [np.nan, 10, 10]], equal_nan=True
    )
    assert values == [40, 50, 10, 10]
    assert sigma == pytest.approx(2 * np.sqrt(5))


def test_no_implicit_frame_patch():
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    signal = np.arange(360.0, dtype=float)
    result = PolarizationAnalyzer(enable_curve_fit=False).analyze_signal(
        signal, angles
    )
    np.testing.assert_array_equal(result["signal"], signal)
    with pytest.raises(InsufficientDataError):
        PolarizationAnalyzer().analyze_signal(np.array([]), np.array([]))
