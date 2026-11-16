"""Optional developer check: compare against a separate source-toolkit checkout.

Run: uv run python scripts/compare_reference.py /path/to/tiff-analyzer
The original toolkit is needed only for this check, never for application use.
"""

import argparse
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference_checkout", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(root / "src"),
    str(args.reference_checkout.resolve() / "src"),
]
from tiff_analyzer.models import (
    SHGd3hModel,
    SHGCos2MovModel,
    FFTMode as OldMode,
)
from tiff_analyzer.polarization import PolarizationAnalyzer as OldAnalyzer
from tiff_analyzer.pipeline_config import AnalysisConfig as OldConfig
from tiff_analyzer.pipeline_sliding import SlidingWindowAnalyzer as OldSliding
from d3h_polarimetry.models import D3hFixedModel, D3hRotatingModel, FFTMode
from d3h_polarimetry.polarization import PolarizationAnalyzer
from d3h_polarimetry.pipeline_config import AnalysisConfig
from d3h_polarimetry.pipeline_sliding import SlidingWindowAnalyzer

angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
for old_model, new_model, old_mode, new_mode in [
    (SHGd3hModel(), D3hFixedModel(), OldMode.SHGd3hanal, FFTMode.D3HFIXED),
    (
        SHGCos2MovModel(),
        D3hRotatingModel(),
        OldMode.SHGd3hanalMOV,
        FFTMode.D3HROTATING,
    ),
]:
    for phi in [0, 12, 37, 59]:
        signal = old_model.model_function(angles, 500, 120, np.deg2rad(phi))
        np.testing.assert_array_equal(
            signal, new_model.model_function(angles, 500, 120, np.deg2rad(phi))
        )
        for noise in [0, 120]:
            observed = signal + np.random.default_rng(0).normal(
                0, noise, signal.shape
            )
            old = OldAnalyzer(old_model, old_mode).analyze_signal(
                observed, angles, patch_data=False
            )
            new = PolarizationAnalyzer(new_model, new_mode).analyze_signal(
                observed, angles
            )
            for method in ["fft", "fit"]:
                assert set(old[method]) == set(new[method])
                for key in old[method]:
                    np.testing.assert_allclose(
                        new[method][key],
                        old[method][key],
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )
            if noise == 0:
                print(
                    new_mode.value,
                    phi,
                    "fit",
                    new["fit"]["angle_output_deg"],
                    new["fit"]["A"],
                    new["fit"]["B"],
                    new["fit"]["r_squared"],
                )
    signal = old_model.model_function(angles, 500, 120, np.deg2rad(23))
    stack = np.broadcast_to(signal[:, None, None], (180, 5, 6)).copy()
    old = OldSliding(
        OldAnalyzer(old_model, old_mode),
        OldConfig(window_size=3, a0_mask_percentile=None, patch_data=False),
    ).analyze(stack, angles)
    new = SlidingWindowAnalyzer(
        PolarizationAnalyzer(new_model, new_mode),
        AnalysisConfig(window_size=3, a0_mask_percentile=None),
    ).analyze(stack, angles)
    for key in [
        "fft_angle",
        "fft_r2",
        "curve_angle",
        "curve_r2",
        "a0_intensity_map",
        "a0_intensity_mask",
    ]:
        np.testing.assert_allclose(
            getattr(new[0], key),
            getattr(old[0], key),
            atol=1e-12,
            equal_nan=True,
        )
    assert old[1] == new[1] or vars(old[1]) == vars(new[1])
    assert len(old[2]) == len(new[2])
print("PASS: 16 reference signal comparisons and both window-map comparisons")
