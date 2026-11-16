"""Configuration builders for the polarization CLI."""

from __future__ import annotations
import argparse
from d3h_polarimetry.cli.polarization_args import (
    _font_compensation_from_presentation_scale,
)
from d3h_polarimetry.pipeline import AnalysisConfig
from d3h_polarimetry.visualization import FigureStyle

ROIMPOIS_SPECIAL_PRESENTATION_SCALE = 0.28472222
ROIMPOIS_SPECIAL_POLAR_FONT_COMPENSATION = (
    1.0 / ROIMPOIS_SPECIAL_PRESENTATION_SCALE
)


def _build_figure_style(args: argparse.Namespace) -> FigureStyle:
    """Build figure styling with minimal special-mode presentation compensation."""
    style = FigureStyle(
        histogram_font_scale=_font_compensation_from_presentation_scale(
            args.hist_font_scale
        ),
        map_font_scale=_font_compensation_from_presentation_scale(
            args.map_font_scale
        ),
        inspect_histograms=args.inspect_histograms,
    )
    style.polar_half_turn = args.half_polar
    if args.figure_format in {"svg", "mixed"} and (
        args.roimpois_special or getattr(args, "roimpois_special2", False)
    ):
        style.polar_font_scale = ROIMPOIS_SPECIAL_POLAR_FONT_COMPENSATION
    if args.polar_font_scale is not None:
        style.polar_font_scale = _font_compensation_from_presentation_scale(
            args.polar_font_scale
        )
    return style


def _build_analysis_config(args: argparse.Namespace) -> AnalysisConfig:
    """Build analysis configuration, including special-mode histogram defaults."""
    roi_histogram_bin_count = args.roi_hist_bins
    if roi_histogram_bin_count is None and args.roimpois_special:
        roi_histogram_bin_count = None
    return AnalysisConfig(
        r2_threshold=args.r2_threshold,
        window_size=args.window_size,
        patch_data=False,
        enable_curve_fit=not args.fft_only,
        use_global_optimizer=args.optimizer,
        phi_vmin=args.phi_vmin,
        phi_vmax=args.phi_vmax,
        a0_mask_percentile=args.a0_mask_percentile,
        temp_ignore_phi_ge_58=args.temp,
        temp_fix_twist_harmonization=args.temp_fix,
        roi_histogram_bin_count=roi_histogram_bin_count,
        poi_histogram_bin_count=args.poi_hist_bins,
    )
