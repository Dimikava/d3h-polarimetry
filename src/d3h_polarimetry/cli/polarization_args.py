"""Argument parser for the polarization CLI."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Sequence


def _positive_presentation_scale(value: str) -> float:
    """Parse a positive presentation scale used for inverse font compensation."""
    try:
        scale = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "presentation scale must be a positive number"
        ) from exc
    if scale <= 0.0:
        raise argparse.ArgumentTypeError(
            "presentation scale must be greater than 0"
        )
    return scale


def _positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI option."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive number"
        ) from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _font_compensation_from_presentation_scale(scale: float) -> float:
    """Return the font multiplier needed after rendering a figure at ``scale``."""
    if scale <= 0.0:
        raise ValueError("presentation scale must be greater than 0")
    return 1.0 / float(scale)


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    """Parse a non-negative floating-point CLI option."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative number"
        ) from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer CLI option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the polarization CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="D3h SHG polarimetry: special ROI/POI workflow with sliding-window maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nExamples:\n  GUI mode (opens folder picker):\n    uv run run_polarization.py\n    \n  Special ROI/POI mode (default):\n    uv run run_polarization.py --images "C:/Data/experiment/tif1"\n\n  Single TIFF mode:\n    uv run run_polarization.py --images "C:/Data/experiment/angles.tif" --tiff-series 0\n    \n  ROI + Manual POIs mode:\n    uv run run_polarization.py --images "C:/Data/experiment/tif1" --twist-angle\n    \n  Custom output directory:\n    uv run run_polarization.py --images "C:/Data/experiment/tif1" --save-dir "C:/Results"\n        ',
    )
    parser.allow_abbrev = False
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Path to a TIFF file or folder containing TIFF images (opens a folder picker if omitted)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save results (default: Desktop/Results/<experiment>/ROIs)",
    )
    parser.add_argument(
        "--tiff-series",
        type=_nonnegative_int,
        default=None,
        help="Zero-based TIFF series index; required when a TIFF contains multiple series.",
    )
    parser.add_argument(
        "--max-input-gib",
        type=_nonnegative_float,
        default=8.0,
        help="Maximum decoded TIFF size in GiB before analysis (default: 8; 0 disables).",
    )
    parser.add_argument(
        "--axis-mapping",
        type=str,
        default=None,
        help="JSON file mapping stored TIFF dimensions to semantic dimensions.",
    )
    parser.add_argument(
        "--axis-preset",
        type=str,
        default=None,
        help="Load a named reusable axis mapping preset.",
    )
    parser.add_argument(
        "--save-axis-preset",
        type=str,
        default=None,
        help="Save the resolved axis mapping as a named reusable preset.",
    )
    parser.add_argument(
        "--axis-select",
        action="append",
        default=[],
        metavar="DIM=VALUE",
        help="Select one coordinate from an extra dimension; may be repeated.",
    )
    parser.add_argument(
        "--axis-gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the axis-mapping window when metadata is ambiguous (default: enabled).",
    )
    parser.add_argument(
        "--preset", type=str, help="Load saved polarization settings by name"
    )
    parser.add_argument(
        "--save-preset",
        type=str,
        help="Save this polarization invocation as a named preset",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List saved polarization presets and exit",
    )
    parser.add_argument("--preset-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--replay-cache",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Replay figures from a saved analysis cache (.npz or results directory). If used without a value, auto-tries the default results folder for the current run settings.",
    )
    parser.add_argument(
        "--reuse-curve-cache",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Keep the interactive pipeline but reuse saved curve-fit window parameters instead of refitting. First version is supported for --roimpois-special.",
    )
    parser.add_argument(
        "--ignore-cache-crop",
        action="store_true",
        help="With --reuse-curve-cache, ignore saved crop bounds and use the full current image coordinate frame.",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        choices=["png", "svg", "mixed"],
        default="mixed",
        help="Figure output profile: png, svg, or mixed (default: mixed)",
    )
    parser.add_argument(
        "--half-polar",
        action="store_true",
        help="Render polar plots over 0-180 degrees instead of the full 0-360 degrees.",
    )
    parser.add_argument(
        "--hist-font-scale",
        type=_positive_presentation_scale,
        default=1.0,
        help="Final presentation scale for histograms; fonts are compensated by 1/scale (default: 1.0).",
    )
    parser.add_argument(
        "--map-font-scale",
        type=_positive_presentation_scale,
        default=1.0,
        help="Final presentation scale for colormaps/maps; fonts are compensated by 1/scale (default: 1.0).",
    )
    parser.add_argument(
        "--polar-font-scale",
        type=_positive_presentation_scale,
        default=None,
        help="Final presentation scale for polar plots; fonts are compensated by 1/scale. If omitted, special SVG modes keep their automatic compensation.",
    )
    parser.add_argument(
        "--inspect-histograms",
        action="store_true",
        help="Interactively preview and edit each single-panel histogram before saving.",
    )
    parser.add_argument(
        "--roi-hist-bins",
        type=_positive_int,
        default=None,
        help="Explicit ROI histogram bin count for the special workflow.",
    )
    parser.add_argument(
        "--poi-hist-bins",
        type=_positive_int,
        default=None,
        help="Explicit POI histogram bin count for the special workflow.",
    )
    parser.add_argument(
        "--roimpois-special",
        action="store_true",
        help="Enable ROI + Manual POIs special mode (letter ROIs, global POI numbers, whole-image colormaps)",
    )
    parser.add_argument(
        "--dat-file",
        type=str,
        default=None,
        help="Path to .dat metadata file for scale bar calculation",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Deprecated: crop selection now opens before ROI selection by default.",
    )
    parser.add_argument(
        "--twist-angle",
        action="store_true",
        help="Perform twist angle arithmetic calculation between pairs of manual POIs (select POI pairs in each ROI)",
    )
    parser.add_argument(
        "--temp",
        action="store_true",
        help="Histogram-only filter for roimpois modes: ignore phi/orientation values >= 58°",
    )
    parser.add_argument(
        "--temp-fix",
        action="store_true",
        help="Enable temporary ROI-A twist-angle harmonization for both twist colormaps and histograms.",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.8,
        help="R² quality threshold (default: 0.8)",
    )
    parser.add_argument(
        "--a0-mask-percentile",
        type=float,
        default=50.0,
        help="Whole-image A0/DC percentile cutoff; 50 keeps pixels with A0 >= the image median (default: 50.0).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=180,
        help="Number of TIFF-folder images to load for analysis (default: 180; folders only)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--d3hfixed",
        action="store_true",
        help="Use d3hfixed (default): FFT bin 4, B + A*cos(3*phi - 2*alpha)^2",
    )
    model_group.add_argument(
        "--d3hrotating",
        action="store_true",
        help="Use d3hrotating model: FFT index 6, cos²(3(φ-α)) curve fit",
    )
    parser.add_argument(
        "--optimizer",
        action="store_true",
        help="Enable global optimizer (differential_evolution) as fallback when curve fitting fails",
    )
    parser.add_argument(
        "--fft-only",
        action="store_true",
        help="Skip curve fitting entirely and run FFT analysis only.",
    )
    parser.add_argument(
        "--phi-vmin",
        type=float,
        default=None,
        help="Lower colormap limit for phi/orientation maps across pipelines (default: auto).",
    )
    parser.add_argument(
        "--phi-vmax",
        type=float,
        default=None,
        help="Upper colormap limit for phi/orientation maps across pipelines (default: auto).",
    )
    parser.add_argument(
        "--window-size",
        type=_positive_int,
        default=10,
        help="Sliding-window side length in pixels (default: 10; stride: 1).",
    )
    parser.set_defaults(roimpois_special=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments with validation."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    args._num_frames_explicit = any(
        (
            token == "--num-frames" or token.startswith("--num-frames=")
            for token in raw_argv
        )
    )
    return args
