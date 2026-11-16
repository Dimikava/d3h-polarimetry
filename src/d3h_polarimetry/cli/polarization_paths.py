"""Path, cache, and projection helpers for the polarization CLI."""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from d3h_polarimetry.cache import AnalysisCache
from d3h_polarimetry.gui import discover_images_gui
from d3h_polarimetry.pipeline import SessionConfig

logger = logging.getLogger(__name__)


def _compute_projection_images(
    stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both mean and sum reference projections for one image stack."""
    stack = np.asarray(stack, dtype=float)
    return (np.mean(stack, axis=0), np.sum(stack, axis=0))


def _save_projection_reference_images(
    session: SessionConfig,
    mean_image: np.ndarray,
    sum_image: np.ndarray,
    *,
    suffix: str,
) -> None:
    """Persist paired mean/sum reference projections under the artifact folder."""
    plt.imsave(
        session.mean_image_path(f"mean_image_{suffix}.png"),
        mean_image,
        cmap="gray",
    )
    plt.imsave(
        session.mean_image_path(f"sum_image_{suffix}.png"),
        sum_image,
        cmap="gray",
    )


def _show_a0_mask_preview(
    session: SessionConfig, mean_image: np.ndarray
) -> None:
    """Save and display the whole-image A0 pass/reject mask before analysis."""
    pass_map = session.analysis.a0_mask_pass_map
    cutoff = session.analysis.a0_mask_cutoff
    if pass_map is None or cutoff is None:
        return
    pass_mask = np.asarray(pass_map, dtype=bool)
    if pass_mask.shape != mean_image.shape:
        return
    overlay = np.zeros((*pass_mask.shape, 4), dtype=float)
    overlay[pass_mask] = (0.0, 0.85, 0.2, 0.42)
    overlay[~pass_mask] = (1.0, 0.05, 0.02, 0.42)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_image, cmap="gray")
    ax.imshow(overlay)
    ax.set_title(
        f"A0 mask preview: green pass, red reject\ncutoff {cutoff:.6g} (whole-image p{session.analysis.a0_mask_percentile:g})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        session.mean_image_path("a0_mask_preview.png"),
        dpi=session.style.dpi,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def validate_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """
    Validate input paths and determine output directory.

    If --images is not provided, opens a GUI folder picker. Explicit paths may
    identify either one TIFF file or a TIFF directory.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (image_path, save_dir).

    Raises:
        SystemExit: If paths are invalid.
    """
    if args.images:
        img_path = Path(args.images)
    else:
        img_path = discover_images_gui()
        if img_path is None:
            logger.error("Could not determine image path. Exiting.")
            sys.exit(1)
    if not img_path.exists():
        logger.error(f"Image path does not exist: {img_path}")
        sys.exit(1)
    if not img_path.is_dir() and img_path.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:
        logger.error(
            f"Image path must be a .tif/.tiff file or a directory: {img_path}"
        )
        sys.exit(1)
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        experiment_folder = img_path.parent.name
        results_base = Path.home() / "Desktop" / "Results"
        save_dir = results_base / experiment_folder / "ROIs"
    if hasattr(args, "roimpois_special") and args.roimpois_special:
        save_dir = save_dir / "roimpois_special"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {save_dir}")
    return (img_path, save_dir)


def resolve_auto_replay_cache(
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path]:
    """
    Resolve the default replay cache location using the same output logic
    as a normal analysis run.
    """
    img_path, save_dir = validate_paths(args)
    return (img_path, save_dir, AnalysisCache.resolve_path(save_dir))


def resolve_auto_curve_cache(
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path]:
    """Resolve default saved curve-fit window table path."""
    img_path, save_dir = validate_paths(args)
    if args.roimpois_special:
        return (
            img_path,
            save_dir,
            save_dir
            / "tables"
            / "curve_fit_cache"
            / "whole_image_curve_fit_window_data.csv",
        )
    raise SystemExit(
        "--reuse-curve-cache auto mode is currently supported only with the special workflow"
    )


def resolve_replay_results_dir(cache_path: Path) -> Path:
    """Resolve the results root that should receive replayed outputs."""
    cache_path = Path(cache_path)
    if (
        cache_path.name == AnalysisCache.DEFAULT_FILENAME
        and cache_path.parent.name == "cache"
    ):
        return cache_path.parent.parent
    return cache_path.parent


def resolve_curve_cache_results_dir(curve_cache_path: Path) -> Path:
    """Resolve the results root from a whole-image curve-cache CSV or its folder."""
    path = Path(curve_cache_path)
    if path.is_dir():
        if (path / "cache" / AnalysisCache.DEFAULT_FILENAME).exists():
            return path
        nested = (
            path
            / "tables"
            / "curve_fit_cache"
            / "whole_image_curve_fit_window_data.csv"
        )
        if nested.exists():
            path = nested
        else:
            candidate = path / "whole_image_curve_fit_window_data.csv"
            if candidate.exists():
                path = candidate
    if (
        path.name == "whole_image_curve_fit_window_data.csv"
        and path.parent.name == "curve_fit_cache"
        and (path.parent.parent.name == "tables")
    ):
        return path.parent.parent.parent
    return path.parent


def resolve_cached_crop_bounds(
    curve_cache_path: Path,
) -> Tuple[int, int, int, int] | None:
    """
    Resolve saved crop bounds for a rerun using an existing curve cache.

    Preference order:
    1. The curve-cache sidecar metadata.
    2. The sibling analysis cache in the same results directory.
    """
    cache_root = resolve_curve_cache_results_dir(curve_cache_path)
    sidecar_path = Path(curve_cache_path)
    if sidecar_path.is_dir():
        sidecar_path = (
            sidecar_path
            / "tables"
            / "curve_fit_cache"
            / "whole_image_curve_fit_window_data.csv"
        )
    sidecar_meta_path = sidecar_path.with_suffix(
        sidecar_path.suffix + ".meta.json"
    )
    if sidecar_meta_path.exists():
        try:
            import json

            metadata = json.loads(
                sidecar_meta_path.read_text(encoding="utf-8")
            )
            crop = metadata.get("crop_bounds")
            if isinstance(crop, list) and len(crop) == 4:
                return tuple((int(v) for v in crop))
        except (OSError, ValueError, TypeError):
            pass
    analysis_cache_path = AnalysisCache.resolve_path(cache_root)
    if analysis_cache_path.exists():
        try:
            metadata, _arrays = AnalysisCache.load(analysis_cache_path)
            crop = metadata.get("crop_bounds")
            if isinstance(crop, list) and len(crop) == 4:
                return tuple((int(v) for v in crop))
        except Exception:
            pass
    return None


def resolve_reuse_curve_cache_crop_policy(
    curve_cache_path: Path, *, ignore_cache_crop: bool = False
) -> tuple[Tuple[int, int, int, int] | None, str]:
    """
    Resolve the enforced crop policy for a reuse-curve-cache run.

    Reused whole-image curve-fit windows are only valid in the same image
    coordinate frame they were generated from. If cached crop bounds exist,
    they are enforced. Otherwise the full image is enforced and interactive
    crop selection is skipped.
    """
    if ignore_cache_crop:
        return (
            None,
            f"Ignoring saved crop metadata for {curve_cache_path}. Enforcing full current image and skipping crop selection to preserve FFT/curve alignment.",
        )
    crop_bounds = resolve_cached_crop_bounds(curve_cache_path)
    if crop_bounds is not None:
        return (
            crop_bounds,
            f"Using cached crop from {curve_cache_path}: rows {crop_bounds[0]}-{crop_bounds[1]}, cols {crop_bounds[2]}-{crop_bounds[3]}",
        )
    return (
        None,
        f"Curve-fit cache {curve_cache_path} has no saved crop metadata. Enforcing full image and skipping crop selection to preserve FFT/curve alignment.",
    )
