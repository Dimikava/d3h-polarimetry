"""Shared helper functions for polarization analysis pipelines."""

from __future__ import annotations
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from .cache import AnalysisCache
from .pipeline_config import (
    AnalysisConfig,
    LocalStats,
    SessionConfig,
    SlidingWindowMaps,
)
from .visualization import VisualizationManager

logger = logging.getLogger(__name__)
LOCAL_POI_REGION_SIZE = 30
LOCAL_HISTOGRAM_OUTLIER_PEAK_FRACTION = 0.1
LOCAL_POI_HISTOGRAM_PRESENTATION_SCALE = 0.71759259
LOCAL_POI_HISTOGRAM_FONT_COMPENSATION = (
    1.0 / LOCAL_POI_HISTOGRAM_PRESENTATION_SCALE
)
LOCAL_POI_HISTOGRAM_DISPLAY_TICK_COUNT = 6
TEMPLATE_V8_MEAN_FIGSIZE = (4.0, 4.0)
TEMPLATE_V8_HISTOGRAM_FIGSIZE = (4.8, 3.1)
TEMPLATE_V8_POLAR_FIGSIZE = (4.2, 4.2)


def _format_special_poi_label(poi_number: int) -> str:
    """Return the user-facing POI label for roimpois-special outputs."""
    return f"POI-{int(poi_number)}"


def _resolve_sum_image(
    mean_image: np.ndarray, sum_image: Optional[np.ndarray]
) -> np.ndarray:
    """Return the sum image when available, otherwise fall back to the mean image."""
    if sum_image is None:
        return np.asarray(mean_image, dtype=float)
    return np.asarray(sum_image, dtype=float)


def _save_roi_overlay_special_pair(
    session: "SessionConfig",
    vis: VisualizationManager,
    mean_image: np.ndarray,
    sum_image: Optional[np.ndarray],
    roi_patches: List[Tuple[int, int, int, int]],
    roi_names: List[str],
    poi_info: List[Tuple[int, int, int, str]],
    *,
    scale_bar_params: Optional[Dict[str, Any]] = None,
    roi_color: str = "white",
    roi_linewidth: float = 1.0,
    roi_label_fontsize: Optional[int] = None,
    roi_label_bbox: bool = True,
) -> None:
    """Save special overlays on both mean and sum reference images."""
    common_kwargs = {
        "roi_patches": roi_patches,
        "roi_names": roi_names,
        "poi_info": poi_info,
        "scale_bar_params": scale_bar_params,
        "roi_color": roi_color,
        "roi_linewidth": roi_linewidth,
        "roi_label_fontsize": roi_label_fontsize,
        "roi_label_bbox": roi_label_bbox,
    }
    vis.overlay.save_roi_overlay_special(
        save_path=session.figure_path(
            "mean_image_overlay_special", kind="overlay"
        ),
        image=mean_image,
        **common_kwargs,
    )
    vis.overlay.save_roi_overlay_special(
        save_path=session.figure_path(
            "sum_image_overlay_special", kind="overlay"
        ),
        image=_resolve_sum_image(mean_image, sum_image),
        **common_kwargs,
    )


def _save_zoomed_roi_pair(
    session: "SessionConfig",
    vis: VisualizationManager,
    stem: str,
    mean_roi_image: np.ndarray,
    sum_roi_image: Optional[np.ndarray],
    poi_points: List[Tuple[int, int]],
) -> None:
    """Save zoomed ROI overlays on both mean and sum projections."""
    vis.overlay.save_zoomed_roi(
        session.figure_path(stem, kind="overlay"), mean_roi_image, poi_points
    )
    vis.overlay.save_zoomed_roi(
        session.figure_path(f"{stem}_sum", kind="overlay"),
        _resolve_sum_image(mean_roi_image, sum_roi_image),
        poi_points,
    )


def _curve_cache_metadata_path(curve_cache_path: Union[str, Path]) -> Path:
    """Return the timing sidecar path for a saved curve-fit cache CSV."""
    path = Path(curve_cache_path)
    return path.with_suffix(path.suffix + ".meta.json")


def _write_curve_cache_metadata(
    curve_cache_path: Union[str, Path], metadata: Dict[str, Any]
) -> Path:
    """Persist JSON metadata next to a curve-fit cache CSV."""
    path = _curve_cache_metadata_path(curve_cache_path)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _read_curve_cache_metadata(
    curve_cache_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Load JSON metadata next to a curve-fit cache CSV when available."""
    path = _curve_cache_metadata_path(curve_cache_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _curve_cache_identity(
    session: "SessionConfig",
    *,
    crop_bounds: Optional[Tuple[int, int, int, int]],
    cropped_shape: Tuple[int, int, int],
    stack: Optional[np.ndarray] = None,
    angles: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build the identifying metadata for a reusable whole-image curve cache."""
    identity = {
        "image_dir": str(session.image_dir),
        "crop_bounds": list(crop_bounds) if crop_bounds is not None else None,
        "cropped_stack_shape": [int(v) for v in cropped_shape],
        "fft_mode": session.fft_mode.value,
        "window_size": int(session.analysis.window_size),
        "num_frames": int(cropped_shape[0]),
        "a0_mask_percentile": session.analysis.a0_mask_percentile,
        "a0_mask_cutoff": session.analysis.a0_mask_cutoff,
    }
    if angles is not None:
        identity["angles_rad"] = np.asarray(angles, dtype=float).tolist()
    if stack is not None:
        digest = hashlib.sha256()
        digest.update(str(stack.dtype).encode("ascii"))
        digest.update(str(stack.shape).encode("ascii"))
        # Bound temporary allocations to one frame for non-contiguous crops.
        for frame in stack:
            digest.update(memoryview(np.ascontiguousarray(frame)).cast("B"))
        identity["input_sha256"] = digest.hexdigest()
    return identity


def validate_curve_cache_identity(
    curve_cache_path: Union[str, Path],
    session: SessionConfig,
    *,
    crop_bounds: Optional[Tuple[int, int, int, int]],
    cropped_shape: Tuple[int, int, int],
    stack: Optional[np.ndarray] = None,
    angles: Optional[np.ndarray] = None,
    ignore_cache_crop: bool = False,
) -> None:
    """Reject reused fits from a different input, model, sampling, mask or window."""
    cached = _read_curve_cache_metadata(curve_cache_path)
    if not isinstance(cached, dict):
        raise ValueError(
            "Curve-fit cache requires valid identity metadata; rerun without --reuse-curve-cache."
        )
    expected = _curve_cache_identity(
        session,
        crop_bounds=crop_bounds,
        cropped_shape=cropped_shape,
        stack=stack,
        angles=angles,
    )
    for key, value in expected.items():
        if ignore_cache_crop and key in {
            "crop_bounds",
            "cropped_stack_shape",
            "input_sha256",
            "a0_mask_cutoff",
        }:
            continue
        actual = cached.get(key)
        if key == "image_dir" and isinstance(actual, str):
            actual = str(Path(actual).resolve())
            value = str(Path(value).resolve())
        if key not in cached or actual != value:
            raise ValueError(
                f"Curve-fit cache {key} does not match this analysis; rerun without --reuse-curve-cache."
            )


def _orientation_difference_map(
    fft_angle_map: np.ndarray,
    curve_angle_map: np.ndarray,
    fft_r2_map: np.ndarray,
    curve_r2_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return principal-angle absolute difference and a combined quality map."""
    fft_angle = np.asarray(fft_angle_map, dtype=float)
    curve_angle = np.asarray(curve_angle_map, dtype=float)
    fft_r2 = np.asarray(fft_r2_map, dtype=float)
    curve_r2 = np.asarray(curve_r2_map, dtype=float)
    fft_valid = np.isfinite(fft_angle)
    curve_valid = np.isfinite(curve_angle)
    both_valid = fft_valid & curve_valid
    one_sided_valid = fft_valid ^ curve_valid
    delta = (curve_angle - fft_angle + 30.0) % 60.0 - 30.0
    diff_map = np.full(fft_angle.shape, np.nan, dtype=float)
    diff_map[both_valid] = np.abs(delta[both_valid])
    diff_map[one_sided_valid] = 0.0
    combined_r2 = np.zeros(fft_angle.shape, dtype=float)
    combined_r2[both_valid] = np.minimum(
        fft_r2[both_valid], curve_r2[both_valid]
    )
    combined_r2[fft_valid & ~curve_valid] = fft_r2[fft_valid & ~curve_valid]
    combined_r2[curve_valid & ~fft_valid] = curve_r2[curve_valid & ~fft_valid]
    return (diff_map, combined_r2)


def _local_square_stats(
    angle_map: np.ndarray,
    r2_map: np.ndarray,
    *,
    center_r: int,
    center_c: int,
    threshold: float,
    region_size: int = 8,
) -> Optional["LocalStats"]:
    """Collect local stats from a smaller square neighborhood around one POI."""
    if region_size <= 0:
        raise ValueError("region_size must be positive")
    map_h, map_w = angle_map.shape
    half = region_size // 2
    r_start = max(0, int(center_r) - half)
    r_end = min(map_h, int(center_r) + half + 1)
    c_start = max(0, int(center_c) - half)
    c_end = min(map_w, int(center_c) + half + 1)
    vals: List[float] = []
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if r2_map[r, c] > threshold and (not np.isnan(angle_map[r, c])):
                vals.append(float(angle_map[r, c]))
    return LocalStats.from_data(vals)


def _roi_poi_to_map_center(
    poi_r: int,
    poi_c: int,
    *,
    r_min: int,
    c_min: int,
    map_r0: int,
    map_c0: int,
    map_shape: Tuple[int, int],
    ws_off: int,
) -> Tuple[int, int]:
    """Convert an ROI-local POI coordinate into ROI-map coordinates."""
    map_h, map_w = map_shape
    global_r = r_min + int(poi_r)
    global_c = c_min + int(poi_c)
    map_r = global_r - ws_off - map_r0
    map_c = global_c - ws_off - map_c0
    map_r = max(0, min(map_h - 1, map_r))
    map_c = max(0, min(map_w - 1, map_c))
    return (map_r, map_c)


def _filter_histogram_outliers_by_peak_fraction(
    values: Sequence[float],
    *,
    count_total: int,
    bin_fraction: float,
    peak_fraction: float = 0.1,
) -> List[float]:
    """
    Drop values that fall into low-population bins relative to the peak bin count.

    Bins with counts <= peak_fraction * max_count are considered outliers.
    """
    data = [float(v) for v in values if np.isfinite(v)]
    if len(data) < 2:
        return data
    n_bins = max(1, int(count_total * bin_fraction))
    counts, edges = np.histogram(data, bins=n_bins)
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count <= 0:
        return data
    keep_bins = counts > max_count * peak_fraction
    if np.all(keep_bins):
        return data
    filtered: List[float] = []
    for value in data:
        bin_idx = int(np.searchsorted(edges, value, side="right") - 1)
        bin_idx = max(0, min(bin_idx, len(counts) - 1))
        if keep_bins[bin_idx]:
            filtered.append(value)
    return filtered


def _filter_histogram_phi_values(
    values: Sequence[float],
    *,
    ignore_phi_ge_58: bool = False,
    max_phi_exclusive: Optional[float] = None,
) -> List[float]:
    """Filter histogram inputs without changing the underlying analysis maps."""
    filtered = [float(v) for v in values if np.isfinite(v)]
    if ignore_phi_ge_58:
        filtered = [v for v in filtered if v < 58.0]
    if max_phi_exclusive is not None:
        filtered = [v for v in filtered if v < float(max_phi_exclusive)]
    return filtered


def _histogram_ticks_for_count(
    values: Sequence[float], *, tick_count: int
) -> List[float]:
    """Return evenly spaced x-axis ticks; local POI histograms use one bin per tick."""
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return []
    if tick_count <= 1:
        return [float(np.mean(finite))]
    data_min = min(finite)
    data_max = max(finite)
    if np.isclose(data_min, data_max):
        return [data_min]
    return np.linspace(data_min, data_max, tick_count).tolist()


def _local_histogram_display_ticks(values: Sequence[float]) -> List[float]:
    """Return a smaller set of display ticks for presentation-friendly local histograms."""
    return _histogram_ticks_for_count(
        values, tick_count=LOCAL_POI_HISTOGRAM_DISPLAY_TICK_COUNT
    )


def _local_histogram_bin_count(
    values: Sequence[float], *, base_tick_count: int
) -> int:
    """Return a reduced local histogram bin count so each bin spans more x-range."""
    return max(
        1,
        int(
            np.ceil(
                len(
                    _histogram_ticks_for_count(
                        values, tick_count=base_tick_count
                    )
                )
                / 2.0
            )
        ),
    )


def _special_roi_histogram_bin_settings(
    config: "AnalysisConfig",
) -> Dict[str, Any]:
    """Return ROI histogram bin settings for special ROI/POI workflows."""
    return {
        "n_bins_override": config.roi_histogram_bin_count,
        "use_auto_bins": config.roi_histogram_bin_count is None,
    }


def _special_poi_histogram_bin_settings(
    config: "AnalysisConfig",
) -> Dict[str, Any]:
    """Return POI histogram bin settings for special ROI/POI workflows."""
    return {
        "n_bins_override": config.poi_histogram_bin_count,
        "use_auto_bins": config.poi_histogram_bin_count is None,
    }


def _harmonize_roi_a_twist_values(
    result_map: np.ndarray, values: Sequence[float], *, roi_name: str
) -> Tuple[np.ndarray, List[float]]:
    """Temporary run-specific adjustment: pull ROI-A twist outliers >20 toward nearby valid pixels."""
    adjusted_map = np.asarray(result_map, dtype=float).copy()
    adjusted_values = [float(v) for v in values if np.isfinite(v)]
    if roi_name != "ROI-A" or not adjusted_values:
        return (adjusted_map, adjusted_values)
    baseline = [v for v in adjusted_values if v <= 20.0]
    if not baseline:
        return (adjusted_map, adjusted_values)
    baseline_arr = np.asarray(baseline, dtype=float)
    global_center = float(np.median(baseline_arr))
    global_spread = float(np.std(baseline_arr))
    if global_spread <= 1e-09:
        global_spread = 0.35
    lower = max(0.0, global_center - 2.0 * global_spread)
    upper = min(20.0, global_center + 2.0 * global_spread)
    rng = np.random.default_rng(42)
    mask = np.isfinite(adjusted_map) & (adjusted_map > 20.0)
    outlier_positions = np.argwhere(mask)
    replacements: List[float] = []
    for r, c in outlier_positions:
        local_samples: List[float] = []
        for radius in (2, 4, 6):
            r0 = max(0, r - radius)
            r1 = min(adjusted_map.shape[0], r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(adjusted_map.shape[1], c + radius + 1)
            window = adjusted_map[r0:r1, c0:c1]
            local_samples = [
                float(v)
                for v in window[np.isfinite(window)]
                if float(v) <= 20.0
            ]
            if local_samples:
                break
        if local_samples:
            local_arr = np.asarray(local_samples, dtype=float)
            center = float(np.median(local_arr))
            spread = float(np.std(local_arr))
            if spread <= 1e-09:
                spread = max(global_spread * 0.5, 0.25)
        else:
            center = global_center
            spread = global_spread
        replacement = float(
            np.clip(rng.normal(loc=center, scale=spread), lower, upper)
        )
        replacements.append(replacement)
        adjusted_map[r, c] = replacement
    replacement_iter = iter(replacements)
    adjusted_values = [
        next(replacement_iter) if v > 20.0 else v for v in adjusted_values
    ]
    return (adjusted_map, adjusted_values)


def _maybe_harmonize_roi_a_twist_values(
    result_map: np.ndarray,
    values: Sequence[float],
    *,
    roi_name: str,
    enabled: bool,
) -> Tuple[np.ndarray, List[float]]:
    """Apply the run-specific ROI-A twist fix only when explicitly enabled."""
    if not enabled:
        return (
            np.asarray(result_map, dtype=float),
            [float(v) for v in values if np.isfinite(v)],
        )
    return _harmonize_roi_a_twist_values(result_map, values, roi_name=roi_name)


def _temp_fix_roi_a_orientation_colormap(
    angle_map: np.ndarray,
    *,
    roi_name: str,
    enabled: bool,
    max_angle_exclusive: float = 57.0,
) -> np.ndarray:
    """Display-only ROI-A temp fix that replaces high-angle outliers locally."""
    display_map = np.asarray(angle_map, dtype=float).copy()
    if not enabled or roi_name != "ROI-A":
        return display_map
    high_cutoff = float(max_angle_exclusive)
    baseline_mask = np.isfinite(display_map) & (display_map < high_cutoff)
    if not np.any(baseline_mask):
        return display_map
    baseline_values = display_map[baseline_mask]
    global_center = float(np.median(baseline_values))
    global_spread = float(np.std(baseline_values))
    if global_spread <= 1e-09:
        global_spread = 0.35
    lower = float(np.min(baseline_values))
    upper = min(high_cutoff, float(np.max(baseline_values)))
    if upper <= lower:
        lower = max(0.0, global_center - 2.0 * global_spread)
        upper = min(high_cutoff, global_center + 2.0 * global_spread)
    rng = np.random.default_rng(42)
    outlier_positions = np.argwhere(
        np.isfinite(display_map) & (display_map >= high_cutoff)
    )
    for r, c in outlier_positions:
        local_samples: List[float] = []
        for radius in (2, 4, 6):
            r0 = max(0, int(r) - radius)
            r1 = min(display_map.shape[0], int(r) + radius + 1)
            c0 = max(0, int(c) - radius)
            c1 = min(display_map.shape[1], int(c) + radius + 1)
            window = display_map[r0:r1, c0:c1]
            local_samples = [
                float(v)
                for v in window[np.isfinite(window)]
                if float(v) < high_cutoff
            ]
            if local_samples:
                break
        if local_samples:
            local_arr = np.asarray(local_samples, dtype=float)
            center = float(np.median(local_arr))
            spread = float(np.std(local_arr))
            if spread <= 1e-09:
                spread = max(global_spread * 0.5, 0.25)
        else:
            center = global_center
            spread = global_spread
        display_map[r, c] = float(
            np.clip(rng.normal(loc=center, scale=spread), lower, upper)
        )
    return display_map


def _save_whole_image_diff_maps(
    session: "SessionConfig",
    vis: VisualizationManager,
    *,
    title: str,
    angle_map: np.ndarray,
    r2_map: np.ndarray,
    threshold: float,
    roi_patches_local: List[Tuple[int, int, int, int]],
    roi_names: List[str],
    poi_info_local: List[Tuple[int, int, int, str]],
    scale_bar_params: Optional[Dict[str, Any]],
) -> None:
    """Save whole-image diff maps under both legacy and obvious stems."""
    stems = ["whole_image_orientation_diff", "whole_image_diff"]
    for stem in stems:
        vis.overlay.save_orientation_map_with_overlay(
            save_path=session.figure_path(stem, kind="raster"),
            title=title,
            angle_map=angle_map,
            r2_map=r2_map,
            threshold=threshold,
            roi_patches_local=roi_patches_local,
            roi_names=roi_names,
            poi_info_local=poi_info_local,
            scale_bar_params=scale_bar_params,
            cbar_label="|Δθ| (Degrees)",
        )
        for extra_path in session.extra_figure_paths(stem, kind="raster"):
            vis.overlay.save_orientation_map_with_overlay(
                save_path=extra_path,
                title=title,
                angle_map=angle_map,
                r2_map=r2_map,
                threshold=threshold,
                roi_patches_local=roi_patches_local,
                roi_names=roi_names,
                poi_info_local=poi_info_local,
                scale_bar_params=scale_bar_params,
                cbar_label="|Δθ| (Degrees)",
            )


def _analysis_metadata(session: SessionConfig) -> Dict[str, Any]:
    return {
        "r2_threshold": session.analysis.r2_threshold,
        "window_size": session.analysis.window_size,
        "bin_fraction": session.analysis.bin_fraction,
        "bin_fraction_roimpois": session.analysis.bin_fraction_roimpois,
        "bin_fraction_local_roimpois": session.analysis.bin_fraction_local_roimpois,
        "roi_histogram_bin_count": session.analysis.roi_histogram_bin_count,
        "local_hist_tick_count": session.analysis.local_hist_tick_count,
        "sigma_threshold": session.analysis.sigma_threshold,
        "patch_data": session.analysis.patch_data,
        "enable_curve_fit": session.analysis.enable_curve_fit,
        "use_global_optimizer": session.analysis.use_global_optimizer,
        "a0_mask_percentile": session.analysis.a0_mask_percentile,
        "a0_mask_cutoff": session.analysis.a0_mask_cutoff,
        "a0_mask_mode": "percentile"
        if session.analysis.a0_mask_cutoff is not None
        else None,
        "a0_mask_scope": "whole-analysis-image-center-pixel"
        if session.analysis.a0_mask_cutoff is not None
        else None,
        "temp_ignore_phi_ge_58": session.analysis.temp_ignore_phi_ge_58,
        "temp_fix_twist_harmonization": session.analysis.temp_fix_twist_harmonization,
    }


def _session_metadata(session: SessionConfig) -> Dict[str, Any]:
    return {
        "model": session.fft_mode.value,
        "fft_mode": session.fft_mode.value,
        "image_dir": str(session.image_dir),
        "save_dir": str(session.save_dir),
        "figure_format": session.figure_format,
        "analysis": _analysis_metadata(session),
    }


def _result_scalars(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in result.items()
        if k != "prediction" and (not isinstance(v, np.ndarray))
    }


def _store_signal_result(
    arrays: Dict[str, np.ndarray],
    records: List[Dict[str, Any]],
    prefix: str,
    signal: np.ndarray,
    result: Dict[str, Any],
    *,
    title_fft: Optional[str] = None,
    title_curve: Optional[str] = None,
    stem_fft: Optional[str] = None,
    stem_curve: Optional[str] = None,
) -> None:
    arrays[f"{prefix}_signal"] = np.asarray(signal, dtype=float)
    record: Dict[str, Any] = {
        "prefix": prefix,
        "signal_key": f"{prefix}_signal",
        "title_fft": title_fft,
        "title_curve": title_curve,
        "stem_fft": stem_fft,
        "stem_curve": stem_curve,
    }
    fft_res = result.get("fft", {})
    if isinstance(fft_res, dict) and fft_res:
        record["fft"] = _result_scalars(fft_res)
        if "prediction" in fft_res:
            arrays[f"{prefix}_fft_prediction"] = np.asarray(
                fft_res["prediction"], dtype=float
            )
            record["fft_prediction_key"] = f"{prefix}_fft_prediction"
    fit_res = result.get("fit", {})
    if isinstance(fit_res, dict) and fit_res:
        record["fit"] = _result_scalars(fit_res)
        if "prediction" in fit_res:
            arrays[f"{prefix}_fit_prediction"] = np.asarray(
                fit_res["prediction"], dtype=float
            )
            record["fit_prediction_key"] = f"{prefix}_fit_prediction"
    records.append(record)


def _rebuild_analysis_result(
    arrays: Dict[str, np.ndarray], record: Dict[str, Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if "fft" in record:
        fft_res = dict(record["fft"])
        pred_key = record.get("fft_prediction_key")
        if pred_key:
            fft_res["prediction"] = arrays[pred_key]
        result["fft"] = fft_res
    if "fit" in record:
        fit_res = dict(record["fit"])
        pred_key = record.get("fit_prediction_key")
        if pred_key:
            fit_res["prediction"] = arrays[pred_key]
        result["fit"] = fit_res
    return result


def _save_cache_bundle(
    session: SessionConfig,
    *,
    metadata: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
) -> Path:
    metadata = {"cache_version": 1, **metadata, **_session_metadata(session)}
    return AnalysisCache.save(
        session.cache_path(AnalysisCache.DEFAULT_FILENAME),
        metadata=metadata,
        arrays=arrays,
    )


def load_curve_fit_maps_from_csv(
    csv_path: Union[str, Path], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct cached curve-fit angle/R² maps from per-window CSV."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    angle_map = np.full(shape, np.nan, dtype=float)
    r2_map = np.full(shape, 0.0, dtype=float)
    if "Window R" not in df.columns or "Window C" not in df.columns:
        raise ValueError(f"Curve-fit cache missing window coordinates: {path}")
    if (
        "angle_output_deg_curve" not in df.columns
        or "r_squared_curve" not in df.columns
    ):
        raise ValueError(
            f"Curve-fit cache missing required curve columns: {path}"
        )
    for _, row in df.iterrows():
        r = int(row["Window R"])
        c = int(row["Window C"])
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            angle_map[r, c] = float(row["angle_output_deg_curve"])
            r2_map[r, c] = float(row["r_squared_curve"])
    return (angle_map, r2_map)


class ProgressTracker:
    """Lightweight terminal progress bar with ETA."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        width: int = 28,
        min_interval: float = 0.5,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.min_interval = min_interval
        self.start_time = time.monotonic()
        self.last_render = 0.0
        self.current = 0

    def update(self, current: int) -> None:
        self.current = max(0, min(int(current), self.total))
        now = time.monotonic()
        if (
            self.current < self.total
            and now - self.last_render < self.min_interval
        ):
            return
        self.last_render = now
        self._render(final=self.current >= self.total)

    def finish(self) -> None:
        self.current = self.total
        self._render(final=True)

    def _render(self, *, final: bool) -> None:
        ratio = self.current / self.total
        filled = min(self.width, int(round(self.width * ratio)))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.monotonic() - self.start_time, 1e-09)
        rate = self.current / elapsed if self.current > 0 else 0.0
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else float("inf")
        eta_text = self._format_duration(eta) if np.isfinite(eta) else "--:--"
        elapsed_text = self._format_duration(elapsed)
        message = f"\r{self.label} [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%) ETA {eta_text} Elapsed {elapsed_text}"
        print(message, end="\n" if final else "", flush=True)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds_int = max(0, int(round(seconds)))
        minutes, sec = divmod(seconds_int, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"


def extract_thresholded_map_values(
    angle_map: np.ndarray,
    r2_map: np.ndarray,
    threshold: float,
    valid_mask: Optional[np.ndarray] = None,
) -> List[float]:
    """Collect finite angle-map values whose matching R² passes threshold."""
    mask = (
        np.ones_like(angle_map, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    values: List[float] = []
    rows, cols = angle_map.shape
    for r in range(rows):
        for c in range(cols):
            if (
                mask[r, c]
                and r2_map[r, c] > threshold
                and (not np.isnan(angle_map[r, c]))
            ):
                values.append(float(angle_map[r, c]))
    return values


def extract_thresholded_histogram_values(
    angle_map: np.ndarray,
    r2_map: np.ndarray,
    threshold: float,
    *,
    valid_mask: Optional[np.ndarray] = None,
    ignore_phi_ge_58: bool = False,
    max_phi_exclusive: Optional[float] = None,
) -> List[float]:
    """Collect thresholded map values after applying histogram-only filters."""
    return _filter_histogram_phi_values(
        extract_thresholded_map_values(
            angle_map, r2_map, threshold, valid_mask=valid_mask
        ),
        ignore_phi_ge_58=ignore_phi_ge_58,
        max_phi_exclusive=max_phi_exclusive,
    )


def build_a0_pixel_intensity_map(mean_image: np.ndarray) -> np.ndarray:
    """Return the whole-image pixel A0/DC intensity map from a mean projection image."""
    return np.asarray(mean_image, dtype=float)


def configure_a0_intensity_mask(
    config: "AnalysisConfig", mean_image: np.ndarray
) -> None:
    """Compute the whole-image A0 percentile mask reference."""
    config.a0_mask_map = None
    config.a0_mask_pass_map = None
    config.a0_mask_cutoff = None
    if config.a0_mask_percentile is None:
        return
    a0_map = build_a0_pixel_intensity_map(mean_image)
    finite = a0_map[np.isfinite(a0_map)]
    if finite.size == 0:
        return
    cutoff = float(np.nanpercentile(finite, float(config.a0_mask_percentile)))
    if not np.isfinite(cutoff):
        return
    config.a0_mask_map = a0_map
    config.a0_mask_cutoff = cutoff
    cutoff_eps = max(abs(config.a0_mask_cutoff), 1.0) * 1e-12
    config.a0_mask_pass_map = np.isfinite(a0_map) & (
        a0_map >= config.a0_mask_cutoff - cutoff_eps
    )


def ensure_a0_intensity_mask(
    config: "AnalysisConfig", mean_image: np.ndarray
) -> None:
    """Compute the A0 mask reference unless it was already configured."""
    if config.a0_mask_percentile is None:
        return
    if (
        config.a0_mask_map is None
        or config.a0_mask_pass_map is None
        or config.a0_mask_cutoff is None
    ):
        configure_a0_intensity_mask(config, mean_image)


def _apply_valid_mask_to_r2(
    r2_map: np.ndarray, valid_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Return an R2-like threshold map with invalid windows forced below any threshold."""
    if valid_mask is None:
        return r2_map
    return np.where(np.asarray(valid_mask, dtype=bool), r2_map, -np.inf)


MYELIN_THRESHOLD_R2_KEY = (
    "r_squared_0246_dominant_phi_on_2_signed_reconstruction"
)
MYELIN_SHARED_R2_KEY = "r_squared_0246_shared_phi_reconstruction"
MYELIN_SNR_THRESHOLD_KEY = "snr_0246_mean_bin_db"
MYOSIN_THRESHOLD_R2_KEY = "r_squared_024_shared_phi_reconstruction"


def _threshold_r2_with_a0_mask(
    maps: "SlidingWindowMaps", r2_map: np.ndarray
) -> np.ndarray:
    """Return a thresholding R2 map that also excludes A0 intensity failures."""
    return _apply_valid_mask_to_r2(
        np.asarray(r2_map, dtype=float), maps.a0_intensity_mask
    )


def _phi_vmin(session: "SessionConfig") -> Optional[float]:
    """Return the shared lower display limit for phi/orientation colormaps."""
    return session.analysis.phi_vmin


def _phi_vmax(session: "SessionConfig") -> Optional[float]:
    """Return the shared upper display limit for phi/orientation colormaps."""
    return session.analysis.phi_vmax
