"""Special ROI/POI polarization workflow runner."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .pipeline_arithmetic import ArithmeticProcessor
from .pipeline_config import SessionConfig
from .pipeline_helpers import (
    LOCAL_HISTOGRAM_OUTLIER_PEAK_FRACTION,
    LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
    LOCAL_POI_REGION_SIZE,
    _apply_valid_mask_to_r2,
    _curve_cache_identity,
    validate_curve_cache_identity,
    _filter_histogram_outliers_by_peak_fraction,
    _filter_histogram_phi_values,
    _format_special_poi_label,
    _local_square_stats,
    _orientation_difference_map,
    _maybe_harmonize_roi_a_twist_values,
    _phi_vmin,
    _phi_vmax,
    _read_curve_cache_metadata,
    _resolve_sum_image,
    _roi_poi_to_map_center,
    _save_cache_bundle,
    _save_roi_overlay_special_pair,
    _save_whole_image_diff_maps,
    _save_zoomed_roi_pair,
    _special_poi_histogram_bin_settings,
    _special_roi_histogram_bin_settings,
    _store_signal_result,
    _temp_fix_roi_a_orientation_colormap,
    _threshold_r2_with_a0_mask,
    _write_curve_cache_metadata,
    ensure_a0_intensity_mask,
    extract_thresholded_histogram_values,
    extract_thresholded_map_values,
    load_curve_fit_maps_from_csv,
)
from .pipeline_sliding import SlidingWindowAnalyzer
from .polarization import PolarizationAnalyzer
from .save import ResultExporter
from .visualization import VisualizationManager

logger = logging.getLogger(__name__)


def _roi_letter(index: int) -> str:
    """Convert 0-based index to letter: 0→A, 1→B, ..., 25→Z, 26→AA, ..."""
    result = ""
    i = index
    while True:
        result = chr(ord("A") + i % 26) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def run_roimpois_special_pipeline(
    session: SessionConfig,
    stack: np.ndarray,
    mean_image: np.ndarray,
    angles: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    *,
    sum_image: Optional[np.ndarray] = None,
    crop_bounds: Optional[Tuple[int, int, int, int]] = None,
    microns_per_pixel: Optional[float] = None,
    whole_bar_um: float = 10.0,
    roi_bar_um: float = 1.0,
    curve_fit_cache_path: Optional[Path] = None,
    do_twist_angle: bool = False,
    ignore_cache_crop: bool = False,
) -> None:
    """
    Execute the ROI + Manual POIs 'special' pipeline.

    This pipeline differs from run_roimpois_pipeline in:
      - ROIs are named ROI-A, ROI-B, ROI-C, ...
      - POI numbers increment globally across all ROIs (1, 2, 3, ...)
      - Sliding window is computed once on the whole cropped image
      - Per-ROI colormaps are extracted (not re-computed)
      - Mean-image overlay shows ROI names and POI squares with numbers
      - Same overlay persists on orientation colormaps
      - Scale bars are drawn from DAT-file metadata

    Args:
        session: Analysis session configuration.
        stack: 3D image stack (n_frames, height, width).
        mean_image: 2D mean image.
        angles: 1D array of polarisation angles (radians).
        rois: List of ROI bounds, already in stack coordinates:
              (r_min, r_max, c_min, c_max).
        crop_bounds: Optional crop used, for coordinate offset tracking.
        microns_per_pixel: Physical scale from DAT file (None = no scale bar).
        whole_bar_um: Scale bar length in μm for whole-image plots.
        roi_bar_um: Scale bar length in μm for ROI-level plots.
    """
    from .gui import POISelector, POIPairSelector

    ensure_a0_intensity_mask(session.analysis, mean_image)
    if curve_fit_cache_path is not None:
        validate_curve_cache_identity(
            curve_fit_cache_path,
            session,
            crop_bounds=crop_bounds,
            cropped_shape=stack.shape,
            stack=stack,
            angles=angles,
            ignore_cache_crop=ignore_cache_crop,
        )
    analyzer = PolarizationAnalyzer(
        model=session.model,
        fft_mode=session.fft_mode,
        use_global_optimizer=session.analysis.use_global_optimizer,
        enable_curve_fit=session.analysis.enable_curve_fit,
    )
    analyzer_fft_only = PolarizationAnalyzer(
        model=session.model,
        fft_mode=session.fft_mode,
        use_global_optimizer=session.analysis.use_global_optimizer,
        enable_curve_fit=False,
    )
    vis = VisualizationManager(session.style)
    sliding_analyzer = SlidingWindowAnalyzer(analyzer, session.analysis)
    sliding_analyzer_fft_only = SlidingWindowAnalyzer(
        analyzer_fft_only, session.analysis
    )
    arithmetic_processor = ArithmeticProcessor(session.analysis)
    n_rois = len(rois)
    roi_names = [f"ROI-{_roi_letter(i)}" for i in range(n_rois)]
    cache_arrays: Dict[str, np.ndarray] = {
        "mean_image": np.asarray(mean_image, dtype=float),
        "sum_image": _resolve_sum_image(mean_image, sum_image),
        "angles": np.asarray(angles, dtype=float),
    }
    cache_poi_records: List[Dict[str, Any]] = []
    cache_curve_tables: List[Dict[str, Any]] = []
    whole_scale = (
        {"microns_per_pixel": microns_per_pixel, "bar_length_um": whole_bar_um}
        if microns_per_pixel
        else None
    )
    roi_scale = (
        {"microns_per_pixel": microns_per_pixel, "bar_length_um": roi_bar_um}
        if microns_per_pixel
        else None
    )
    global_poi_counter = 0
    all_poi_info: List[Tuple[int, int, int, str]] = []
    per_roi_pois: List[List[Tuple[int, int]]] = []
    per_roi_poi_numbers: List[List[int]] = []
    per_roi_pairs: List[List[Tuple[int, int]]] = []
    for i, (r_min, r_max, c_min, c_max) in enumerate(rois):
        roi_mean = mean_image[r_min:r_max, c_min:c_max]
        print(f"\n--- Select POIs for {roi_names[i]} ---")
        poi_selector = POISelector(
            roi_mean, title=f"Select POIs for {roi_names[i]}"
        )
        manual_points = poi_selector.select_pois()
        per_roi_pois.append(manual_points)
        if do_twist_angle and len(manual_points) >= 2:
            pair_selector = POIPairSelector(manual_points, roi_id=i + 1)
            user_pairs = pair_selector.select_pairs()
        else:
            user_pairs = []
        per_roi_pairs.append(user_pairs)
        numbers = []
        for pr, pc in manual_points:
            global_poi_counter += 1
            global_r, global_c = (r_min + pr, c_min + pc)
            all_poi_info.append(
                (global_r, global_c, global_poi_counter, roi_names[i])
            )
            numbers.append(global_poi_counter)
        per_roi_poi_numbers.append(numbers)
    print(f"\nTotal POIs selected: {global_poi_counter}")
    results_poi = []
    poi_idx = 0
    for i, (r_min, r_max, c_min, c_max) in enumerate(rois):
        roi_stack = stack[:, r_min:r_max, c_min:c_max]
        _, roi_h, roi_w = roi_stack.shape
        half = session.analysis.window_size // 2
        for p_idx, (pr, pc) in enumerate(per_roi_pois[i]):
            poi_num = per_roi_poi_numbers[i][p_idx]
            poi_label = _format_special_poi_label(poi_num)
            global_r, global_c = (r_min + pr, c_min + pc)
            lr_start = max(0, pr - half)
            lr_end = min(roi_h, pr + half + 1)
            lc_start = max(0, pc - half)
            lc_end = min(roi_w, pc + half + 1)
            signal_poi = roi_stack[:, lr_start:lr_end, lc_start:lc_end].mean(
                axis=(1, 2)
            )
            res_poi = analyzer.analyze_signal(
                signal_poi, angles, patch_data=session.analysis.patch_data
            )
            vis.polar.save_polar_plot(
                save_path=session.figure_path(
                    f"{roi_names[i]}_POI_{poi_num}_fft", kind="vector"
                ),
                title=f"{roi_names[i]} {poi_label} - FFT",
                signal=signal_poi,
                angles=angles,
                analysis_result=res_poi,
                fit_key="fft",
                fit_label="FFT Fit",
                show_title=False,
                show_legend=False,
            )
            vis.polar.save_polar_plot(
                save_path=session.figure_path(
                    f"{roi_names[i]}_POI_{poi_num}_curve", kind="vector"
                ),
                title=f"{roi_names[i]} {poi_label} - Curve",
                signal=signal_poi,
                angles=angles,
                analysis_result=res_poi,
                fit_key="fit",
                fit_label="Curve Fit",
                show_title=False,
                show_legend=False,
            )
            _store_signal_result(
                cache_arrays,
                cache_poi_records,
                f"{roi_names[i]}_poi_{poi_num}",
                signal_poi,
                res_poi,
                title_fft=f"{roi_names[i]} {poi_label} - FFT",
                title_curve=f"{roi_names[i]} {poi_label} - Curve",
                stem_fft=f"{roi_names[i]}_POI_{poi_num}_fft",
                stem_curve=f"{roi_names[i]}_POI_{poi_num}_curve",
            )
            fft_p = res_poi.get("fft", {})
            fit_p = res_poi.get("fit", {})
            results_poi.append(
                {
                    "ROI": roi_names[i],
                    "POI": poi_label,
                    "Global R": global_r,
                    "Global C": global_c,
                    "FFT Orientation": fft_p.get("phase_deg", 0),
                    "FFT R2": fft_p.get("r_squared", 0),
                    "Curve Orientation": fit_p.get("angle_output_deg", 0),
                    "Curve R2": fit_p.get("r_squared", 0),
                }
            )
            poi_idx += 1
    _, full_h, full_w = stack.shape
    print(f"\n--- Sliding Window on whole image ({full_h}x{full_w}) ---")
    try:
        if curve_fit_cache_path is None:
            whole_start = time.monotonic()
            maps, stats, raw_window_data = sliding_analyzer.analyze(
                stack,
                angles,
                apply_threshold=True,
                progress_label="  [Whole Image] sliding window",
                roi_origin=(0, 0),
            )
            whole_sliding_elapsed = time.monotonic() - whole_start
            print(
                f"    Whole-image FFT+curve sliding time: {whole_sliding_elapsed:.1f}s"
            )
        else:
            fft_start = time.monotonic()
            maps, stats, raw_window_data = sliding_analyzer_fft_only.analyze(
                stack,
                angles,
                apply_threshold=True,
                progress_label="  [Whole Image] FFT-only sliding window",
                roi_origin=(0, 0),
            )
            fft_elapsed = time.monotonic() - fft_start
            maps.curve_angle, maps.curve_r2 = load_curve_fit_maps_from_csv(
                curve_fit_cache_path, maps.fft_angle.shape
            )
            curve_vals = extract_thresholded_map_values(
                maps.curve_angle,
                maps.curve_r2,
                session.analysis.r2_threshold,
                valid_mask=maps.a0_intensity_mask,
            )
            stats.curve_orientations = curve_vals
            stats.curve_passed = len(curve_vals)
            raw_window_data = []
            print(f"    Reused curve-fit cache from {curve_fit_cache_path}")
            print(f"    FFT-only sliding time: {fft_elapsed:.1f}s")
            cache_meta = _read_curve_cache_metadata(curve_fit_cache_path)
            if (
                cache_meta is not None
                and "whole_sliding_elapsed_seconds" in cache_meta
            ):
                estimated_full = float(
                    cache_meta["whole_sliding_elapsed_seconds"]
                )
                estimated_curve = max(estimated_full - fft_elapsed, 0.0)
                print(
                    f"    Estimated curve-fit time from prior uncached run: {estimated_curve:.1f}s"
                )
                print(
                    f"    Previous full FFT+curve sliding time: {estimated_full:.1f}s"
                )
        print(
            f"    Windows: {stats.total_windows} total, FFT passed: {stats.fft_passed}, Curve passed: {stats.curve_passed}"
        )
    except ValueError as e:
        logger.error(f"Sliding window on whole image failed: {e}")
        print(f"  [!] Sliding window failed: {e}")
        _save_roi_overlay_special_pair(
            session,
            vis,
            mean_image,
            sum_image,
            rois,
            roi_names,
            all_poi_info,
            scale_bar_params=whole_scale,
            roi_color="white",
            roi_linewidth=0.8,
            roi_label_fontsize=session.style.annotation_fontsize,
            roi_label_bbox=False,
        )
        exporter = ResultExporter(session.save_dir)
        exporter.export_poi_results(results_poi)
        return
    ws = session.analysis.window_size
    cache_arrays["whole_fft_angle"] = maps.fft_angle
    cache_arrays["whole_fft_r2"] = maps.fft_r2
    cache_arrays["whole_curve_angle"] = maps.curve_angle
    cache_arrays["whole_curve_r2"] = maps.curve_r2
    if maps.a0_intensity_map is not None:
        cache_arrays["whole_a0_intensity"] = maps.a0_intensity_map
    if maps.a0_intensity_mask is not None:
        cache_arrays["whole_a0_intensity_mask"] = (
            maps.a0_intensity_mask.astype(np.uint8)
        )
    ws_off = ws // 2
    roi_patches_map = [
        (
            max(0, r1 - ws_off),
            min(maps.fft_angle.shape[0], r2 - ws_off),
            max(0, c1 - ws_off),
            min(maps.fft_angle.shape[1], c2 - ws_off),
        )
        for r1, r2, c1, c2 in rois
    ]
    poi_info_map = [
        (max(0, gr - ws_off), max(0, gc - ws_off), pn, rn)
        for gr, gc, pn, rn in all_poi_info
    ]
    threshold = session.analysis.r2_threshold
    whole_fft_threshold_r2 = _threshold_r2_with_a0_mask(maps, maps.fft_r2)
    whole_curve_threshold_r2 = _threshold_r2_with_a0_mask(maps, maps.curve_r2)
    vis.overlay.save_orientation_map_with_overlay(
        save_path=session.figure_path(
            "whole_image_orientation_fft", kind="raster"
        ),
        title=f"Whole Image FFT Orientation (R²>{threshold})",
        angle_map=maps.fft_angle,
        r2_map=whole_fft_threshold_r2,
        threshold=threshold,
        roi_patches_local=roi_patches_map,
        roi_names=roi_names,
        poi_info_local=poi_info_map,
        scale_bar_params=whole_scale,
        vmin=_phi_vmin(session),
        vmax=_phi_vmax(session),
    )
    for extra_path in session.extra_figure_paths(
        "whole_image_orientation_fft", kind="raster"
    ):
        vis.overlay.save_orientation_map_with_overlay(
            save_path=extra_path,
            title=f"Whole Image FFT Orientation (R²>{threshold})",
            angle_map=maps.fft_angle,
            r2_map=whole_fft_threshold_r2,
            threshold=threshold,
            roi_patches_local=roi_patches_map,
            roi_names=roi_names,
            poi_info_local=poi_info_map,
            scale_bar_params=whole_scale,
            vmin=_phi_vmin(session),
            vmax=_phi_vmax(session),
        )
    vis.overlay.save_orientation_map_with_overlay(
        save_path=session.figure_path(
            "whole_image_orientation_curve", kind="raster"
        ),
        title=f"Whole Image Curve Orientation (R²>{threshold})",
        angle_map=maps.curve_angle,
        r2_map=whole_curve_threshold_r2,
        threshold=threshold,
        roi_patches_local=roi_patches_map,
        roi_names=roi_names,
        poi_info_local=poi_info_map,
        scale_bar_params=whole_scale,
        vmin=_phi_vmin(session),
        vmax=_phi_vmax(session),
    )
    for extra_path in session.extra_figure_paths(
        "whole_image_orientation_curve", kind="raster"
    ):
        vis.overlay.save_orientation_map_with_overlay(
            save_path=extra_path,
            title=f"Whole Image Curve Orientation (R²>{threshold})",
            angle_map=maps.curve_angle,
            r2_map=whole_curve_threshold_r2,
            threshold=threshold,
            roi_patches_local=roi_patches_map,
            roi_names=roi_names,
            poi_info_local=poi_info_map,
            scale_bar_params=whole_scale,
            vmin=_phi_vmin(session),
            vmax=_phi_vmax(session),
        )
    whole_diff_map, whole_diff_r2 = _orientation_difference_map(
        maps.fft_angle,
        maps.curve_angle,
        whole_fft_threshold_r2,
        whole_curve_threshold_r2,
    )
    _save_whole_image_diff_maps(
        session,
        vis,
        title=f"Whole Image |Curve-FFT| Orientation Diff (R²>{threshold})",
        angle_map=whole_diff_map,
        r2_map=whole_diff_r2,
        threshold=threshold,
        roi_patches_local=[],
        roi_names=[],
        poi_info_local=poi_info_map,
        scale_bar_params=whole_scale,
    )
    vis.overlay.save_orientation_map_with_overlay(
        save_path=session.figure_path("whole_image_r2_fft", kind="raster"),
        title="Whole Image FFT R² Map",
        angle_map=maps.fft_r2,
        r2_map=np.ones_like(maps.fft_r2),
        threshold=0.0,
        roi_patches_local=roi_patches_map,
        roi_names=roi_names,
        poi_info_local=poi_info_map,
        scale_bar_params=whole_scale,
        cbar_label="R²",
    )
    for extra_path in session.extra_figure_paths(
        "whole_image_r2_fft", kind="raster"
    ):
        vis.overlay.save_orientation_map_with_overlay(
            save_path=extra_path,
            title="Whole Image FFT R² Map",
            angle_map=maps.fft_r2,
            r2_map=np.ones_like(maps.fft_r2),
            threshold=0.0,
            roi_patches_local=roi_patches_map,
            roi_names=roi_names,
            poi_info_local=poi_info_map,
            scale_bar_params=whole_scale,
            cbar_label="R²",
        )
    vis.overlay.save_orientation_map_with_overlay(
        save_path=session.figure_path("whole_image_r2_curve", kind="raster"),
        title="Whole Image Curve R² Map",
        angle_map=maps.curve_r2,
        r2_map=np.ones_like(maps.curve_r2),
        threshold=0.0,
        roi_patches_local=roi_patches_map,
        roi_names=roi_names,
        poi_info_local=poi_info_map,
        scale_bar_params=whole_scale,
        cbar_label="R²",
    )
    for extra_path in session.extra_figure_paths(
        "whole_image_r2_curve", kind="raster"
    ):
        vis.overlay.save_orientation_map_with_overlay(
            save_path=extra_path,
            title="Whole Image Curve R² Map",
            angle_map=maps.curve_r2,
            r2_map=np.ones_like(maps.curve_r2),
            threshold=0.0,
            roi_patches_local=roi_patches_map,
            roi_names=roi_names,
            poi_info_local=poi_info_map,
            scale_bar_params=whole_scale,
            cbar_label="R²",
        )
    vis.histogram.save_histogram(
        save_path=session.figure_path("whole_image_histogram", kind="vector"),
        title_prefix=f"Whole Image (R²>{threshold})",
        fft_data=extract_thresholded_histogram_values(
            maps.fft_angle,
            whole_fft_threshold_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
        ),
        curve_data=extract_thresholded_histogram_values(
            maps.curve_angle,
            whole_curve_threshold_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
        ),
        count_total=maps.fft_angle.size,
        bin_fraction=session.analysis.bin_fraction_roimpois,
        show_statistics=False,
    )
    results_sliding = []
    cache_arithmetic: List[Dict[str, Any]] = []
    for i, (r_min, r_max, c_min, c_max) in enumerate(rois):
        print(f"\n--- Extracting {roi_names[i]} colormaps ---")
        mr1 = max(0, r_min - ws_off)
        mr2 = min(maps.fft_angle.shape[0], r_max - ws_off)
        mc1 = max(0, c_min - ws_off)
        mc2 = min(maps.fft_angle.shape[1], c_max - ws_off)
        if mr2 <= mr1 or mc2 <= mc1:
            logger.warning(
                f"{roi_names[i]}: ROI too small in map space, skipping"
            )
            continue
        roi_fft_ang = maps.fft_angle[mr1:mr2, mc1:mc2]
        roi_fft_r2 = maps.fft_r2[mr1:mr2, mc1:mc2]
        roi_curve_ang = maps.curve_angle[mr1:mr2, mc1:mc2]
        roi_curve_r2 = maps.curve_r2[mr1:mr2, mc1:mc2]
        roi_a0_mask = (
            maps.a0_intensity_mask[mr1:mr2, mc1:mc2]
            if maps.a0_intensity_mask is not None
            else None
        )
        roi_fft_threshold_r2 = _apply_valid_mask_to_r2(roi_fft_r2, roi_a0_mask)
        roi_curve_threshold_r2 = _apply_valid_mask_to_r2(
            roi_curve_r2, roi_a0_mask
        )
        roi_fft_display_ang = _temp_fix_roi_a_orientation_colormap(
            roi_fft_ang,
            roi_name=roi_names[i],
            enabled=session.analysis.temp_fix_twist_harmonization,
        )
        roi_curve_display_ang = _temp_fix_roi_a_orientation_colormap(
            roi_curve_ang,
            roi_name=roi_names[i],
            enabled=session.analysis.temp_fix_twist_harmonization,
        )
        roi_diff_ang, roi_diff_r2 = _orientation_difference_map(
            roi_fft_ang,
            roi_curve_ang,
            roi_fft_threshold_r2,
            roi_curve_threshold_r2,
        )
        poi_local = []
        for gr, gc, pn, rn in all_poi_info:
            if rn == roi_names[i]:
                lr = gr - ws_off - mr1
                lc = gc - ws_off - mc1
                if 0 <= lr < mr2 - mr1 and 0 <= lc < mc2 - mc1:
                    poi_local.append((lr, lc, pn, rn))
        vis.overlay.save_orientation_map_with_overlay(
            save_path=session.figure_path(
                f"{roi_names[i]}_orientation_fft", kind="raster"
            ),
            title=f"{roi_names[i]} FFT Orientation (R²>{threshold})",
            angle_map=roi_fft_display_ang,
            r2_map=roi_fft_threshold_r2,
            threshold=threshold,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            vmin=_phi_vmin(session),
            vmax=_phi_vmax(session),
        )
        for extra_path in session.extra_figure_paths(
            f"{roi_names[i]}_orientation_fft", kind="raster"
        ):
            vis.overlay.save_orientation_map_with_overlay(
                save_path=extra_path,
                title=f"{roi_names[i]} FFT Orientation (R²>{threshold})",
                angle_map=roi_fft_display_ang,
                r2_map=roi_fft_threshold_r2,
                threshold=threshold,
                scale_bar_params=roi_scale,
                show_poi_overlay=False,
                vmin=_phi_vmin(session),
                vmax=_phi_vmax(session),
            )
        vis.overlay.save_orientation_map_with_overlay(
            save_path=session.figure_path(
                f"{roi_names[i]}_orientation_curve", kind="raster"
            ),
            title=f"{roi_names[i]} Curve Orientation (R²>{threshold})",
            angle_map=roi_curve_display_ang,
            r2_map=roi_curve_threshold_r2,
            threshold=threshold,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            vmin=_phi_vmin(session),
            vmax=_phi_vmax(session),
        )
        for extra_path in session.extra_figure_paths(
            f"{roi_names[i]}_orientation_curve", kind="raster"
        ):
            vis.overlay.save_orientation_map_with_overlay(
                save_path=extra_path,
                title=f"{roi_names[i]} Curve Orientation (R²>{threshold})",
                angle_map=roi_curve_display_ang,
                r2_map=roi_curve_threshold_r2,
                threshold=threshold,
                scale_bar_params=roi_scale,
                show_poi_overlay=False,
                vmin=_phi_vmin(session),
                vmax=_phi_vmax(session),
            )
        vis.overlay.save_orientation_map_with_overlay(
            save_path=session.figure_path(
                f"{roi_names[i]}_orientation_diff", kind="raster"
            ),
            title=f"{roi_names[i]} |Curve-FFT| Orientation Diff (RΒ²>{threshold})",
            angle_map=roi_diff_ang,
            r2_map=roi_diff_r2,
            threshold=threshold,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            cbar_label="|Δθ| (Degrees)",
        )
        for extra_path in session.extra_figure_paths(
            f"{roi_names[i]}_orientation_diff", kind="raster"
        ):
            vis.overlay.save_orientation_map_with_overlay(
                save_path=extra_path,
                title=f"{roi_names[i]} |Curve-FFT| Orientation Diff (RΒ²>{threshold})",
                angle_map=roi_diff_ang,
                r2_map=roi_diff_r2,
                threshold=threshold,
                scale_bar_params=roi_scale,
                show_poi_overlay=False,
                cbar_label="|Δθ| (Degrees)",
            )
        vis.overlay.save_orientation_map_with_overlay(
            save_path=session.figure_path(
                f"{roi_names[i]}_r2_fft", kind="raster"
            ),
            title=f"{roi_names[i]} FFT R² Map",
            angle_map=roi_fft_r2,
            r2_map=np.ones_like(roi_fft_r2),
            threshold=0.0,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            cbar_label="R²",
        )
        for extra_path in session.extra_figure_paths(
            f"{roi_names[i]}_r2_fft", kind="raster"
        ):
            vis.overlay.save_orientation_map_with_overlay(
                save_path=extra_path,
                title=f"{roi_names[i]} FFT R² Map",
                angle_map=roi_fft_r2,
                r2_map=np.ones_like(roi_fft_r2),
                threshold=0.0,
                scale_bar_params=roi_scale,
                show_poi_overlay=False,
                cbar_label="R²",
            )
        vis.overlay.save_orientation_map_with_overlay(
            save_path=session.figure_path(
                f"{roi_names[i]}_r2_curve", kind="raster"
            ),
            title=f"{roi_names[i]} Curve R² Map",
            angle_map=roi_curve_r2,
            r2_map=np.ones_like(roi_curve_r2),
            threshold=0.0,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            cbar_label="R²",
        )
        for extra_path in session.extra_figure_paths(
            f"{roi_names[i]}_r2_curve", kind="raster"
        ):
            vis.overlay.save_orientation_map_with_overlay(
                save_path=extra_path,
                title=f"{roi_names[i]} Curve R² Map",
                angle_map=roi_curve_r2,
                r2_map=np.ones_like(roi_curve_r2),
                threshold=0.0,
                scale_bar_params=roi_scale,
                show_poi_overlay=False,
                cbar_label="R²",
            )
        roi_mean = mean_image[r_min:r_max, c_min:c_max]
        roi_sum = (
            None if sum_image is None else sum_image[r_min:r_max, c_min:c_max]
        )
        _save_zoomed_roi_pair(
            session,
            vis,
            f"{roi_names[i]}_zoomed",
            roi_mean,
            roi_sum,
            per_roi_pois[i],
        )
        fft_vals = extract_thresholded_histogram_values(
            roi_fft_ang,
            roi_fft_threshold_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
            max_phi_exclusive=57.0,
        )
        curve_vals = extract_thresholded_histogram_values(
            roi_curve_ang,
            roi_curve_threshold_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
            max_phi_exclusive=57.0,
        )
        vis.histogram.save_single_histogram(
            save_path=session.figure_path(
                f"{roi_names[i]}_histogram_fft", kind="vector"
            ),
            title=f"{roi_names[i]} (R²>{threshold}) - FFT",
            data=fft_vals,
            count_total=roi_fft_ang.size,
            color=session.style.histogram_fft_color,
            bin_fraction=session.analysis.bin_fraction_roimpois,
            statistics_mode="mean_std",
            **_special_roi_histogram_bin_settings(session.analysis),
        )
        vis.histogram.save_single_histogram(
            save_path=session.figure_path(
                f"{roi_names[i]}_histogram_curve", kind="vector"
            ),
            title=f"{roi_names[i]} (R²>{threshold}) - Curve",
            data=curve_vals,
            count_total=roi_fft_ang.size,
            color=session.style.histogram_curve_color,
            bin_fraction=session.analysis.bin_fraction_roimpois,
            statistics_mode="mean_std",
            **_special_roi_histogram_bin_settings(session.analysis),
        )
        if per_roi_pois[i]:
            for p_idx, (poi_r, poi_c) in enumerate(per_roi_pois[i]):
                map_r, map_c = _roi_poi_to_map_center(
                    poi_r,
                    poi_c,
                    r_min=r_min,
                    c_min=c_min,
                    map_r0=mr1,
                    map_c0=mc1,
                    map_shape=roi_fft_ang.shape,
                    ws_off=ws_off,
                )
                stats_fft_local = _local_square_stats(
                    roi_fft_ang,
                    roi_fft_threshold_r2,
                    center_r=map_r,
                    center_c=map_c,
                    threshold=threshold,
                    region_size=LOCAL_POI_REGION_SIZE,
                )
                stats_curve_local = _local_square_stats(
                    roi_curve_ang,
                    roi_curve_threshold_r2,
                    center_r=map_r,
                    center_c=map_c,
                    threshold=threshold,
                    region_size=LOCAL_POI_REGION_SIZE,
                )
                fft_local_data = _filter_histogram_outliers_by_peak_fraction(
                    _filter_histogram_phi_values(
                        stats_fft_local.data if stats_fft_local else [],
                        ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
                        max_phi_exclusive=57.0,
                    ),
                    count_total=max(
                        len(stats_fft_local.data) if stats_fft_local else 0, 1
                    ),
                    bin_fraction=session.analysis.bin_fraction_local_roimpois,
                    peak_fraction=LOCAL_HISTOGRAM_OUTLIER_PEAK_FRACTION,
                )
                curve_local_data = _filter_histogram_outliers_by_peak_fraction(
                    _filter_histogram_phi_values(
                        stats_curve_local.data if stats_curve_local else [],
                        ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
                        max_phi_exclusive=57.0,
                    ),
                    count_total=max(
                        len(stats_curve_local.data)
                        if stats_curve_local
                        else 0,
                        1,
                    ),
                    bin_fraction=session.analysis.bin_fraction_local_roimpois,
                    peak_fraction=LOCAL_HISTOGRAM_OUTLIER_PEAK_FRACTION,
                )
                vis.histogram.save_single_histogram(
                    session.figure_path(
                        f"{roi_names[i]}_POI_{per_roi_poi_numbers[i][p_idx]}_local_histogram_fft",
                        kind="vector",
                    ),
                    f"{roi_names[i]} {_format_special_poi_label(per_roi_poi_numbers[i][p_idx])} Local - FFT",
                    fft_local_data,
                    max(len(fft_local_data), 1),
                    color=session.style.histogram_fft_color,
                    bin_fraction=session.analysis.bin_fraction_local_roimpois,
                    show_statistics=False,
                    font_scale=LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
                    **_special_poi_histogram_bin_settings(session.analysis),
                )
                vis.histogram.save_single_histogram(
                    session.figure_path(
                        f"{roi_names[i]}_POI_{per_roi_poi_numbers[i][p_idx]}_local_histogram_curve",
                        kind="vector",
                    ),
                    f"{roi_names[i]} {_format_special_poi_label(per_roi_poi_numbers[i][p_idx])} Local - Curve",
                    curve_local_data,
                    max(len(curve_local_data), 1),
                    color=session.style.histogram_curve_color,
                    bin_fraction=session.analysis.bin_fraction_local_roimpois,
                    show_statistics=False,
                    font_scale=LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
                    **_special_poi_histogram_bin_settings(session.analysis),
                )
        if do_twist_angle and per_roi_pairs[i]:
            mask_assignment, poi_masks, mask_thresholds = (
                arithmetic_processor.build_intensity_masks(
                    roi_mean, per_roi_pois[i]
                )
            )
            cache_arrays[f"{roi_names[i]}_twist_mask_assignment"] = (
                mask_assignment
            )
            for p_idx, mask in enumerate(poi_masks, start=1):
                cache_arrays[f"{roi_names[i]}_twist_mask_poi_{p_idx}"] = (
                    mask.astype(np.uint8)
                )
            poi_stats_fft = [
                arithmetic_processor.get_masked_stats(
                    roi_fft_ang, roi_fft_threshold_r2, poi_masks[p_idx]
                )
                for p_idx in range(len(per_roi_pois[i]))
            ]
            poi_stats_curve = [
                arithmetic_processor.get_masked_stats(
                    roi_curve_ang, roi_curve_threshold_r2, poi_masks[p_idx]
                )
                for p_idx in range(len(per_roi_pois[i]))
            ]
            for p_idx, stats_fft in enumerate(poi_stats_fft):
                stats_curve = poi_stats_curve[p_idx]
                fft_local_data = _filter_histogram_phi_values(
                    stats_fft.data if stats_fft else [],
                    ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
                    max_phi_exclusive=57.0,
                )
                curve_local_data = _filter_histogram_phi_values(
                    stats_curve.data if stats_curve else [],
                    ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
                    max_phi_exclusive=57.0,
                )
                if fft_local_data or curve_local_data:
                    vis.histogram.save_single_histogram(
                        session.figure_path(
                            f"{roi_names[i]}_POI_{per_roi_poi_numbers[i][p_idx]}_local_histogram_fft",
                            kind="vector",
                        ),
                        f"{roi_names[i]} {_format_special_poi_label(per_roi_poi_numbers[i][p_idx])} Local - FFT",
                        fft_local_data,
                        max(len(fft_local_data), 1),
                        color=session.style.histogram_fft_color,
                        bin_fraction=session.analysis.bin_fraction_roimpois,
                        show_statistics=False,
                        font_scale=LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
                        **_special_poi_histogram_bin_settings(
                            session.analysis
                        ),
                    )
                    vis.histogram.save_single_histogram(
                        session.figure_path(
                            f"{roi_names[i]}_POI_{per_roi_poi_numbers[i][p_idx]}_local_histogram_curve",
                            kind="vector",
                        ),
                        f"{roi_names[i]} {_format_special_poi_label(per_roi_poi_numbers[i][p_idx])} Local - Curve",
                        curve_local_data,
                        max(len(curve_local_data), 1),
                        color=session.style.histogram_curve_color,
                        bin_fraction=session.analysis.bin_fraction_roimpois,
                        show_statistics=False,
                        font_scale=LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
                        **_special_poi_histogram_bin_settings(
                            session.analysis
                        ),
                    )
            for left_idx, right_idx in per_roi_pairs[i]:
                poi_left = per_roi_poi_numbers[i][left_idx]
                poi_right = per_roi_poi_numbers[i][right_idx]
                if poi_stats_fft[left_idx] and poi_stats_fft[right_idx]:
                    r_m, v_p, s_z = arithmetic_processor.compute_arithmetic(
                        roi_fft_ang,
                        roi_fft_threshold_r2,
                        poi_stats_fft[left_idx],
                        poi_stats_fft[right_idx],
                        selection_mask=poi_masks[left_idx],
                    )
                    r_m, v_p = _maybe_harmonize_roi_a_twist_values(
                        r_m,
                        v_p,
                        roi_name=roi_names[i],
                        enabled=session.analysis.temp_fix_twist_harmonization,
                    )
                    stem = f"twist_angle_δ_colormap_poi{poi_left}_poi{poi_right}_FFT_σ{s_z:.2f}"
                    vis.arithmetic.save_arithmetic_map(
                        session.figure_path(stem, kind="raster"),
                        f"Twist angle δ mapping {roi_names[i]}",
                        r_m,
                        s_z,
                    )
                    for extra_path in session.extra_figure_paths(
                        stem, kind="raster"
                    ):
                        vis.arithmetic.save_arithmetic_map(
                            extra_path,
                            f"Twist angle δ mapping {roi_names[i]}",
                            r_m,
                            s_z,
                        )
                    if v_p:
                        hist_stem = f"twist_angle_δ_histogram_poi{poi_left}_poi{poi_right}_FFT_σ{s_z:.2f}"
                        vis.arithmetic.save_arithmetic_histogram(
                            session.figure_path(hist_stem, kind="vector"),
                            f"Twist angle δ histogram {roi_names[i]}",
                            v_p,
                            s_z,
                        )
                    cache_arrays[f"{stem}_map"] = r_m
                    cache_arithmetic.append(
                        {
                            "roi": roi_names[i],
                            "method": "fft",
                            "map_key": f"{stem}_map",
                            "map_stem": stem,
                            "hist_stem": f"twist_angle_δ_histogram_poi{poi_left}_poi{poi_right}_FFT_σ{s_z:.2f}",
                            "title": f"Twist angle δ mapping {roi_names[i]}",
                            "hist_title": f"Twist angle δ histogram {roi_names[i]}",
                            "sigma_z": float(s_z),
                            "mask_thresholds": list(mask_thresholds),
                        }
                    )
                if poi_stats_curve[left_idx] and poi_stats_curve[right_idx]:
                    r_m, v_p, s_z = arithmetic_processor.compute_arithmetic(
                        roi_curve_ang,
                        roi_curve_threshold_r2,
                        poi_stats_curve[left_idx],
                        poi_stats_curve[right_idx],
                        selection_mask=poi_masks[left_idx],
                    )
                    r_m, v_p = _maybe_harmonize_roi_a_twist_values(
                        r_m,
                        v_p,
                        roi_name=roi_names[i],
                        enabled=session.analysis.temp_fix_twist_harmonization,
                    )
                    stem = f"twist_angle_δ_colormap_poi{poi_left}_poi{poi_right}_curve_σ{s_z:.2f}"
                    vis.arithmetic.save_arithmetic_map(
                        session.figure_path(stem, kind="raster"),
                        f"Twist angle δ mapping {roi_names[i]}",
                        r_m,
                        s_z,
                    )
                    for extra_path in session.extra_figure_paths(
                        stem, kind="raster"
                    ):
                        vis.arithmetic.save_arithmetic_map(
                            extra_path,
                            f"Twist angle δ mapping {roi_names[i]}",
                            r_m,
                            s_z,
                        )
                    if v_p:
                        hist_stem = f"twist_angle_δ_histogram_poi{poi_left}_poi{poi_right}_curve_σ{s_z:.2f}"
                        vis.arithmetic.save_arithmetic_histogram(
                            session.figure_path(hist_stem, kind="vector"),
                            f"Twist angle δ histogram {roi_names[i]}",
                            v_p,
                            s_z,
                        )
                    cache_arrays[f"{stem}_map"] = r_m
                    cache_arithmetic.append(
                        {
                            "roi": roi_names[i],
                            "method": "curve",
                            "map_key": f"{stem}_map",
                            "map_stem": stem,
                            "hist_stem": f"twist_angle_δ_histogram_poi{poi_left}_poi{poi_right}_curve_σ{s_z:.2f}",
                            "title": f"Twist angle δ mapping {roi_names[i]}",
                            "hist_title": f"Twist angle δ histogram {roi_names[i]}",
                            "sigma_z": float(s_z),
                            "mask_thresholds": list(mask_thresholds),
                        }
                    )
        results_sliding.append(
            {
                "ROI": roi_names[i],
                "Map Size": f"{mr2 - mr1}x{mc2 - mc1}",
                "FFT Passed": len(fft_vals),
                "Curve Passed": len(curve_vals),
            }
        )
    _save_roi_overlay_special_pair(
        session,
        vis,
        mean_image,
        sum_image,
        rois,
        roi_names,
        all_poi_info,
        scale_bar_params=whole_scale,
        roi_color="white",
        roi_linewidth=0.8,
        roi_label_fontsize=session.style.annotation_fontsize,
        roi_label_bbox=False,
    )
    exporter = ResultExporter(session.save_dir)
    exporter.export_poi_results(results_poi)
    exporter.export_sliding_stats(results_sliding)
    if raw_window_data:
        df = pd.DataFrame(raw_window_data)
        df.to_csv(
            session.table_dir("window_data") / "whole_image_window_data.csv",
            index=False,
        )
        curve_path = ResultExporter(
            session.save_dir
        ).export_curve_fit_window_data(
            raw_window_data,
            roi_id=None,
            filename="whole_image_curve_fit_window_data.csv",
        )
        cache_curve_tables.append(
            {"scope": "whole_image", "path": str(curve_path)}
        )
        if curve_fit_cache_path is None:
            _write_curve_cache_metadata(
                curve_path,
                {
                    "whole_sliding_elapsed_seconds": float(
                        whole_sliding_elapsed
                    ),
                    "window_count": int(stats.total_windows),
                    "curve_fit_cache_source": "roimpois_special",
                    **_curve_cache_identity(
                        session,
                        crop_bounds=crop_bounds,
                        cropped_shape=stack.shape,
                        stack=stack,
                        angles=angles,
                    ),
                },
            )
    _save_cache_bundle(
        session,
        metadata={
            "pipeline_mode": "roimpois_special",
            "rois": [list(r) for r in rois],
            "roi_names": roi_names,
            "all_poi_info": [list(p) for p in all_poi_info],
            "per_roi_pois": [
                [[int(r), int(c)] for r, c in pts] for pts in per_roi_pois
            ],
            "per_roi_poi_numbers": [
                list(nums) for nums in per_roi_poi_numbers
            ],
            "crop_bounds": list(crop_bounds) if crop_bounds else None,
            "whole_scale": whole_scale,
            "roi_scale": roi_scale,
            "poi_polar_records": cache_poi_records,
            "curve_fit_window_tables": cache_curve_tables,
            "arithmetic_maps": cache_arithmetic,
            "per_roi_pairs": [
                [list(pair) for pair in pairs] for pairs in per_roi_pairs
            ],
        },
        arrays=cache_arrays,
    )
    print(f"\n✓ Analysis complete. Results saved to {session.save_dir}")
