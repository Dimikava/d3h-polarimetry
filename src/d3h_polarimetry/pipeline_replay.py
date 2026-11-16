"""Replay saved polarization analysis cache bundles."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
from .pipeline_config import SessionConfig
from .visualization import VisualizationManager
from .cache import AnalysisCache
from .pipeline_helpers import (
    LOCAL_HISTOGRAM_OUTLIER_PEAK_FRACTION,
    LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
    LOCAL_POI_REGION_SIZE,
    _filter_histogram_outliers_by_peak_fraction,
    _filter_histogram_phi_values,
    _local_histogram_bin_count,
    _local_histogram_display_ticks,
    _local_square_stats,
    _orientation_difference_map,
    _phi_vmin,
    _phi_vmax,
    _rebuild_analysis_result,
    _roi_poi_to_map_center,
    _save_roi_overlay_special_pair,
    _save_whole_image_diff_maps,
    _save_zoomed_roi_pair,
    _special_roi_histogram_bin_settings,
    _temp_fix_roi_a_orientation_colormap,
    extract_thresholded_histogram_values,
)

logger = logging.getLogger(__name__)


def replay_saved_analysis(
    session: SessionConfig, cache_path: Union[str, Path]
) -> None:
    """Rebuild plots from a saved cache bundle without rerunning analysis."""
    metadata, arrays = AnalysisCache.load(Path(cache_path))
    analysis_meta = metadata.get("analysis", {})
    for key, value in analysis_meta.items():
        if hasattr(session.analysis, key):
            setattr(session.analysis, key, value)
    mode = metadata.get("pipeline_mode")
    vis = VisualizationManager(session.style)
    mean_image = arrays["mean_image"]
    sum_image = arrays.get("sum_image")
    angles = arrays["angles"]
    if mode != "roimpois_special":
        raise ValueError(f"Unsupported cached pipeline mode: {mode}")
    _replay_roimpois_special(
        session, vis, metadata, arrays, mean_image, angles, sum_image=sum_image
    )
    for item in metadata.get("arithmetic_maps", []):
        result_map = arrays[item["map_key"]]
        vis.arithmetic.save_arithmetic_map(
            session.figure_path(item["map_stem"], kind="raster"),
            item["title"],
            result_map,
            item["sigma_z"],
        )
        for extra_path in session.extra_figure_paths(
            item["map_stem"], kind="raster"
        ):
            vis.arithmetic.save_arithmetic_map(
                extra_path, item["title"], result_map, item["sigma_z"]
            )
        values = [float(v) for v in result_map[np.isfinite(result_map)]]
        if values:
            vis.arithmetic.save_arithmetic_histogram(
                session.figure_path(item["hist_stem"], kind="vector"),
                item["hist_title"],
                values,
                item["sigma_z"],
            )
    print(f"Replay complete. Figures regenerated in {session.save_dir}")


def _replay_signal_records(
    session: SessionConfig,
    vis: VisualizationManager,
    arrays: Dict[str, np.ndarray],
    angles: np.ndarray,
    records: Sequence[Dict[str, Any]],
) -> None:
    for record in records:
        signal = arrays[record["signal_key"]]
        analysis_result = _rebuild_analysis_result(arrays, record)
        if record.get("stem_fft") and "fft" in analysis_result:
            vis.polar.save_polar_plot(
                save_path=session.figure_path(
                    record["stem_fft"], kind="vector"
                ),
                title=record.get("title_fft") or record["stem_fft"],
                signal=signal,
                angles=angles,
                analysis_result=analysis_result,
                fit_key="fft",
                fit_label="FFT Fit",
            )
        if record.get("stem_curve") and "fit" in analysis_result:
            vis.polar.save_polar_plot(
                save_path=session.figure_path(
                    record["stem_curve"], kind="vector"
                ),
                title=record.get("title_curve") or record["stem_curve"],
                signal=signal,
                angles=angles,
                analysis_result=analysis_result,
                fit_key="fit",
                fit_label="Curve Fit",
            )


def _replay_roimpois_special(
    session: SessionConfig,
    vis: VisualizationManager,
    metadata: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    mean_image: np.ndarray,
    angles: np.ndarray,
    sum_image: Optional[np.ndarray] = None,
) -> None:
    _replay_signal_records(
        session, vis, arrays, angles, metadata.get("poi_polar_records", [])
    )
    rois = [tuple(r) for r in metadata.get("rois", [])]
    roi_names = list(metadata.get("roi_names", []))
    all_poi_info = [tuple(p) for p in metadata.get("all_poi_info", [])]
    per_roi_pois = [
        [tuple(p) for p in row] for row in metadata.get("per_roi_pois", [])
    ]
    whole_scale = metadata.get("whole_scale")
    roi_scale = metadata.get("roi_scale")
    threshold = session.analysis.r2_threshold
    maps_fft_angle = arrays["whole_fft_angle"]
    maps_fft_r2 = arrays["whole_fft_r2"]
    maps_curve_angle = arrays["whole_curve_angle"]
    maps_curve_r2 = arrays["whole_curve_r2"]
    ws_off = session.analysis.window_size // 2
    roi_patches_map = [
        (
            max(0, r1 - ws_off),
            min(maps_fft_angle.shape[0], r2 - ws_off),
            max(0, c1 - ws_off),
            min(maps_fft_angle.shape[1], c2 - ws_off),
        )
        for r1, r2, c1, c2 in rois
    ]
    poi_info_map = [
        (max(0, gr - ws_off), max(0, gc - ws_off), pn, rn)
        for gr, gc, pn, rn in all_poi_info
    ]
    vis.overlay.save_orientation_map_with_overlay(
        save_path=session.figure_path(
            "whole_image_orientation_fft", kind="raster"
        ),
        title=f"Whole Image FFT Orientation (R²>{threshold})",
        angle_map=maps_fft_angle,
        r2_map=maps_fft_r2,
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
        angle_map=maps_curve_angle,
        r2_map=maps_curve_r2,
        threshold=threshold,
        roi_patches_local=roi_patches_map,
        roi_names=roi_names,
        poi_info_local=poi_info_map,
        scale_bar_params=whole_scale,
        vmin=_phi_vmin(session),
        vmax=_phi_vmax(session),
    )
    whole_diff_map, whole_diff_r2 = _orientation_difference_map(
        maps_fft_angle, maps_curve_angle, maps_fft_r2, maps_curve_r2
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
        angle_map=maps_fft_r2,
        r2_map=np.ones_like(maps_fft_r2),
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
        angle_map=maps_curve_r2,
        r2_map=np.ones_like(maps_curve_r2),
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
            maps_fft_angle,
            maps_fft_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
        ),
        curve_data=extract_thresholded_histogram_values(
            maps_curve_angle,
            maps_curve_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
        ),
        count_total=maps_fft_angle.size,
        bin_fraction=session.analysis.bin_fraction_roimpois,
        show_statistics=False,
    )
    for i, roi_name in enumerate(roi_names):
        r_min, r_max, c_min, c_max = rois[i]
        mr1 = max(0, r_min - ws_off)
        mr2 = min(maps_fft_angle.shape[0], r_max - ws_off)
        mc1 = max(0, c_min - ws_off)
        mc2 = min(maps_fft_angle.shape[1], c_max - ws_off)
        roi_fft_ang = maps_fft_angle[mr1:mr2, mc1:mc2]
        roi_fft_r2 = maps_fft_r2[mr1:mr2, mc1:mc2]
        roi_curve_ang = maps_curve_angle[mr1:mr2, mc1:mc2]
        roi_curve_r2 = maps_curve_r2[mr1:mr2, mc1:mc2]
        roi_fft_display_ang = _temp_fix_roi_a_orientation_colormap(
            roi_fft_ang,
            roi_name=roi_name,
            enabled=session.analysis.temp_fix_twist_harmonization,
        )
        roi_curve_display_ang = _temp_fix_roi_a_orientation_colormap(
            roi_curve_ang,
            roi_name=roi_name,
            enabled=session.analysis.temp_fix_twist_harmonization,
        )
        roi_diff_ang, roi_diff_r2 = _orientation_difference_map(
            roi_fft_ang, roi_curve_ang, roi_fft_r2, roi_curve_r2
        )
        roi_mean = mean_image[r_min:r_max, c_min:c_max]
        roi_sum = (
            None if sum_image is None else sum_image[r_min:r_max, c_min:c_max]
        )
        vis.overlay.save_orientation_map_with_overlay(
            session.figure_path(f"{roi_name}_orientation_diff", kind="raster"),
            f"{roi_name} |Curve-FFT| Orientation Diff (RΒ²>{threshold})",
            roi_diff_ang,
            roi_diff_r2,
            threshold,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            cbar_label="|Δθ| (Degrees)",
        )
        vis.overlay.save_orientation_map_with_overlay(
            session.figure_path(f"{roi_name}_orientation_fft", kind="raster"),
            f"{roi_name} FFT Orientation (R²>{threshold})",
            roi_fft_display_ang,
            roi_fft_r2,
            threshold,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            vmin=_phi_vmin(session),
            vmax=_phi_vmax(session),
        )
        vis.overlay.save_orientation_map_with_overlay(
            session.figure_path(
                f"{roi_name}_orientation_curve", kind="raster"
            ),
            f"{roi_name} Curve Orientation (R²>{threshold})",
            roi_curve_display_ang,
            roi_curve_r2,
            threshold,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            vmin=_phi_vmin(session),
            vmax=_phi_vmax(session),
        )
        vis.overlay.save_orientation_map_with_overlay(
            session.figure_path(f"{roi_name}_r2_fft", kind="raster"),
            f"{roi_name} FFT R² Map",
            roi_fft_r2,
            np.ones_like(roi_fft_r2),
            0.0,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            cbar_label="R²",
        )
        vis.overlay.save_orientation_map_with_overlay(
            session.figure_path(f"{roi_name}_r2_curve", kind="raster"),
            f"{roi_name} Curve R² Map",
            roi_curve_r2,
            np.ones_like(roi_curve_r2),
            0.0,
            scale_bar_params=roi_scale,
            show_poi_overlay=False,
            cbar_label="R²",
        )
        _save_zoomed_roi_pair(
            session,
            vis,
            f"{roi_name}_zoomed",
            roi_mean,
            roi_sum,
            per_roi_pois[i],
        )
        fft_vals = extract_thresholded_histogram_values(
            roi_fft_ang,
            roi_fft_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
            max_phi_exclusive=57.0,
        )
        curve_vals = extract_thresholded_histogram_values(
            roi_curve_ang,
            roi_curve_r2,
            threshold,
            ignore_phi_ge_58=session.analysis.temp_ignore_phi_ge_58,
            max_phi_exclusive=57.0,
        )
        vis.histogram.save_single_histogram(
            save_path=session.figure_path(
                f"{roi_name}_histogram_fft", kind="vector"
            ),
            title=f"{roi_name} (R²>{threshold}) - FFT",
            data=fft_vals,
            count_total=roi_fft_ang.size,
            color=session.style.histogram_fft_color,
            bin_fraction=session.analysis.bin_fraction_roimpois,
            statistics_mode="mean_std",
            **_special_roi_histogram_bin_settings(session.analysis),
        )
        vis.histogram.save_single_histogram(
            save_path=session.figure_path(
                f"{roi_name}_histogram_curve", kind="vector"
            ),
            title=f"{roi_name} (R²>{threshold}) - Curve",
            data=curve_vals,
            count_total=roi_fft_ang.size,
            color=session.style.histogram_curve_color,
            bin_fraction=session.analysis.bin_fraction_roimpois,
            statistics_mode="mean_std",
            **_special_roi_histogram_bin_settings(session.analysis),
        )
        if per_roi_pois[i]:
            poi_numbers = metadata.get("per_roi_poi_numbers", [])[i]
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
                    roi_fft_r2,
                    center_r=map_r,
                    center_c=map_c,
                    threshold=threshold,
                    region_size=LOCAL_POI_REGION_SIZE,
                )
                stats_curve_local = _local_square_stats(
                    roi_curve_ang,
                    roi_curve_r2,
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
                        f"{roi_name}_POI_{poi_numbers[p_idx]}_local_histogram_fft",
                        kind="vector",
                    ),
                    f"{roi_name} POI {poi_numbers[p_idx]} Local - FFT",
                    fft_local_data,
                    max(len(fft_local_data), 1),
                    color=session.style.histogram_fft_color,
                    bin_fraction=session.analysis.bin_fraction_local_roimpois,
                    n_bins_override=_local_histogram_bin_count(
                        fft_local_data,
                        base_tick_count=session.analysis.local_hist_tick_count,
                    ),
                    xticks_override=_local_histogram_display_ticks(
                        fft_local_data
                    ),
                    show_statistics=False,
                    font_scale=LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
                )
                vis.histogram.save_single_histogram(
                    session.figure_path(
                        f"{roi_name}_POI_{poi_numbers[p_idx]}_local_histogram_curve",
                        kind="vector",
                    ),
                    f"{roi_name} POI {poi_numbers[p_idx]} Local - Curve",
                    curve_local_data,
                    max(len(curve_local_data), 1),
                    color=session.style.histogram_curve_color,
                    bin_fraction=session.analysis.bin_fraction_local_roimpois,
                    n_bins_override=_local_histogram_bin_count(
                        curve_local_data,
                        base_tick_count=session.analysis.local_hist_tick_count,
                    ),
                    xticks_override=_local_histogram_display_ticks(
                        curve_local_data
                    ),
                    show_statistics=False,
                    font_scale=LOCAL_POI_HISTOGRAM_FONT_COMPENSATION,
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
