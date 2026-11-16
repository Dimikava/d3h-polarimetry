"""D3h special ROI/POI CLI with whole-image sliding-window analysis."""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from d3h_polarimetry.reader import DatParser
from d3h_polarimetry.io import MicroscopyReaderError, read_tiff_source_traced
from d3h_polarimetry.axis_data import (
    AmbiguousAxesError,
    AxisMapping,
    AxisPresetStore,
    AxisResolutionError,
    MicroscopyDataset,
    POLARIZATION_CUBE_DIMS,
    RawMicroscopyData,
    canonicalize,
    complete_mapping,
    describe_array,
    parse_axis_selections,
)
from d3h_polarimetry.models import FFTMode, D3hFixedModel, D3hRotatingModel
from d3h_polarimetry.gui import ROISelector, CropSelector
from d3h_polarimetry.pipeline import (
    SessionConfig,
    configure_a0_intensity_mask,
    replay_saved_analysis,
    run_roimpois_special_pipeline,
)
from d3h_polarimetry.cache import AnalysisCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
ROIMPOIS_SPECIAL_PRESENTATION_SCALE = 0.28472222
ROIMPOIS_SPECIAL_POLAR_FONT_COMPENSATION = (
    1.0 / ROIMPOIS_SPECIAL_PRESENTATION_SCALE
)
from d3h_polarimetry.cli.polarization_args import build_parser
from d3h_polarimetry.cli.polarization_config import (
    ROIMPOIS_SPECIAL_PRESENTATION_SCALE,
    _build_analysis_config,
    _build_figure_style,
)
from d3h_polarimetry.cli.polarization_paths import (
    _compute_projection_images,
    _save_projection_reference_images,
    _show_a0_mask_preview,
    resolve_auto_curve_cache,
    resolve_auto_replay_cache,
    resolve_replay_results_dir,
    resolve_reuse_curve_cache_crop_policy,
    validate_paths,
)
from d3h_polarimetry.cli.polarization_presets import (
    _default_preset_file,
    resolve_polarization_presets,
)


def _resolve_polarization_axes(
    stack: np.ndarray,
    filenames: Sequence[str],
    image_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, MicroscopyDataset, dict[str, object]]:
    """Resolve one folder stack into a validated angle/y/x analysis cube."""
    if stack.ndim != 3:
        raise AxisResolutionError(
            f"The legacy polarization CLI currently expects a folder stack with three stored dimensions, got {stack.shape}. Use load_dataset() to inspect general ND TIFF files."
        )
    descriptor = describe_array(
        stack,
        source_dims=("sequence", "y", "x"),
        source=image_dir,
        source_kind="tiff-directory",
        suggestions={"y": "y", "x": "x"},
        coordinates={"sequence": tuple(filenames)},
        evidence={"y": "2D TIFF image row", "x": "2D TIFF image column"},
    )
    raw = RawMicroscopyData(np.asarray(stack), descriptor, tuple(filenames))
    return _resolve_polarization_raw(raw, args)


def _resolve_polarization_raw(
    raw: RawMicroscopyData, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, MicroscopyDataset, dict[str, object]]:
    """Resolve decoded TIFF data into a validated angle/y/x analysis cube."""
    descriptor = raw.descriptor
    mapping_path = getattr(args, "axis_mapping", None)
    preset_name = getattr(args, "axis_preset", None)
    if mapping_path and preset_name:
        raise AxisResolutionError(
            "Use either --axis-mapping or --axis-preset, not both"
        )
    if mapping_path:
        mapping = AxisMapping.load(mapping_path)
    elif preset_name:
        mapping = AxisPresetStore.default().load(preset_name)
    else:
        mapping = None
    try:
        resolved_mapping = complete_mapping(descriptor, mapping)
    except AmbiguousAxesError:
        gui_enabled = bool(getattr(args, "axis_gui", True))
        is_batch = bool(getattr(args, "batch_manifest", None))
        if not gui_enabled or is_batch:
            raise AxisResolutionError(
                "A TIFF dimension could be time, polarization, or another dimension. Provide --axis-mapping/--axis-preset, or run interactively with --axis-gui."
            )
        from d3h_polarimetry.axis_gui import resolve_axes_interactive

        logger.info("Opening axis mapping window...")
        print("Opening axis mapping window...")
        resolved_mapping = resolve_axes_interactive(descriptor)
    dataset = canonicalize(raw, resolved_mapping)
    selections = parse_axis_selections(getattr(args, "axis_select", None))
    cube = dataset.select_analysis_cube(
        POLARIZATION_CUBE_DIMS, selections=selections
    )
    save_preset_name = getattr(args, "save_axis_preset", None)
    if save_preset_name:
        AxisPresetStore.default().save(save_preset_name, resolved_mapping)
    return (
        np.asarray(cube.values),
        np.asarray(cube.coords["polarization_angle"].values, dtype=float),
        dataset,
        selections,
    )


def _run_single_analysis(args: argparse.Namespace) -> None:
    """Run one polarization analysis from already-resolved arguments."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    curve_cache_path = None
    replay_cache_missing = False
    if args.replay_cache is not None:
        if args.replay_cache == "auto":
            img_path, inferred_save_dir, cache_path = (
                resolve_auto_replay_cache(args)
            )
            save_dir = (
                Path(args.save_dir) if args.save_dir else inferred_save_dir
            )
        else:
            cache_path = AnalysisCache.resolve_path(Path(args.replay_cache))
            if not cache_path.exists():
                logger.error(f"Replay cache not found: {cache_path}")
                sys.exit(1)
            save_dir = (
                Path(args.save_dir)
                if args.save_dir
                else resolve_replay_results_dir(cache_path)
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            img_path = save_dir
        if not cache_path.exists():
            replay_cache_missing = True
            logger.error(f"Replay cache not found: {cache_path}")
            sys.exit(1)
    elif None and args.save_dir and (not args.images):
        save_dir = Path(args.save_dir)
        img_path = save_dir
    else:
        img_path, save_dir = validate_paths(args)
    if args.reuse_curve_cache is not None:
        if args.reuse_curve_cache == "auto":
            _img_path_curve, _save_dir_curve, curve_cache_path = (
                resolve_auto_curve_cache(args)
            )
        else:
            curve_cache_path = Path(args.reuse_curve_cache)
            if curve_cache_path.is_dir():
                nested = (
                    curve_cache_path
                    / "tables"
                    / "curve_fit_cache"
                    / "whole_image_curve_fit_window_data.csv"
                )
                curve_cache_path = (
                    nested
                    if nested.exists()
                    else curve_cache_path
                    / "whole_image_curve_fit_window_data.csv"
                )
        if not curve_cache_path.exists():
            logger.error(f"Curve-fit cache not found: {curve_cache_path}")
            sys.exit(1)
    analysis = _build_analysis_config(args)
    style = _build_figure_style(args)
    if args.figure_format == "svg":
        logger.info(
            "Saving matplotlib figures as SVG. Raw image exports remain PNG."
        )
    elif args.figure_format == "mixed":
        logger.info(
            "Using mixed figure export: vector-friendly plots and overlays as SVG, raster-heavy figures as PNG. Raw image exports remain PNG."
        )
    if style.polar_half_turn:
        logger.info("Rendering polar plots over 0-180 degrees.")
    if style.polar_font_scale != 1.0:
        logger.info(
            "Applying polar font compensation multiplier x%.3f.",
            style.polar_font_scale,
        )
    if style.histogram_font_scale != 1.0:
        logger.info(
            "Applying histogram font compensation multiplier x%.3f.",
            style.histogram_font_scale,
        )
    if style.map_font_scale != 1.0:
        logger.info(
            "Applying map/colorbar font compensation multiplier x%.3f.",
            style.map_font_scale,
        )
    if args.d3hrotating:
        model = D3hRotatingModel()
        fft_mode = FFTMode.D3HROTATING
        print("Using model: d3hrotating (FFT index 6)")
    else:
        model = D3hFixedModel()
        fft_mode = FFTMode.D3HFIXED
        print("Using model: d3hfixed (FFT index 4)")
    session = SessionConfig(
        image_dir=img_path,
        save_dir=save_dir,
        model=model,
        fft_mode=fft_mode,
        analysis=analysis,
        style=style,
        figure_format=args.figure_format,
    )
    if args.replay_cache:
        replay_saved_analysis(session, cache_path)
        return
    print("Loading TIFF input...")
    try:
        max_input_gib = float(getattr(args, "max_input_gib", 8.0))
        max_decoded_bytes = (
            None if max_input_gib == 0.0 else int(max_input_gib * 1024**3)
        )
        if img_path.is_file() and (
            bool(getattr(args, "_num_frames_explicit", False))
            or args.num_frames != 180
        ):
            raise ValueError("--num-frames is supported only for TIFF folders")
        read_result = read_tiff_source_traced(
            img_path,
            sort_by="name",
            tiff_series=getattr(args, "tiff_series", None),
            max_frames=args.num_frames if img_path.is_dir() else None,
            max_decoded_bytes=max_decoded_bytes,
        )
        read_result.trace.write_json(session.save_dir / "reader_trace.json")
        raw_input = read_result.raw
        files = list(raw_input.filenames)
    except (FileNotFoundError, MicroscopyReaderError, ValueError) as e:
        logger.error(f"Failed to load TIFF input: {e}")
        sys.exit(1)
    try:
        stack, angles, named_dataset, axis_selections = (
            _resolve_polarization_raw(raw_input, args)
        )
        named_dataset.write_provenance(
            session.save_dir / "input_axis_mapping.json",
            selections=axis_selections,
        )
    except (AxisResolutionError, OSError, ValueError) as e:
        logger.error(f"Failed to resolve input axes: {e}")
        sys.exit(1)
    n_frames, height, width = stack.shape
    print(f"Stack Loaded: {n_frames} frames, {height}x{width}")
    print("Computing mean and sum images...")
    mean_image, sum_image = _compute_projection_images(stack)
    _save_projection_reference_images(
        session, mean_image, sum_image, suffix="clean"
    )
    crop_bounds = None
    if curve_cache_path is not None:
        crop_bounds, crop_message = resolve_reuse_curve_cache_crop_policy(
            curve_cache_path, ignore_cache_crop=args.ignore_cache_crop
        )
        print(crop_message)
    else:
        print("Opening Crop Selector...")
        crop_sel = CropSelector(mean_image)
        crop_bounds = crop_sel.select_crop()
    if crop_bounds:
        cr_min, cr_max, cc_min, cc_max = crop_bounds
        print(f"Cropping to rows {cr_min}-{cr_max}, cols {cc_min}-{cc_max}")
        stack = stack[:, cr_min:cr_max, cc_min:cc_max]
        mean_image = mean_image[cr_min:cr_max, cc_min:cc_max]
        sum_image = sum_image[cr_min:cr_max, cc_min:cc_max]
        n_frames, height, width = stack.shape
        print(f"Cropped stack: {n_frames} frames, {height}x{width}")
        _save_projection_reference_images(
            session, mean_image, sum_image, suffix="cropped"
        )
    else:
        print("No crop selected, using full image.")
    configure_a0_intensity_mask(session.analysis, mean_image)
    if session.analysis.a0_mask_cutoff is not None:
        print(
            f"A0 intensity mask: keeping pixels with A0 >= {session.analysis.a0_mask_cutoff:.6g} (whole-image p{session.analysis.a0_mask_percentile:g})"
        )
        _show_a0_mask_preview(session, mean_image)
    print("Opening ROI Selector...")
    selector = ROISelector(mean_image)
    rois = selector.select_rois()
    if not rois:
        print("No ROIs selected. Exiting.")
        sys.exit(0)
    print(f"Selected {len(rois)} ROIs. Starting analysis...")
    microns_per_pixel = None
    if args.dat_file:
        try:
            dat = DatParser(args.dat_file)
            _, _, microns_per_pixel = dat.get_scale_info()
            print(f"Scale: {microns_per_pixel:.4f} μm/px")
        except Exception as e:
            logger.warning(f"Could not read scale from DAT file: {e}")
            print(
                f"  [!] DAT parsing failed: {e} — continuing without scale bar"
            )
    run_roimpois_special_pipeline(
        session,
        stack,
        mean_image,
        angles,
        rois,
        sum_image=sum_image,
        crop_bounds=crop_bounds,
        microns_per_pixel=microns_per_pixel,
        curve_fit_cache_path=curve_cache_path,
        do_twist_angle=args.twist_angle,
        ignore_cache_crop=args.ignore_cache_crop,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for polarization analysis."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    preset_file = (
        Path(args.preset_file) if args.preset_file else _default_preset_file()
    )
    resolved_args = resolve_polarization_presets(
        args, raw_argv, parser, preset_file
    )
    if resolved_args is None:
        return
    resolved_args._num_frames_explicit = any(
        (
            token == "--num-frames" or token.startswith("--num-frames=")
            for token in raw_argv
        )
    )
    try:
        _run_single_analysis(resolved_args)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
