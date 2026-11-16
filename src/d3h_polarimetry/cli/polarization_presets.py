"""Named preset support for the polarization CLI."""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any

POLARIZATION_PRESET_FIELDS = (
    "replay_cache",
    "reuse_curve_cache",
    "ignore_cache_crop",
    "figure_format",
    "half_polar",
    "hist_font_scale",
    "map_font_scale",
    "polar_font_scale",
    "inspect_histograms",
    "roi_hist_bins",
    "poi_hist_bins",
    "roimpois_special",
    "crop",
    "twist_angle",
    "temp",
    "temp_fix",
    "r2_threshold",
    "a0_mask_percentile",
    "tiff_series",
    "max_input_gib",
    "num_frames",
    "verbose",
    "d3hfixed",
    "d3hrotating",
    "optimizer",
    "fft_only",
    "phi_vmax",
    "phi_vmin",
    "window_size",
)
POLARIZATION_PRESET_OPTIONS = {
    "replay_cache": "--replay-cache",
    "reuse_curve_cache": "--reuse-curve-cache",
    "ignore_cache_crop": "--ignore-cache-crop",
    "figure_format": "--figure-format",
    "half_polar": "--half-polar",
    "hist_font_scale": "--hist-font-scale",
    "map_font_scale": "--map-font-scale",
    "polar_font_scale": "--polar-font-scale",
    "inspect_histograms": "--inspect-histograms",
    "roi_hist_bins": "--roi-hist-bins",
    "poi_hist_bins": "--poi-hist-bins",
    "roimpois_special": "--roimpois-special",
    "crop": "--crop",
    "twist_angle": "--twist-angle",
    "temp": "--temp",
    "temp_fix": "--temp-fix",
    "r2_threshold": "--r2-threshold",
    "a0_mask_percentile": "--a0-mask-percentile",
    "tiff_series": "--tiff-series",
    "max_input_gib": "--max-input-gib",
    "num_frames": "--num-frames",
    "verbose": ("--verbose", "-v"),
    "d3hfixed": "--d3hfixed",
    "d3hrotating": "--d3hrotating",
    "optimizer": "--optimizer",
    "fft_only": "--fft-only",
    "phi_vmax": "--phi-vmax",
    "phi_vmin": "--phi-vmin",
    "window_size": "--window-size",
}
MODEL_FIELDS = {"d3hfixed", "d3hrotating"}
ROI_MODE_FIELDS = {"roimpois_special"}


def _default_preset_file() -> Path:
    """Return the default polarization preset file path."""
    return Path.home() / ".d3h_polarimetry" / "polarization_presets.json"


def _validate_preset_name(name: str) -> None:
    """Validate a preset name before using it as a JSON key."""
    if not re.fullmatch("[A-Za-z0-9_.-]+", name):
        raise ValueError(
            "Preset names may only contain letters, numbers, '.', '_', and '-'"
        )


def _load_presets(preset_file: Path) -> dict[str, dict[str, Any]]:
    """Load named polarization presets from disk."""
    if not preset_file.exists():
        return {}
    with preset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Preset file must contain a JSON object: {preset_file}"
        )
    presets: dict[str, dict[str, Any]] = {}
    for name, settings in data.items():
        if isinstance(name, str) and isinstance(settings, dict):
            presets[name] = settings
    return presets


def _save_presets(
    preset_file: Path, presets: dict[str, dict[str, Any]]
) -> None:
    """Write named polarization presets to disk."""
    preset_file.parent.mkdir(parents=True, exist_ok=True)
    with preset_file.open("w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, sort_keys=True)
        f.write("\n")


def _option_was_provided(
    argv: list[str], options: str | tuple[str, ...]
) -> bool:
    """Return True when any option appears in argv, including --flag=value form."""
    if isinstance(options, str):
        options = (options,)
    return any(
        (
            arg == option or arg.startswith(f"{option}=")
            for option in options
            for arg in argv
        )
    )


def _provided_preset_fields(argv: list[str]) -> set[str]:
    """Return preset fields explicitly provided by the user."""
    fields = {
        field
        for field, option in POLARIZATION_PRESET_OPTIONS.items()
        if _option_was_provided(argv, option)
    }
    if fields & MODEL_FIELDS:
        fields.update(MODEL_FIELDS)
    if fields & ROI_MODE_FIELDS:
        fields.update(ROI_MODE_FIELDS)
    return fields


def _coerce_preset_value(field: str, value: Any) -> Any:
    """Coerce JSON-loaded preset values into parser-equivalent values."""
    if field == "template_output" and value is not None:
        return Path(value).expanduser()
    return value


def _jsonable_preset_value(value: Any) -> Any:
    """Convert parser values to JSON-safe preset values."""
    if isinstance(value, Path):
        return str(value)
    return value


def _apply_preset(
    args: argparse.Namespace, preset: dict[str, Any], provided_fields: set[str]
) -> None:
    """Apply preset values for options not explicitly supplied on the command line."""
    for field in POLARIZATION_PRESET_FIELDS:
        if field in preset and field not in provided_fields:
            setattr(args, field, _coerce_preset_value(field, preset[field]))


def _collect_preset_values(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the current polarization invocation as a reusable preset."""
    return {
        field: _jsonable_preset_value(getattr(args, field, None))
        for field in POLARIZATION_PRESET_FIELDS
    }


def resolve_polarization_presets(
    args: argparse.Namespace,
    argv: list[str],
    parser: argparse.ArgumentParser,
    preset_file: Path,
) -> argparse.Namespace | None:
    """Resolve parsed CLI arguments after applying named presets."""
    try:
        presets = _load_presets(preset_file)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        parser.error(f"Could not load polarization presets: {e}")
    if args.list_presets:
        if presets:
            for name in sorted(presets):
                print(name)
        else:
            print("No polarization presets saved.")
        return None
    if args.preset:
        try:
            _validate_preset_name(args.preset)
        except ValueError as e:
            parser.error(str(e))
        if args.preset not in presets:
            parser.error(f"Polarization preset not found: {args.preset}")
        _apply_preset(
            args, presets[args.preset], _provided_preset_fields(argv)
        )
    if args.save_preset:
        try:
            _validate_preset_name(args.save_preset)
        except ValueError as e:
            parser.error(str(e))
        presets[args.save_preset] = _collect_preset_values(args)
        try:
            _save_presets(preset_file, presets)
        except OSError as e:
            parser.error(f"Could not save polarization preset: {e}")
        print(
            f"Saved polarization preset '{args.save_preset}' to: {preset_file}"
        )
    return args
