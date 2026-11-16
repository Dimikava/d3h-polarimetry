import numpy as np
import pytest
import tifffile

from d3h_polarimetry.io import (
    read_tiff_source_traced,
    ReaderValidationError,
    ReaderDecodeError,
    ReaderResourceError,
)
from d3h_polarimetry.cli.polarization_args import build_parser
from d3h_polarimetry.cli.polarization_config import _build_analysis_config
from d3h_polarimetry.cli.polarization_presets import (
    resolve_polarization_presets,
)
from d3h_polarimetry.pipeline_helpers import (
    _write_curve_cache_metadata,
    _read_curve_cache_metadata,
)
from d3h_polarimetry.cli.polarization_paths import (
    resolve_reuse_curve_cache_crop_policy,
)


def test_strict_folder_ordering_and_trace(tmp_path):
    for name, level in [("002.TIF", 2), ("001.tiff", 1)]:
        tifffile.imwrite(tmp_path / name, np.full((4, 5), level, np.uint16))
    result = read_tiff_source_traced(tmp_path)
    np.testing.assert_array_equal(result.raw.values[:, 0, 0], [1, 2])
    assert result.raw.descriptor.source_dims == ("sequence", "y", "x")
    assert len(result.trace.source_files) == 2
    assert result.trace.backend_versions["tifffile"]


def test_corrupt_frame_fails_whole_read(tmp_path):
    tifffile.imwrite(tmp_path / "000.tif", np.ones((4, 5), np.uint16))
    (tmp_path / "001.tif").write_bytes(b"not a tiff")
    with pytest.raises((ReaderValidationError, ReaderDecodeError)):
        read_tiff_source_traced(tmp_path)


def test_inconsistent_frames_and_resource_limit(tmp_path):
    tifffile.imwrite(tmp_path / "000.tif", np.ones((4, 5), np.uint16))
    with pytest.raises(ReaderResourceError):
        read_tiff_source_traced(tmp_path, max_decoded_bytes=1)
    tifffile.imwrite(tmp_path / "001.tif", np.ones((5, 5), np.uint16))
    with pytest.raises(ReaderValidationError):
        read_tiff_source_traced(tmp_path)


def test_single_tiff_stack_and_explicit_series(tmp_path):
    path = tmp_path / "stack.tif"
    values = np.arange(36 * 5 * 6, dtype=np.uint16).reshape(36, 5, 6)
    tifffile.imwrite(
        path, values, photometric="minisblack", metadata={"axes": "QYX"}
    )
    np.testing.assert_array_equal(
        read_tiff_source_traced(path).raw.values, values
    )
    multi = tmp_path / "multi.tif"
    with tifffile.TiffWriter(multi) as writer:
        writer.write(np.ones((5, 6), np.uint16))
        writer.write(np.ones((7, 8), np.uint16))
    with pytest.raises(ReaderValidationError):
        read_tiff_source_traced(multi)
    assert read_tiff_source_traced(multi, tiff_series=1).raw.values.shape == (
        7,
        8,
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--d3hanal",
        "--d3hmov",
        "--myelin",
        "--thgmm2",
        "--roimpois",
        "--roimpois-special2",
        "--patch",
        "--template",
        "--batch-manifest",
    ],
)
def test_removed_options_rejected(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag])


def test_model_exclusivity_and_window_configuration():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--d3hfixed", "--d3hrotating"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--window-size", "0"])
    config = _build_analysis_config(
        parser.parse_args(["--window-size", "3", "--fft-only"])
    )
    assert (
        config.window_size == 3
        and not config.enable_curve_fit
        and not config.patch_data
    )


def test_preset_round_trip_and_explicit_model_override(tmp_path):
    parser = build_parser()
    file = tmp_path / "presets.json"
    argv = ["--d3hrotating", "--window-size", "3", "--save-preset", "test"]
    resolve_polarization_presets(parser.parse_args(argv), argv, parser, file)
    argv = ["--preset", "test", "--d3hfixed"]
    args = resolve_polarization_presets(
        parser.parse_args(argv), argv, parser, file
    )
    assert args.d3hfixed and not args.d3hrotating
    assert args.window_size == 3


def test_curve_cache_crop_policy_round_trip(tmp_path):
    path = tmp_path / "cache.csv"
    path.write_text("Window R,Window C\n0,0\n", encoding="utf-8")
    metadata = {"crop_bounds": [2, 12, 3, 13], "window_size": 3}
    _write_curve_cache_metadata(path, metadata)
    assert _read_curve_cache_metadata(path) == metadata
    crop, _ = resolve_reuse_curve_cache_crop_policy(
        path, ignore_cache_crop=False
    )
    assert crop == (2, 12, 3, 13)
    crop, _ = resolve_reuse_curve_cache_crop_policy(
        path, ignore_cache_crop=True
    )
    assert crop is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("fft_mode", "d3hrotating"),
        ("window_size", 7),
        ("num_frames", 72),
        ("image_dir", "another_dataset"),
        ("crop_bounds", [1, 9, 1, 9]),
        ("cropped_stack_shape", [36, 10, 10]),
        ("a0_mask_percentile", 0),
        ("a0_mask_cutoff", 1),
    ],
)
def test_curve_cache_rejects_wrong_identity(tmp_path, field, value):
    from d3h_polarimetry.pipeline_config import SessionConfig, AnalysisConfig
    from d3h_polarimetry.models import D3hFixedModel, FFTMode
    from d3h_polarimetry.pipeline_helpers import (
        _curve_cache_identity,
        validate_curve_cache_identity,
    )

    session = SessionConfig(
        tmp_path,
        tmp_path / "out",
        D3hFixedModel(),
        FFTMode.D3HFIXED,
        AnalysisConfig(window_size=3),
    )
    cache = tmp_path / "curve.csv"
    meta = _curve_cache_identity(
        session, crop_bounds=None, cropped_shape=(36, 12, 12)
    )
    meta[field] = value
    _write_curve_cache_metadata(cache, meta)
    with pytest.raises(ValueError, match=field):
        validate_curve_cache_identity(
            cache, session, crop_bounds=None, cropped_shape=(36, 12, 12)
        )


def test_curve_cache_crop_override_is_explicit(tmp_path):
    from d3h_polarimetry.pipeline_config import SessionConfig
    from d3h_polarimetry.models import D3hFixedModel, FFTMode
    from d3h_polarimetry.pipeline_helpers import (
        _curve_cache_identity,
        validate_curve_cache_identity,
    )

    session = SessionConfig(
        tmp_path, tmp_path / "out", D3hFixedModel(), FFTMode.D3HFIXED
    )
    cache = tmp_path / "curve.csv"
    _write_curve_cache_metadata(
        cache,
        _curve_cache_identity(
            session, crop_bounds=(1, 9, 1, 9), cropped_shape=(36, 8, 8)
        ),
    )
    validate_curve_cache_identity(
        cache,
        session,
        crop_bounds=None,
        cropped_shape=(36, 12, 12),
        ignore_cache_crop=True,
    )


@pytest.mark.parametrize("change", ["angles", "pixels"])
def test_curve_cache_rejects_changed_acquisition(tmp_path, change):
    from d3h_polarimetry.pipeline_config import SessionConfig
    from d3h_polarimetry.models import D3hFixedModel, FFTMode
    from d3h_polarimetry.pipeline_helpers import (
        _curve_cache_identity,
        validate_curve_cache_identity,
    )

    session = SessionConfig(
        tmp_path, tmp_path / "out", D3hFixedModel(), FFTMode.D3HFIXED
    )
    cache = tmp_path / "curve.csv"
    stack = np.ones((36, 12, 12))
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    _write_curve_cache_metadata(
        cache,
        _curve_cache_identity(
            session,
            crop_bounds=None,
            cropped_shape=stack.shape,
            stack=stack,
            angles=angles,
        ),
    )
    if change == "angles":
        angles = angles + 0.1
    else:
        stack[0, 0, 0] = 2
    with pytest.raises(ValueError, match="angles_rad|input_sha256"):
        validate_curve_cache_identity(
            cache,
            session,
            crop_bounds=None,
            cropped_shape=stack.shape,
            stack=stack,
            angles=angles,
        )


def test_dat_scale_and_numeric_parsing(tmp_path):
    from d3h_polarimetry.reader import DatParser

    path = tmp_path / "sample.dat"
    path.write_text(
        "Scan Settings Kami:\nNumber of lines: 1200\nScan field X: 150.5 um\nTop Left h: -1,25 V\nGain: 1.2e3 counts\n",
        encoding="utf-8",
    )
    parser = DatParser(path)
    assert parser.get_scale_info() == pytest.approx(
        (150.5, 1200, 150.5 / 1200)
    )
    assert parser.get_section("Scan Settings Kami")["Top Left h"] == -1.25
    assert parser.get_section("Scan Settings Kami")["Gain"] == 1200
