"""Run actual plotting/export/replay; replace only interactive user selections."""

from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import tifffile
import matplotlib.pyplot as plt

from d3h_polarimetry.models import D3hFixedModel, D3hRotatingModel, FFTMode
from d3h_polarimetry.pipeline_config import AnalysisConfig, SessionConfig
from d3h_polarimetry.pipeline_workflow_special import (
    run_roimpois_special_pipeline,
)
from d3h_polarimetry.pipeline_replay import replay_saved_analysis
from d3h_polarimetry.cache import AnalysisCache
from d3h_polarimetry.visualization import FigureStyle


@pytest.mark.parametrize(
    "model,mode,figure_format,curve",
    [
        (D3hFixedModel(), FFTMode.D3HFIXED, "svg", True),
        (D3hRotatingModel(), FFTMode.D3HROTATING, "mixed", False),
    ],
)
def test_special_workflow_twist_export_and_replay(
    tmp_path, monkeypatch, model, mode, figure_format, curve
):
    import d3h_polarimetry.gui as gui

    def select_pois(self):
        plt.close(self.fig)
        return [(3, 3), (8, 8)]

    monkeypatch.setattr(gui.POISelector, "select_pois", select_pois)
    monkeypatch.setattr(
        gui.POIPairSelector, "select_pairs", lambda self: [(0, 1)]
    )
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    phi = np.full((12, 12), np.deg2rad(12.0))
    phi[:, 6:] = np.deg2rad(32.0)
    stack = np.stack([model.model_function(a, 500, 120, phi) for a in angles])
    session = SessionConfig(
        tmp_path,
        tmp_path / "results",
        model,
        mode,
        AnalysisConfig(
            window_size=3, a0_mask_percentile=None, enable_curve_fit=curve
        ),
        FigureStyle(dpi=40),
        figure_format=figure_format,
    )
    run_roimpois_special_pipeline(
        session,
        stack,
        stack.mean(0),
        angles,
        [(0, 12, 0, 12)],
        sum_image=stack.sum(0),
        do_twist_angle=True,
    )
    cache_path = session.cache_path(AnalysisCache.DEFAULT_FILENAME)
    assert cache_path.is_file()
    metadata, arrays = AnalysisCache.load(cache_path)
    assert metadata["pipeline_mode"] == "roimpois_special"
    assert metadata["analysis"]["window_size"] == 3
    assert arrays["whole_fft_angle"].shape == (10, 10)
    assert any("twist" in k for k in arrays)
    figures = list(session.save_dir.rglob("*.svg"))
    assert figures and all(
        "<svg" in f.read_text(encoding="utf-8") for f in figures
    )
    assert any("twist_angle" in str(f) for f in session.save_dir.rglob("*"))
    assert list(session.save_dir.rglob("*.csv"))
    replay_session = replace(session, save_dir=tmp_path / "replayed")
    replay_saved_analysis(replay_session, cache_path)
    assert list(replay_session.save_dir.rglob("twist_angle*colormap*"))
    assert list(replay_session.save_dir.rglob("twist_angle*histogram*"))
    assert not __import__(
        "matplotlib.pyplot", fromlist=["get_fignums"]
    ).get_fignums()
    if curve:
        original_curve = arrays["whole_curve_angle"].copy()
        curve_table = next(
            session.save_dir.rglob("whole_image_curve_fit_window_data.csv")
        )
        run_roimpois_special_pipeline(
            session,
            stack,
            stack.mean(0),
            angles,
            [(0, 12, 0, 12)],
            sum_image=stack.sum(0),
            curve_fit_cache_path=curve_table,
        )
        _, reused = AnalysisCache.load(cache_path)
        np.testing.assert_allclose(
            reused["whole_curve_angle"],
            original_curve,
            atol=1e-12,
            equal_nan=True,
        )


def test_cli_tiff_to_special_pipeline(tmp_path, monkeypatch):
    from d3h_polarimetry.cli.run_polarization import main
    import d3h_polarimetry.cli.run_polarization as cli
    import d3h_polarimetry.gui as gui
    from d3h_polarimetry.axis_data import AxisMapping

    monkeypatch.setattr(
        gui.POISelector, "select_pois", lambda self: plt.close(self.fig) or []
    )
    monkeypatch.setattr(cli.CropSelector, "select_crop", lambda self: None)
    monkeypatch.setattr(
        cli.ROISelector, "select_rois", lambda self: [(0, 5, 0, 5)]
    )
    monkeypatch.setattr(cli, "_show_a0_mask_preview", lambda *args: None)
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    data = D3hFixedModel().model_function(angles, 500, 120, np.deg2rad(23))
    source = tmp_path / "images"
    source.mkdir()
    for i, value in enumerate(data):
        tifffile.imwrite(
            source / f"{i:03}.tif", np.full((5, 5), value, dtype=np.float32)
        )
    mapping_path = tmp_path / "axis.json"
    AxisMapping(
        assignments={"sequence": "polarization_angle", "y": "y", "x": "x"},
        coordinates={"polarization_angle": tuple(angles)},
        units={"polarization_angle": "rad"},
    ).save(mapping_path)
    main(
        [
            "--images",
            str(source),
            "--save-dir",
            str(tmp_path / "out"),
            "--axis-mapping",
            str(mapping_path),
            "--no-axis-gui",
            "--fft-only",
            "--window-size",
            "3",
            "--figure-format",
            "png",
            "--preset-file",
            str(tmp_path / "presets.json"),
        ]
    )
    output = tmp_path / "out" / "roimpois_special"
    assert (output / "reader_trace.json").is_file()
    assert (output / "input_axis_mapping.json").is_file()
    assert list(output.rglob("*.png"))
    assert list(output.rglob("*.npz"))
