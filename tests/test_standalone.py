"""Standalone package and renamed CLI contracts."""

import importlib.util


def test_standalone_package_exists():
    assert importlib.util.find_spec("d3h_polarimetry") is not None


def test_default_special_workflow_and_models():
    from d3h_polarimetry.cli.polarization_args import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.roimpois_special is True
    assert args.twist_angle is False
    assert args.d3hrotating is False
    assert parser.parse_args(["--d3hrotating"]).d3hrotating
    assert parser.parse_args(["--d3hfixed"]).d3hfixed


def test_only_two_models_are_exposed():
    from d3h_polarimetry.models import FFTMode

    assert {mode.value for mode in FFTMode} == {"d3hfixed", "d3hrotating"}
