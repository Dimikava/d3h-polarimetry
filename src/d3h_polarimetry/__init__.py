"""D3h fixed- and rotating-analyzer SHG polarimetry.

Imports stay lazy so TIFF inspection does not initialize the GUI.
"""

from importlib import import_module

__version__ = "0.1.0"
_EXPORTS = {
    "D3hFixedModel": ("models", "D3hFixedModel"),
    "D3hRotatingModel": ("models", "D3hRotatingModel"),
    "FFTMode": ("models", "FFTMode"),
    "PolarizationAnalyzer": ("polarization", "PolarizationAnalyzer"),
    "AnalysisConfig": ("pipeline_config", "AnalysisConfig"),
    "SessionConfig": ("pipeline_config", "SessionConfig"),
    "SlidingWindowAnalyzer": ("pipeline_sliding", "SlidingWindowAnalyzer"),
    "ArithmeticProcessor": ("pipeline_arithmetic", "ArithmeticProcessor"),
    "run_roimpois_special_pipeline": (
        "pipeline_workflow_special",
        "run_roimpois_special_pipeline",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), symbol)
    globals()[name] = value
    return value
