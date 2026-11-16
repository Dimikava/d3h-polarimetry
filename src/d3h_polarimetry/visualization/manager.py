"""Specialized visualization plotter implementation."""

from __future__ import annotations
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from .common import FigureStyle, logger


def _style_colorbar(
    cbar: Any,
    *,
    label: Optional[str] = None,
    font_scale: float = 1.0,
    nbins: Optional[int] = None,
) -> None:
    """Delegate through the package facade so compatibility monkeypatches still work."""
    from . import _style_colorbar as package_style_colorbar

    return package_style_colorbar(
        cbar, label=label, font_scale=font_scale, nbins=nbins
    )


from .arithmetic import ArithmeticPlotter
from .histogram import HistogramPlotter
from .maps import OrientationMapPlotter
from .overlay import OverlayPlotter
from .polar import PolarPlotter


class VisualizationManager:
    """
    Facade for managing all specialized plotters with a single style configuration.

    This manager provides centralized access to all plotting utilities,
    preventing the need to instantiate multiple individual plotters in
    the pipeline orchestrators.

    Attributes:
        style: Centralized figure styling configuration.
        polar: Polar plotter instance.
        histogram: Histogram plotter instance.
        orientation: Orientation map plotter instance.
        arithmetic: Arithmetic map plotter instance.
        overlay: Image overlay plotter instance.
        perovskite: Perovskite specific plotter instance.
    """

    def __init__(self, style: Optional[FigureStyle] = None) -> None:
        """
        Initialize the visualization manager with a unified style.

        Args:
            style: Styling configuration. Uses defaults if not provided.
        """
        self.style = style or FigureStyle()
        self.polar = PolarPlotter(self.style)
        self.histogram = HistogramPlotter(self.style)
        self.orientation = OrientationMapPlotter(self.style)
        self.arithmetic = ArithmeticPlotter(self.style)
        self.overlay = OverlayPlotter(self.style)
        logger.debug("VisualizationManager initialized")
