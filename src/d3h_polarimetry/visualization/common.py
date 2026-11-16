"""
TIFF Analyzer - Visualization Module

Provides publication-grade plotting utilities for polarization analysis results.
All figures are generated at 300 DPI with consistent styling for academic publication.

Classes:
    FigureStyle: Centralized styling configuration.
    PolarPlotter: Generates polar plots for FFT/curve fit results.
    HistogramPlotter: Creates orientation distribution histograms.
    OrientationMapPlotter: Renders 2D orientation heatmaps.
    ArithmeticPlotter: Specialized plots for arithmetic analysis with error propagation.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)


def _apply_matplotlib_text_style(
    font_family: Union[str, Sequence[str]],
    base_size: int,
    *,
    title_size: Optional[int] = None,
    label_size: Optional[int] = None,
    legend_size: Optional[int] = None,
) -> None:
    """Apply a consistent text style for exported matplotlib figures."""
    if isinstance(font_family, str):
        families = [font_family]
    else:
        families = [str(name) for name in font_family if str(name).strip()]
    plt.rcParams["font.family"] = families
    plt.rcParams["font.serif"] = families
    plt.rcParams["font.size"] = base_size
    plt.rcParams["axes.titlesize"] = (
        title_size if title_size is not None else base_size
    )
    plt.rcParams["axes.labelsize"] = (
        label_size if label_size is not None else base_size
    )
    plt.rcParams["legend.fontsize"] = (
        legend_size if legend_size is not None else base_size
    )
    plt.rcParams["xtick.labelsize"] = (
        label_size if label_size is not None else base_size
    )
    plt.rcParams["ytick.labelsize"] = (
        label_size if label_size is not None else base_size
    )
    plt.rcParams["svg.fonttype"] = "none"


_apply_matplotlib_text_style("Times New Roman", 12)


@dataclass
class FigureStyle:
    """
    Centralized configuration for publication-grade figure styling.

    Attributes:
        dpi: Resolution for saved figures (default: 300 for publication).
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        legend_fontsize: Font size for legend text.
        colormap: Default colormap for heatmaps.
        background_color: Background color for masked regions.

    Example:
        >>> style = FigureStyle(dpi=150)  # Lower DPI for faster previews
        >>> plotter = PolarPlotter(style=style)
    """

    dpi: int = 300
    title_fontsize: int = 12
    label_fontsize: int = 12
    legend_fontsize: int = 12
    annotation_fontsize: int = 12
    polar_tick_label_fontsize: float = 15
    polar_title_fontsize: float = 17
    polar_legend_fontsize: float = 14
    polar_font_scale: float = 1.0
    histogram_font_scale: float = 1.0
    map_font_scale: float = 1.0
    polar_theta_tick_pad: float = 30.0
    polar_radial_tick_pad: float = 6.0
    polar_half_turn: bool = False
    inspect_histograms: bool = False
    font_family: str = "Times New Roman"
    colormap: str = "jet"
    background_color: str = "black"
    spine_visible: bool = False
    fft_color: str = "yellow"
    curve_color: str = "red"
    data_color: str = "blue"
    histogram_fft_color: Optional[str] = None
    histogram_curve_color: Optional[str] = None
    mean_line_color: str = "black"
    std_region_color: str = "#D3D3D3"

    def __post_init__(self) -> None:
        if self.histogram_fft_color is None:
            self.histogram_fft_color = self.fft_color
        if self.histogram_curve_color is None:
            self.histogram_curve_color = self.curve_color
        _apply_matplotlib_text_style(
            self.font_family,
            self.label_fontsize,
            title_size=self.title_fontsize,
            label_size=self.label_fontsize,
            legend_size=self.legend_fontsize,
        )

    @property
    def effective_polar_tick_label_fontsize(self) -> float:
        return self.polar_tick_label_fontsize * self.polar_font_scale

    @property
    def effective_polar_title_fontsize(self) -> float:
        return self.polar_title_fontsize * self.polar_font_scale

    @property
    def effective_polar_legend_fontsize(self) -> float:
        return self.polar_legend_fontsize * self.polar_font_scale

    @property
    def effective_histogram_font_scale(self) -> float:
        return self.histogram_font_scale

    @property
    def effective_map_title_fontsize(self) -> float:
        return TITLE_FONTSIZE * self.map_font_scale

    @property
    def effective_map_label_fontsize(self) -> float:
        return AXIS_LABEL_FONTSIZE * self.map_font_scale

    @property
    def effective_map_tick_label_fontsize(self) -> float:
        return AXIS_TICK_LABEL_FONTSIZE * self.map_font_scale


@dataclass
class PlotResult:
    """Result from a plotting operation."""

    filepath: Path
    figure: Optional[plt.Figure] = None
    success: bool = True
    error: Optional[str] = None


def _prepare_save_path(save_path: Union[str, Path]) -> Path:
    """Normalize a save path and ensure its parent directory exists."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


HISTOGRAM_SINGLE_PANEL_FIGSIZE: Tuple[float, float] = (6.0, 4.5)
HISTOGRAM_DUAL_PANEL_FIGSIZE: Tuple[float, float] = (6.0, 9.0)
MAP_FIGSIZE: Tuple[float, float] = (8.0, 6.0)
POLAR_FIGSIZE: Tuple[float, float] = (8.0, 8.0)
AXIS_TICK_LABEL_FONTSIZE: int = 15
AXIS_LABEL_FONTSIZE: int = 17
TITLE_FONTSIZE: int = 17
LEGEND_FONTSIZE: int = 14
SPINE_LINEWIDTH: float = 1.3
POLAR_SPINE_LINEWIDTH: float = 1.4
GRID_LINEWIDTH: float = 1.0
TICK_LINEWIDTH: float = 1.1
TICK_LENGTH: float = 6.0
BAR_EDGE_LINEWIDTH: float = 0.9
HISTOGRAM_ALPHA: float = 0.75
MAJOR_X_TICKS: int = 5
MAJOR_Y_TICKS: int = 6
COLORBAR_FRACTION: float = 0.046
COLORBAR_PAD: float = 0.04


def _apply_figure_layout(
    fig: plt.Figure,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    hspace: Optional[float] = None,
) -> None:
    """Apply consistent roomier margins for exported figures."""
    fig.tight_layout(pad=1.0)
    kwargs: Dict[str, float] = {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
    }
    if hspace is not None:
        kwargs["hspace"] = hspace
    fig.subplots_adjust(**kwargs)


def _format_histogram_tick_label(value: float, _position: float) -> str:
    """Format histogram ticks without distracting float noise."""
    if not np.isfinite(value):
        return ""
    rounded_int = round(value)
    if np.isclose(value, rounded_int, atol=1e-09):
        return str(int(rounded_int))
    if np.isclose(value, round(value, 1), atol=1e-09):
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _style_histogram_axes(
    ax: plt.Axes,
    *,
    spine_visible: bool,
    use_default_x_locator: bool = True,
    tick_label_fontsize: float = AXIS_TICK_LABEL_FONTSIZE,
) -> None:
    """Apply publication-ready axis styling for histogram-like panels."""
    for spine_name, spine in ax.spines.items():
        if spine_name in {"top", "right"} and (not spine_visible):
            spine.set_visible(False)
            continue
        spine.set_linewidth(SPINE_LINEWIDTH)
    ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_label_fontsize,
        width=TICK_LINEWIDTH,
        length=TICK_LENGTH,
    )
    if use_default_x_locator:
        ax.xaxis.set_major_locator(
            MaxNLocator(
                nbins=MAJOR_X_TICKS, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10]
            )
        )
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=MAJOR_Y_TICKS, min_n_ticks=4, integer=True)
    )


def _style_polar_axes_shared(
    ax: plt.Axes,
    *,
    show_radial_ticks: bool,
    tick_label_fontsize: float = AXIS_TICK_LABEL_FONTSIZE,
    theta_tick_pad: float = 12.0,
    radial_tick_pad: float = 6.0,
    half_turn: bool = False,
) -> None:
    """Increase polar axis visibility for exported figures."""
    theta_degrees = (
        np.arange(0, 181, 45) if half_turn else np.arange(0, 360, 45)
    )
    if half_turn:
        ax.set_thetamin(0.0)
        ax.set_thetamax(180.0)
    ax.set_thetagrids(
        theta_degrees, labels=[f"{deg}°" for deg in theta_degrees]
    )
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=tick_label_fontsize,
        width=TICK_LINEWIDTH,
        length=TICK_LENGTH,
        pad=theta_tick_pad,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=tick_label_fontsize,
        width=TICK_LINEWIDTH,
        length=TICK_LENGTH,
        pad=radial_tick_pad,
    )
    ax.spines["polar"].set_linewidth(POLAR_SPINE_LINEWIDTH)
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=0.7, color="#666666")
    if not show_radial_ticks:
        ax.set_yticks([])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))


def _style_colorbar(
    cbar: Any,
    *,
    label: Optional[str] = None,
    font_scale: float = 1.0,
    nbins: Optional[int] = None,
) -> None:
    """Apply consistent colorbar typography, tick density, and line weight."""
    if label is not None:
        cbar.set_label(label, fontsize=AXIS_LABEL_FONTSIZE * font_scale)
    cbar.outline.set_linewidth(SPINE_LINEWIDTH)
    axis = cbar.ax.yaxis if cbar.orientation == "vertical" else cbar.ax.xaxis
    axis.set_major_locator(
        MaxNLocator(
            nbins=nbins or MAJOR_Y_TICKS,
            min_n_ticks=4,
            steps=[1, 2, 2.5, 5, 10],
        )
    )
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(
        axis="both",
        which="major",
        labelsize=AXIS_TICK_LABEL_FONTSIZE * font_scale,
        width=TICK_LINEWIDTH,
        length=TICK_LENGTH,
    )
    cbar.update_ticks()


def _prepare_polar_series(
    angles: np.ndarray,
    values: np.ndarray,
    *,
    half_turn: bool,
    close_loop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return plotting-ready polar coordinates, optionally wrapped to [0, pi]."""
    angles_array = np.asarray(angles, dtype=float)
    values_array = np.asarray(values, dtype=float)
    if half_turn:
        angles_array = np.mod(angles_array, np.pi)
        order = np.argsort(angles_array, kind="stable")
        angles_array = angles_array[order]
        values_array = values_array[order]
    if close_loop and angles_array.size:
        closure_angle = np.pi if half_turn else angles_array[0]
        angles_array = np.append(angles_array, closure_angle)
        values_array = np.append(values_array, values_array[0])
    return (angles_array, values_array)
