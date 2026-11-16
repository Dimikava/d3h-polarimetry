"""Specialized visualization plotter implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .common import (
    AXIS_LABEL_FONTSIZE,
    AXIS_TICK_LABEL_FONTSIZE,
    BAR_EDGE_LINEWIDTH,
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    HISTOGRAM_ALPHA,
    HISTOGRAM_SINGLE_PANEL_FIGSIZE,
    LEGEND_FONTSIZE,
    MAP_FIGSIZE,
    TITLE_FONTSIZE,
    FigureStyle,
    PlotResult,
    _apply_figure_layout,
    _prepare_save_path,
    _style_histogram_axes,
    logger,
)


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


from .histogram import HistogramPlotter


class ArithmeticPlotter:
    """
    Generates specialized plots for arithmetic analysis with error propagation.

    Creates both 2D maps and histograms showing transformed values
    with propagated uncertainty visualization.

    Attributes:
        style: Figure styling configuration.
    """

    def __init__(self, style: Optional[FigureStyle] = None) -> None:
        """
        Initialize the arithmetic plotter.

        Args:
            style: Styling configuration. Uses defaults if not provided.
        """
        self.style = style or FigureStyle()
        logger.debug("ArithmeticPlotter initialized")

    def save_arithmetic_map(
        self,
        save_path: Union[str, Path],
        title: str,
        result_map: np.ndarray,
        sigma_z: float,
        figsize: Tuple[float, float] = MAP_FIGSIZE,
    ) -> PlotResult:
        """
        Save an arithmetic result map with transparent background.

        Args:
            save_path: Output file path.
            title: Plot title.
            result_map: 2D array of transformed values.
            sigma_z: Propagated error for annotation.
            figsize: Figure dimensions.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        try:
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            masked_result = np.ma.masked_invalid(result_map)
            cmap = plt.get_cmap(self.style.colormap).copy()
            cmap = cmap.with_extremes(bad=(0, 0, 0, 0))
            im = ax.imshow(masked_result, cmap=cmap, origin="upper")
            cbar = plt.colorbar(
                im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
            )
            _style_colorbar(
                cbar,
                label="Twist angle δ (°)",
                font_scale=self.style.map_font_scale,
            )
            ax.set_title(
                title,
                fontsize=self.style.effective_map_title_fontsize,
                fontweight="bold",
            )
            ax.axis("off")
            _apply_figure_layout(
                fig, left=0.08, right=0.92, bottom=0.08, top=0.94
            )
            fig.savefig(save_path, dpi=self.style.dpi, transparent=True)
            plt.close(fig)
            logger.debug(f"Saved arithmetic map: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save arithmetic map: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def save_arithmetic_histogram(
        self,
        save_path: Union[str, Path],
        title: str,
        values: Sequence[float],
        sigma_z: float,
        n_bins: int = 30,
        figsize: Tuple[float, float] = HISTOGRAM_SINGLE_PANEL_FIGSIZE,
        histogram_color: Optional[str] = None,
    ) -> PlotResult:
        """
        Save a histogram of arithmetic results using statistics from the plotted data.

        The histogram includes:
        - Mean line
        - Shaded region showing ±1 std of the plotted histogram values

        Args:
            save_path: Output file path.
            title: Plot title.
            values: Array of transformed values.
            sigma_z: Propagated error value.
            n_bins: Number of histogram bins.
            figsize: Figure dimensions.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        if not values:
            logger.warning("No data provided for arithmetic histogram")
            return PlotResult(
                filepath=save_path, success=False, error="No data"
            )
        try:
            path_text = str(save_path).lower()
            title_text = title.lower()
            if histogram_color is None:
                if "fft" in path_text or "fft" in title_text:
                    histogram_color = self.style.histogram_fft_color
                elif "curve" in path_text or "curve" in title_text:
                    histogram_color = self.style.histogram_curve_color
                else:
                    histogram_color = "purple"
            x_label = "Twist angle δ (°)"
            y_label = "Number of pixels"
            show_title = False
            show_statistics = True
            statistics_mode = "mean_std"
            xlim: Optional[Tuple[Optional[float], Optional[float]]] = None
            ylim: Optional[Tuple[Optional[float], Optional[float]]] = None
            inspected = HistogramPlotter(
                self.style
            )._inspect_histogram_options(
                title=title,
                data=values,
                count_total=len(values),
                color=histogram_color,
                bin_fraction=0.0,
                n_bins_override=n_bins,
                figsize=figsize,
                show_title=show_title,
                show_statistics=show_statistics,
                statistics_mode=statistics_mode,
                x_label=x_label,
                y_label=y_label,
            )
            if inspected is not None:
                if inspected.get("cancel_inspection"):
                    self.style.inspect_histograms = False
                if inspected.get("skip"):
                    return PlotResult(
                        filepath=save_path,
                        success=False,
                        error="Skipped by histogram inspector",
                    )
                title = inspected.get("title", title)
                histogram_color = inspected.get("color", histogram_color)
                n_bins = int(inspected.get("n_bins_override", n_bins))
                figsize = tuple(inspected.get("figsize", figsize))
                show_title = bool(inspected.get("show_title", show_title))
                show_statistics = bool(
                    inspected.get("show_statistics", show_statistics)
                )
                statistics_mode = str(
                    inspected.get("statistics_mode", statistics_mode)
                )
                x_label = str(inspected.get("x_label", x_label))
                y_label = str(inspected.get("y_label", y_label))
                xlim = inspected.get("xlim")
                ylim = inspected.get("ylim")
            fig, ax = plt.subplots(figsize=figsize)
            z_mean = float(np.mean(values))
            z_std = float(np.std(values))
            ax.hist(
                values,
                bins=n_bins,
                color=histogram_color,
                alpha=HISTOGRAM_ALPHA,
                edgecolor="black",
                linewidth=BAR_EDGE_LINEWIDTH,
            )
            if show_statistics and statistics_mode != "none":
                ax.axvline(z_mean, color="black", linestyle="-", linewidth=2.0)
                lower_err = z_mean - z_std
                upper_err = z_mean + z_std
                ax.axvspan(
                    lower_err,
                    upper_err,
                    color=self.style.std_region_color,
                    alpha=0.4,
                )
            histogram_font_scale = self.style.effective_histogram_font_scale
            ax.set_xlabel(
                x_label, fontsize=AXIS_LABEL_FONTSIZE * histogram_font_scale
            )
            ax.set_ylabel(
                y_label, fontsize=AXIS_LABEL_FONTSIZE * histogram_font_scale
            )
            if show_title:
                ax.set_title(
                    title,
                    fontsize=TITLE_FONTSIZE * histogram_font_scale,
                    fontweight="bold",
                )
            ax.set_xlabel(
                x_label, fontsize=AXIS_LABEL_FONTSIZE * histogram_font_scale
            )
            ax.set_ylabel(
                y_label, fontsize=AXIS_LABEL_FONTSIZE * histogram_font_scale
            )
            ax.set_xlabel(
                x_label, fontsize=AXIS_LABEL_FONTSIZE * histogram_font_scale
            )
            ax.set_ylabel(
                y_label, fontsize=AXIS_LABEL_FONTSIZE * histogram_font_scale
            )
            _style_histogram_axes(
                ax,
                spine_visible=self.style.spine_visible,
                tick_label_fontsize=AXIS_TICK_LABEL_FONTSIZE
                * histogram_font_scale,
            )
            if show_statistics and statistics_mode != "none":
                handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2.0,
                        label=f"Mean: {z_mean:.2f}",
                    ),
                    Patch(
                        facecolor="#D3D3D3",
                        edgecolor="none",
                        alpha=0.4,
                        label=f"±1 Std: {z_std:.2f}",
                    ),
                ]
                if statistics_mode == "full":
                    z_min = float(np.min(values))
                    z_max = float(np.max(values))
                    ax.axvline(
                        z_min, color="green", linestyle=":", linewidth=1.5
                    )
                    ax.axvline(
                        z_max, color="purple", linestyle=":", linewidth=1.5
                    )
                    handles.insert(
                        1,
                        Line2D(
                            [0],
                            [0],
                            color="green",
                            linestyle=":",
                            lw=1.5,
                            label=f"Min: {z_min:.2f}",
                        ),
                    )
                    handles.insert(
                        2,
                        Line2D(
                            [0],
                            [0],
                            color="purple",
                            linestyle=":",
                            lw=1.5,
                            label=f"Max: {z_max:.2f}",
                        ),
                    )
                ax.legend(
                    handles=handles,
                    loc="upper right",
                    fontsize=LEGEND_FONTSIZE * histogram_font_scale,
                )
            HistogramPlotter._apply_axis_limits(ax, xlim=xlim, ylim=ylim)
            _apply_figure_layout(
                fig, left=0.18, right=0.97, bottom=0.18, top=0.97
            )
            fig.savefig(save_path, dpi=self.style.dpi)
            plt.close(fig)
            logger.debug(f"Saved arithmetic histogram: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save arithmetic histogram: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))
