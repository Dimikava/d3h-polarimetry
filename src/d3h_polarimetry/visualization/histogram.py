"""Specialized visualization plotter implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from .common import (
    AXIS_LABEL_FONTSIZE,
    AXIS_TICK_LABEL_FONTSIZE,
    BAR_EDGE_LINEWIDTH,
    HISTOGRAM_ALPHA,
    HISTOGRAM_DUAL_PANEL_FIGSIZE,
    HISTOGRAM_SINGLE_PANEL_FIGSIZE,
    LEGEND_FONTSIZE,
    MAJOR_X_TICKS,
    TITLE_FONTSIZE,
    FigureStyle,
    PlotResult,
    _apply_figure_layout,
    _format_histogram_tick_label,
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


class HistogramPlotter:
    """
    Generates publication-grade orientation histograms.

    Creates dual-panel histograms showing FFT and curve-fit orientation
    distributions with statistical annotations.

    Attributes:
        style: Figure styling configuration.
    """

    def __init__(self, style: Optional[FigureStyle] = None) -> None:
        """
        Initialize the histogram plotter.

        Args:
            style: Styling configuration. Uses defaults if not provided.
        """
        self.style = style or FigureStyle()
        logger.debug("HistogramPlotter initialized")

    @staticmethod
    def _apply_axis_limits(
        ax: plt.Axes,
        *,
        xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> None:
        """Apply optional axis limits returned by the histogram inspector."""
        if xlim is not None and any((v is not None for v in xlim)):
            ax.set_xlim(left=xlim[0], right=xlim[1])
        if ylim is not None and any((v is not None for v in ylim)):
            ax.set_ylim(bottom=ylim[0], top=ylim[1])

    def _inspect_histogram_options(
        self,
        *,
        title: str,
        data: Sequence[float],
        count_total: int,
        color: str,
        bin_fraction: float,
        n_bins_override: Optional[int],
        figsize: Tuple[float, float],
        show_title: bool,
        show_statistics: bool,
        statistics_mode: str,
        x_label: str,
        y_label: str,
    ) -> Optional[Dict[str, Any]]:
        """Return user-edited histogram options when interactive inspection is enabled."""
        if not self.style.inspect_histograms:
            return None
        try:
            from d3h_polarimetry.gui import inspect_histogram_options

            def _render_preview(
                ax: plt.Axes,
                *,
                title: str,
                data: Sequence[float],
                color: str,
                n_bins: int,
                show_title: bool,
                show_statistics: bool,
                statistics_mode: str,
                x_label: str,
                y_label: str,
                xlim: Optional[Tuple[Optional[float], Optional[float]]],
                ylim: Optional[Tuple[Optional[float], Optional[float]]],
            ) -> None:
                font_scale = self.style.effective_histogram_font_scale
                self._style_subplot(
                    ax,
                    [float(v) for v in data if np.isfinite(v)],
                    color,
                    title,
                    n_bins,
                    show_title=show_title,
                    show_statistics=show_statistics,
                    statistics_mode=statistics_mode,
                    x_label=x_label,
                    y_label=y_label,
                    font_scale=font_scale,
                )
                self._apply_axis_limits(ax, xlim=xlim, ylim=ylim)

            return inspect_histogram_options(
                title=title,
                data=data,
                count_total=count_total,
                color=color,
                bin_fraction=bin_fraction,
                n_bins_override=n_bins_override,
                figsize=figsize,
                show_title=show_title,
                show_statistics=show_statistics,
                statistics_mode=statistics_mode,
                x_label=x_label,
                y_label=y_label,
                preview_renderer=_render_preview,
            )
        except ImportError as exc:
            raise RuntimeError(
                "--inspect-histograms requires an interactive GUI-capable matplotlib/Tk environment"
            ) from exc

    def save_histogram(
        self,
        save_path: Union[str, Path],
        title_prefix: str,
        fft_data: Sequence[float],
        curve_data: Sequence[float],
        count_total: int,
        bin_fraction: float = 0.1,
        figsize: Tuple[float, float] = HISTOGRAM_DUAL_PANEL_FIGSIZE,
        show_titles: bool = False,
        show_statistics: bool = True,
        statistics_mode: str = "full",
        x_label: str = "Armchair orientation θ (°)",
        y_label: str = "Number of pixels",
    ) -> PlotResult:
        """
        Save a dual-panel histogram of orientation distributions.

        Args:
            save_path: Output file path.
            title_prefix: Prefix for subplot titles.
            fft_data: FFT orientation values (degrees).
            curve_data: Curve fit orientation values (degrees).
            count_total: Total number of analysis windows (used for bin calculation).
            bin_fraction: Fraction of count_total to use as number of bins.
            figsize: Figure dimensions.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        try:
            n_bins = max(1, int(count_total * bin_fraction))
            font_scale = self.style.effective_histogram_font_scale
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            self._style_subplot(
                axes[0],
                list(fft_data) if fft_data else [],
                self.style.histogram_fft_color,
                f"{title_prefix} - FFT Orientation Distribution",
                n_bins,
                show_title=show_titles,
                show_statistics=show_statistics,
                x_label=x_label,
                y_label=y_label,
                font_scale=font_scale,
            )
            self._style_subplot(
                axes[1],
                list(curve_data) if curve_data else [],
                self.style.histogram_curve_color,
                f"{title_prefix} - Curve Fit Orientation Distribution",
                n_bins,
                show_title=show_titles,
                show_statistics=show_statistics,
                x_label=x_label,
                y_label=y_label,
                font_scale=font_scale,
            )
            if not fft_data and (not curve_data):
                for ax, label in ((axes[0], "FFT"), (axes[1], "Curve Fit")):
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        f"No threshold-passing {label} data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=self.style.label_fontsize * font_scale,
                    )
            _apply_figure_layout(
                fig, left=0.16, right=0.97, bottom=0.08, top=0.97, hspace=0.35
            )
            fig.savefig(save_path, dpi=self.style.dpi)
            plt.close(fig)
            logger.debug(f"Saved histogram: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save histogram: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def save_single_histogram(
        self,
        save_path: Union[str, Path],
        title: str,
        data: Sequence[float],
        count_total: int,
        *,
        color: str,
        bin_fraction: float = 0.1,
        n_bins_override: Optional[int] = None,
        use_auto_bins: bool = False,
        quantize_step: Optional[float] = None,
        xticks_override: Optional[Sequence[float]] = None,
        figsize: Tuple[float, float] = HISTOGRAM_SINGLE_PANEL_FIGSIZE,
        show_title: bool = False,
        show_statistics: bool = True,
        statistics_mode: str = "full",
        x_label: str = "Armchair orientation θ (°)",
        y_label: str = "Number of pixels",
        font_scale: float = 1.0,
    ) -> PlotResult:
        """Save a single-method histogram."""
        save_path = _prepare_save_path(save_path)
        try:
            xlim: Optional[Tuple[Optional[float], Optional[float]]] = None
            ylim: Optional[Tuple[Optional[float], Optional[float]]] = None
            inspected = self._inspect_histogram_options(
                title=title,
                data=data,
                count_total=count_total,
                color=color,
                bin_fraction=bin_fraction,
                n_bins_override=n_bins_override,
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
                color = inspected.get("color", color)
                bin_fraction = float(
                    inspected.get("bin_fraction", bin_fraction)
                )
                n_bins_override = inspected.get(
                    "n_bins_override", n_bins_override
                )
                if inspected.get("bins_changed"):
                    xticks_override = None
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
            font_scale *= self.style.effective_histogram_font_scale
            resolved_statistics_mode = self._resolve_statistics_mode(
                show_statistics=show_statistics,
                statistics_mode=statistics_mode,
            )
            n_bins = None
            if not use_auto_bins or n_bins_override is not None:
                n_bins = (
                    max(1, int(n_bins_override))
                    if n_bins_override is not None
                    else max(1, int(count_total * bin_fraction))
                )
            fig, ax = plt.subplots(figsize=figsize)
            if data:
                data_list = list(data)
                if quantize_step is not None and quantize_step > 0:
                    quantized = (
                        np.round(
                            np.asarray(data_list, dtype=float) / quantize_step
                        )
                        * quantize_step
                    )
                    centers, counts = np.unique(quantized, return_counts=True)
                    ax.bar(
                        centers,
                        counts,
                        width=quantize_step * 0.9,
                        color=color,
                        edgecolor="black",
                        linewidth=BAR_EDGE_LINEWIDTH,
                        alpha=HISTOGRAM_ALPHA,
                        align="center",
                    )
                    self._apply_histogram_statistics(
                        ax, quantized.tolist(), resolved_statistics_mode
                    )
                    if show_title:
                        ax.set_title(
                            title,
                            fontsize=TITLE_FONTSIZE * font_scale,
                            fontweight="bold",
                        )
                    ax.set_xlabel(
                        x_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale
                    )
                    ax.set_ylabel(
                        y_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale
                    )
                    self._configure_histogram_axes(
                        ax,
                        use_default_x_locator=xticks_override is None,
                        font_scale=font_scale,
                    )
                    self._apply_histogram_legend(
                        ax,
                        quantized.tolist(),
                        resolved_statistics_mode,
                        font_scale=font_scale,
                    )
                    if xticks_override is not None:
                        ax.set_xticks(list(xticks_override))
                        ax.xaxis.set_major_formatter(
                            FuncFormatter(_format_histogram_tick_label)
                        )
                    else:
                        ax.set_xticks(centers.tolist())
                    self._finalize_histogram_ticks(
                        ax, use_default_x_locator=xticks_override is None
                    )
                elif xticks_override is not None:
                    tick_centers = np.asarray(
                        list(xticks_override), dtype=float
                    )
                    tick_centers = tick_centers[np.isfinite(tick_centers)]
                    if tick_centers.size >= 2:
                        tick_centers = np.sort(tick_centers)
                        midpoints = (
                            tick_centers[:-1] + tick_centers[1:]
                        ) / 2.0
                        left_edge = tick_centers[0] - (
                            midpoints[0] - tick_centers[0]
                        )
                        right_edge = tick_centers[-1] + (
                            tick_centers[-1] - midpoints[-1]
                        )
                        bin_edges = np.concatenate(
                            ([left_edge], midpoints, [right_edge])
                        )
                        ax.hist(
                            data_list,
                            bins=bin_edges,
                            alpha=HISTOGRAM_ALPHA,
                            color=color,
                            edgecolor="black",
                            linewidth=BAR_EDGE_LINEWIDTH,
                        )
                        self._apply_histogram_statistics(
                            ax, data_list, resolved_statistics_mode
                        )
                        self._apply_histogram_legend(
                            ax,
                            data_list,
                            resolved_statistics_mode,
                            font_scale=font_scale,
                        )
                        if show_title:
                            ax.set_title(
                                title,
                                fontsize=TITLE_FONTSIZE * font_scale,
                                fontweight="bold",
                            )
                        ax.set_xlabel(
                            x_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale
                        )
                        ax.set_ylabel(
                            y_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale
                        )
                        self._configure_histogram_axes(
                            ax,
                            use_default_x_locator=False,
                            font_scale=font_scale,
                        )
                        ax.set_xticks(tick_centers.tolist())
                        ax.xaxis.set_major_formatter(
                            FuncFormatter(_format_histogram_tick_label)
                        )
                        ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
                        self._finalize_histogram_ticks(
                            ax,
                            use_default_x_locator=False,
                            font_scale=font_scale,
                        )
                    else:
                        self._style_subplot(
                            ax,
                            data_list,
                            color,
                            title,
                            n_bins,
                            show_title=show_title,
                            show_statistics=show_statistics,
                            statistics_mode=resolved_statistics_mode,
                            x_label=x_label,
                            y_label=y_label,
                            font_scale=font_scale,
                        )
                        ax.set_xticks(tick_centers.tolist())
                        ax.xaxis.set_major_formatter(
                            FuncFormatter(_format_histogram_tick_label)
                        )
                        self._finalize_histogram_ticks(
                            ax,
                            use_default_x_locator=False,
                            font_scale=font_scale,
                        )
                elif use_auto_bins and n_bins_override is None:
                    ax.hist(
                        data_list,
                        bins="auto",
                        alpha=HISTOGRAM_ALPHA,
                        color=color,
                        edgecolor="black",
                        linewidth=BAR_EDGE_LINEWIDTH,
                    )
                    self._apply_histogram_statistics(
                        ax, data_list, resolved_statistics_mode
                    )
                    if show_title:
                        ax.set_title(
                            title,
                            fontsize=TITLE_FONTSIZE * font_scale,
                            fontweight="bold",
                        )
                    ax.set_xlabel(
                        x_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale
                    )
                    ax.set_ylabel(
                        y_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale
                    )
                    self._configure_histogram_axes(
                        ax, use_default_x_locator=True, font_scale=font_scale
                    )
                    self._apply_histogram_legend(
                        ax,
                        data_list,
                        resolved_statistics_mode,
                        font_scale=font_scale,
                    )
                    self._finalize_histogram_ticks(
                        ax, use_default_x_locator=True, font_scale=font_scale
                    )
                else:
                    self._style_subplot(
                        ax,
                        data_list,
                        color,
                        title,
                        n_bins,
                        show_title=show_title,
                        show_statistics=show_statistics,
                        statistics_mode=resolved_statistics_mode,
                        x_label=x_label,
                        y_label=y_label,
                        font_scale=font_scale,
                    )
            else:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "No threshold-passing data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=self.style.label_fontsize * font_scale,
                )
            self._apply_axis_limits(ax, xlim=xlim, ylim=ylim)
            _apply_figure_layout(
                fig, left=0.18, right=0.97, bottom=0.18, top=0.97
            )
            fig.savefig(save_path, dpi=self.style.dpi)
            plt.close(fig)
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save single histogram: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def _style_subplot(
        self,
        ax: plt.Axes,
        data: List[float],
        color: str,
        title: str,
        n_bins: int,
        show_title: bool = False,
        show_statistics: bool = True,
        statistics_mode: str = "full",
        x_label: str = "Armchair orientation θ (°)",
        y_label: str = "Number of pixels",
        font_scale: float = 1.0,
    ) -> None:
        """Apply publication styling to a histogram subplot."""
        if not data:
            ax.axis("off")
            return
        resolved_statistics_mode = self._resolve_statistics_mode(
            show_statistics=show_statistics, statistics_mode=statistics_mode
        )
        ax.hist(
            data,
            bins=n_bins,
            alpha=HISTOGRAM_ALPHA,
            color=color,
            edgecolor="black",
            linewidth=BAR_EDGE_LINEWIDTH,
        )
        self._apply_histogram_statistics(ax, data, resolved_statistics_mode)
        if show_title:
            ax.set_title(
                title, fontsize=TITLE_FONTSIZE * font_scale, fontweight="bold"
            )
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale)
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE * font_scale)
        self._configure_histogram_axes(ax, font_scale=font_scale)
        self._apply_histogram_legend(
            ax, data, resolved_statistics_mode, font_scale=font_scale
        )
        self._finalize_histogram_ticks(ax, font_scale=font_scale)

    @staticmethod
    def _resolve_statistics_mode(
        *, show_statistics: bool, statistics_mode: str
    ) -> str:
        if not show_statistics:
            return "none"
        if statistics_mode not in {"none", "mean_std", "full"}:
            raise ValueError(
                "statistics_mode must be 'none', 'mean_std', or 'full'"
            )
        return statistics_mode

    def _apply_histogram_statistics(
        self, ax: plt.Axes, data: Sequence[float], statistics_mode: str
    ) -> None:
        if statistics_mode == "none" or not data:
            return
        values = np.asarray(list(data), dtype=float)
        d_mean = float(np.mean(values))
        d_std = float(np.std(values))
        ax.axvline(
            d_mean,
            color=self.style.mean_line_color,
            linestyle="-",
            linewidth=1.5,
        )
        ax.axvspan(
            d_mean - d_std,
            d_mean + d_std,
            color=self.style.std_region_color,
            alpha=0.4,
        )
        if statistics_mode == "full":
            d_min = float(np.min(values))
            d_max = float(np.max(values))
            ax.axvline(d_min, color="green", linestyle=":", linewidth=1.5)
            ax.axvline(d_max, color="purple", linestyle=":", linewidth=1.5)

    def _apply_histogram_legend(
        self,
        ax: plt.Axes,
        data: Sequence[float],
        statistics_mode: str,
        *,
        font_scale: float = 1.0,
    ) -> None:
        if statistics_mode == "none" or not data:
            return
        values = np.asarray(list(data), dtype=float)
        d_mean = float(np.mean(values))
        d_std = float(np.std(values))
        handles: List[Line2D | Patch] = [
            Line2D(
                [0], [0], color="black", lw=1.5, label=f"Mean: {d_mean:.2f}°"
            ),
            Patch(
                facecolor="#D3D3D3",
                edgecolor="none",
                alpha=0.4,
                label=f"±1 Std: {d_std:.2f}°",
            ),
        ]
        if statistics_mode == "full":
            d_min = float(np.min(values))
            d_max = float(np.max(values))
            handles.insert(
                1,
                Line2D(
                    [0],
                    [0],
                    color="green",
                    linestyle=":",
                    lw=1.5,
                    label=f"Min: {d_min:.2f}°",
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
                    label=f"Max: {d_max:.2f}°",
                ),
            )
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=LEGEND_FONTSIZE * font_scale,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
        )

    def _configure_histogram_axes(
        self,
        ax: plt.Axes,
        *,
        use_default_x_locator: bool = True,
        font_scale: float = 1.0,
    ) -> None:
        """Apply publication-ready axis styling for histogram panels."""
        _style_histogram_axes(
            ax,
            spine_visible=self.style.spine_visible,
            use_default_x_locator=use_default_x_locator,
            tick_label_fontsize=AXIS_TICK_LABEL_FONTSIZE * font_scale,
        )

    def _finalize_histogram_ticks(
        self,
        ax: plt.Axes,
        *,
        use_default_x_locator: bool = True,
        font_scale: float = 1.0,
    ) -> None:
        """Finalize tick density after plotting and explicit tick overrides."""
        self._configure_histogram_axes(
            ax,
            use_default_x_locator=use_default_x_locator,
            font_scale=font_scale,
        )
        if not use_default_x_locator:
            tick_values = ax.get_xticks()
            if len(tick_values) > MAJOR_X_TICKS + 1:
                ax.set_xticks(tick_values[: MAJOR_X_TICKS + 1])
