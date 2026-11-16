"""Specialized visualization plotter implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from .common import (
    POLAR_FIGSIZE,
    FigureStyle,
    PlotResult,
    _prepare_polar_series,
    _prepare_save_path,
    _style_polar_axes_shared,
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


class PolarPlotter:
    """
    Generates polar plots for polarization analysis results.

    Polar plots display angular intensity data with fitted curves,
    ideal for visualizing FFT and curve-fit analysis results.

    Attributes:
        style: Figure styling configuration.

    Example:
        >>> plotter = PolarPlotter()
        >>> result = plotter.save_polar_plot(
        ...     save_path=Path("output/roi_1_fft.png"),
        ...     title="ROI 1 - FFT Fit",
        ...     signal=signal_data,
        ...     angles=angle_data,
        ...     analysis_result=fft_result,
        ...     fit_key='fft'
        ... )
    """

    def __init__(self, style: Optional[FigureStyle] = None) -> None:
        """
        Initialize the polar plotter.

        Args:
            style: Styling configuration. Uses defaults if not provided.
        """
        self.style = style or FigureStyle()
        logger.debug("PolarPlotter initialized")

    def save_polar_plot(
        self,
        save_path: Union[str, Path],
        title: str,
        signal: np.ndarray,
        angles: np.ndarray,
        analysis_result: Dict[str, Any],
        fit_key: str = "fft",
        fit_color: Optional[str] = None,
        fit_label: str = "Fit",
        figsize: Tuple[float, float] = POLAR_FIGSIZE,
        show_title: bool = True,
        show_legend: bool = True,
        show_radial_ticks: bool = False,
        axes_rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> PlotResult:
        """
        Save a polar plot showing data and fitted curve.

        Args:
            save_path: Output file path.
            title: Plot title.
            signal: 1D array of intensity values.
            angles: 1D array of angles in radians.
            analysis_result: Dictionary containing fit results with 'prediction' key.
            fit_key: Key in analysis_result containing fit data ('fft' or 'fit').
            fit_color: Color for fit line. Uses style defaults if None.
            fit_label: Legend label for fit line.
            figsize: Figure dimensions.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        try:
            if fit_color is None:
                fit_color = (
                    self.style.fft_color
                    if fit_key == "fft"
                    else self.style.curve_color
                )
            plot_signal = analysis_result.get("signal", signal)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="polar")
            if axes_rect is not None:
                ax.set_position(axes_rect)
            self._style_polar_axes(ax, show_radial_ticks=show_radial_ticks)
            scatter_angles, scatter_signal = _prepare_polar_series(
                angles,
                plot_signal,
                half_turn=self.style.polar_half_turn,
                close_loop=False,
            )
            ax.scatter(
                scatter_angles,
                scatter_signal,
                s=20,
                label="Data",
                color=self.style.data_color,
                alpha=0.6,
            )
            fit_data = analysis_result.get(fit_key, {})
            if isinstance(fit_data, dict) and "prediction" in fit_data:
                pred = fit_data["prediction"]
                plot_angles, plot_pred = _prepare_polar_series(
                    angles,
                    pred,
                    half_turn=self.style.polar_half_turn,
                    close_loop=True,
                )
                ax.plot(
                    plot_angles,
                    plot_pred,
                    color=fit_color,
                    linewidth=2.5,
                    label=fit_label,
                )
            if show_title:
                ax.set_title(
                    title,
                    fontsize=self.style.effective_polar_title_fontsize,
                    fontweight="bold",
                    pad=18,
                )
            if show_legend:
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=(1.0, 0.0),
                    fontsize=self.style.effective_polar_legend_fontsize,
                )
            if axes_rect is None:
                plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.debug(f"Saved polar plot: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save polar plot: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def _style_polar_axes(
        self, ax: plt.Axes, *, show_radial_ticks: bool
    ) -> None:
        """Increase polar axis visibility for exported figures."""
        _style_polar_axes_shared(
            ax,
            show_radial_ticks=show_radial_ticks,
            tick_label_fontsize=self.style.effective_polar_tick_label_fontsize,
            theta_tick_pad=self.style.polar_theta_tick_pad,
            radial_tick_pad=self.style.polar_radial_tick_pad,
            half_turn=self.style.polar_half_turn,
        )
