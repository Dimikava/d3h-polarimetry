"""Specialized visualization plotter implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from .common import (
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    FigureStyle,
    PlotResult,
    _apply_figure_layout,
    _prepare_save_path,
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


class OverlayPlotter:
    """
    Generates overlay images showing ROIs and POIs on mean images.

    Attributes:
        style: Figure styling configuration.
    """

    def __init__(self, style: Optional[FigureStyle] = None) -> None:
        """
        Initialize the overlay plotter.

        Args:
            style: Styling configuration. Uses defaults if not provided.
        """
        self.style = style or FigureStyle()
        logger.debug("OverlayPlotter initialized")

    def save_roi_overlay(
        self,
        save_path: Union[str, Path],
        image: np.ndarray,
        roi_patches: List[Tuple[int, int, int, int]],
        poi_patches: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> PlotResult:
        """
        Save mean image with ROI and POI overlays.

        Args:
            save_path: Output file path.
            image: 2D grayscale image array.
            roi_patches: List of ROI bounds as (r_min, r_max, c_min, c_max).
            poi_patches: Optional list of POI bounds.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        try:
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            for i, (r1, r2, c1, c2) in enumerate(roi_patches):
                rect = plt.Rectangle(
                    (c1, r1),
                    c2 - c1,
                    r2 - r1,
                    linewidth=1.5,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    c1,
                    r1 - 5,
                    f"{i + 1}",
                    color="r",
                    fontsize=self.style.annotation_fontsize,
                    fontweight="bold",
                )
            if poi_patches:
                for pr1, pr2, pc1, pc2 in poi_patches:
                    rect_poi = plt.Rectangle(
                        (pc1, pr1),
                        pc2 - pc1,
                        pr2 - pr1,
                        linewidth=1.0,
                        edgecolor="yellow",
                        facecolor="none",
                    )
                    ax.add_patch(rect_poi)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.debug(f"Saved ROI overlay: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save ROI overlay: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def save_zoomed_roi(
        self,
        save_path: Union[str, Path],
        roi_image: np.ndarray,
        poi_points: List[Tuple[int, int]],
        circle_radius: float = 3,
        figsize: Tuple[float, float] = (6, 6),
    ) -> PlotResult:
        """
        Save zoomed ROI image with filled POI circles.

        Args:
            save_path: Output file path.
            roi_image: 2D cropped ROI image.
            poi_points: List of POI coordinates as (row, col) relative to crop.
            circle_radius: Radius of POI markers.
            figsize: Figure dimensions.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(roi_image, cmap="gray")
            ax.axis("off")
            for i, (pr, pc) in enumerate(poi_points):
                circ = plt.Circle(
                    (pc, pr), radius=circle_radius, color="yellow", fill=True
                )
                ax.add_patch(circ)
                ax.text(
                    pc + 4,
                    pr + 4,
                    f"{i + 1}",
                    color="yellow",
                    fontsize=self.style.annotation_fontsize,
                    fontweight="bold",
                )
            fig.savefig(save_path, bbox_inches="tight", dpi=150, pad_inches=0)
            plt.close(fig)
            logger.debug(f"Saved zoomed ROI: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save zoomed ROI: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def save_roi_overlay_special(
        self,
        save_path: Union[str, Path],
        image: np.ndarray,
        roi_patches: List[Tuple[int, int, int, int]],
        roi_names: List[str],
        poi_info: List[Tuple[int, int, int, str]],
        scale_bar_params: Optional[Dict] = None,
        figsize: Optional[Tuple[float, float]] = None,
        roi_color: str = "white",
        roi_linewidth: float = 1.0,
        roi_label_fontsize: Optional[int] = None,
        roi_label_bbox: bool = True,
    ) -> PlotResult:
        """
        Save mean image with letter-named ROI rectangles and numbered POI squares.

        Args:
            save_path: Output file path.
            image: 2D grayscale image array.
            roi_patches: List of ROI bounds as (r_min, r_max, c_min, c_max).
            roi_names: List of ROI names (e.g. ['ROI-A', 'ROI-B']).
            poi_info: List of (global_r, global_c, poi_number, roi_name) for each POI.
            scale_bar_params: Optional dict with 'microns_per_pixel', 'bar_length_um'.
            figsize: Figure dimensions. Auto-calculated if None.
        """
        save_path = _prepare_save_path(save_path)
        try:
            if figsize is None:
                h, w = image.shape[:2]
                figsize = (max(8, w / 80), max(6, h / 80))
            image = np.asarray(image, dtype=float)
            image_min = float(np.min(image))
            image_max = float(np.max(image))
            if image_max > image_min:
                normalized_image = (image - image_min) / (
                    image_max - image_min
                )
            else:
                normalized_image = np.zeros_like(image, dtype=float)
            display_image = normalized_image * 6.0
            fig, ax = plt.subplots(figsize=figsize)
            special_mean_cmap = LinearSegmentedColormap.from_list(
                "special_mean_black_green_intense_green",
                [(0.0, "black"), (1.0 / 6.0, "#00a000"), (1.0, "#00ff00")],
                N=256,
            )
            im = ax.imshow(
                display_image,
                cmap=special_mean_cmap,
                origin="upper",
                vmin=0.0,
                vmax=6.0,
            )
            ax.axis("off")
            cbar = plt.colorbar(
                im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
            )
            _style_colorbar(
                cbar,
                label="Normalized Mean Intensity",
                font_scale=self.style.map_font_scale,
            )
            for (r1, r2, c1, c2), name in zip(roi_patches, roi_names):
                rect = plt.Rectangle(
                    (c1, r1),
                    c2 - c1,
                    r2 - r1,
                    linewidth=roi_linewidth,
                    edgecolor=roi_color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                text_kwargs = {}
                if roi_label_bbox:
                    text_kwargs["bbox"] = dict(
                        boxstyle="round,pad=0.2", facecolor="black", alpha=0.7
                    )
                ax.text(
                    c1,
                    r1 - 4,
                    name,
                    color=roi_color,
                    fontsize=roi_label_fontsize
                    or self.style.annotation_fontsize,
                    fontweight="bold",
                    **text_kwargs,
                )
            sq_half = 3
            for global_r, global_c, poi_num, _ in poi_info:
                rect_poi = plt.Rectangle(
                    (global_c - sq_half, global_r - sq_half),
                    sq_half * 2,
                    sq_half * 2,
                    linewidth=1.5,
                    edgecolor="yellow",
                    facecolor="yellow",
                    alpha=0.6,
                )
                ax.add_patch(rect_poi)
                ax.text(
                    global_c + sq_half + 2,
                    global_r + sq_half,
                    f"POI-{poi_num}",
                    color="yellow",
                    fontsize=self.style.annotation_fontsize,
                    fontweight="bold",
                )
            if scale_bar_params:
                self._draw_scale_bar_on_axes(ax, image.shape, scale_bar_params)
            fig.savefig(save_path, bbox_inches="tight", dpi=self.style.dpi)
            plt.close(fig)
            logger.debug(f"Saved special ROI overlay: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save special ROI overlay: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def save_orientation_map_with_overlay(
        self,
        save_path: Union[str, Path],
        title: str,
        angle_map: np.ndarray,
        r2_map: np.ndarray,
        threshold: float,
        roi_patches_local: Optional[List[Tuple[int, int, int, int]]] = None,
        roi_names: Optional[List[str]] = None,
        poi_info_local: Optional[List[Tuple[int, int, int, str]]] = None,
        scale_bar_params: Optional[Dict] = None,
        colormap: Optional[str] = None,
        cbar_label: str = "Orientation (Degrees)",
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        figsize: Tuple[float, float] = (10, 8),
        show_poi_overlay: bool = True,
    ) -> PlotResult:
        """
        Save orientation colormap with ROI/POI overlay composited on top.

        Args:
            save_path: Output file path.
            title: Plot title.
            angle_map: 2D array of angle values.
            r2_map: 2D array of R² values for masking.
            threshold: R² threshold.
            roi_patches_local: ROI bounds in map coordinate space.
            roi_names: ROI labels.
            poi_info_local: POI info in map coordinate space (r, c, number, name).
            scale_bar_params: Optional dict with 'microns_per_pixel', 'bar_length_um'.
            colormap: Colormap name override.
            cbar_label: Colorbar label.
            figsize: Figure dimensions.
        """
        save_path = _prepare_save_path(save_path)
        try:
            r2_clean = np.nan_to_num(r2_map, nan=0.0)
            mask = r2_clean < threshold
            masked_angles = np.ma.masked_where(mask, angle_map)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor(self.style.background_color)
            cmap = plt.get_cmap(colormap or self.style.colormap).copy()
            cmap = cmap.with_extremes(bad=self.style.background_color)
            im = ax.imshow(
                masked_angles, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax
            )
            cbar = plt.colorbar(
                im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
            )
            _style_colorbar(
                cbar,
                label=cbar_label,
                font_scale=self.style.map_font_scale,
                nbins=9 if cbar_label == "Orientation (Degrees)" else None,
            )
            ax.set_title(
                title,
                fontsize=self.style.effective_map_title_fontsize,
                fontweight="bold",
            )
            ax.axis("off")
            roi_colors = [
                "#FF4444",
                "#44FF44",
                "#4488FF",
                "#FFAA00",
                "#FF44FF",
                "#44FFFF",
                "#FF8844",
                "#88FF44",
            ]
            if roi_patches_local and roi_names:
                for i, ((r1, r2, c1, c2), name) in enumerate(
                    zip(roi_patches_local, roi_names)
                ):
                    color = roi_colors[i % len(roi_colors)]
                    rect = plt.Rectangle(
                        (c1, r1),
                        c2 - c1,
                        r2 - r1,
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor="none",
                        linestyle="--",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        c1,
                        r1 - 3,
                        name,
                        color=color,
                        fontsize=self.style.annotation_fontsize,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="black",
                            alpha=0.6,
                        ),
                    )
            sq_half = 2
            if show_poi_overlay and poi_info_local:
                for r, c, poi_num, _ in poi_info_local:
                    rect_poi = plt.Rectangle(
                        (c - sq_half, r - sq_half),
                        sq_half * 2,
                        sq_half * 2,
                        linewidth=1.0,
                        edgecolor="yellow",
                        facecolor="yellow",
                        alpha=0.5,
                    )
                    ax.add_patch(rect_poi)
                    ax.text(
                        c + sq_half + 1,
                        r + sq_half,
                        f"POI-{poi_num}",
                        color="yellow",
                        fontsize=self.style.annotation_fontsize,
                        fontweight="bold",
                    )
            if scale_bar_params:
                self._draw_scale_bar_on_axes(
                    ax, angle_map.shape, scale_bar_params
                )
            _apply_figure_layout(
                fig, left=0.08, right=0.92, bottom=0.08, top=0.94
            )
            fig.savefig(save_path, dpi=self.style.dpi)
            plt.close(fig)
            logger.debug(f"Saved orientation map with overlay: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save orientation map with overlay: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def _draw_scale_bar_on_axes(
        self, ax: plt.Axes, image_shape: Tuple[int, ...], params: Dict
    ) -> None:
        """Draw a scale bar annotation on matplotlib axes."""
        mpp = params.get("microns_per_pixel", 1.0)
        bar_um = params.get("bar_length_um", 10.0)
        bar_px = bar_um / mpp
        h = image_shape[0]
        w = image_shape[1] if len(image_shape) > 1 else image_shape[0]
        margin = 0.05
        x_end = w * (1 - margin)
        x_start = x_end - bar_px
        y_bar = h * (1 - margin)
        ax.plot([x_start, x_end], [y_bar, y_bar], color="white", linewidth=3)
        label = (
            f"{int(bar_um)} μm"
            if bar_um == int(bar_um)
            else f"{bar_um:.1f} μm"
        )
        label = (
            f"{int(bar_um)} μm"
            if bar_um == int(bar_um)
            else f"{bar_um:.1f} μm"
        )
        ax.text(
            (x_start + x_end) / 2,
            y_bar - h * 0.02,
            label,
            color="white",
            fontsize=self.style.annotation_fontsize,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )
