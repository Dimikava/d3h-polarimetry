"""Specialized visualization plotter implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from .common import (
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    MAP_FIGSIZE,
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


class OrientationMapPlotter:
    """
    Generates 2D orientation heatmaps with R² masking.

    Creates color-coded orientation maps where low-quality pixels
    (below R² threshold) are masked.

    Attributes:
        style: Figure styling configuration.
    """

    def __init__(self, style: Optional[FigureStyle] = None) -> None:
        """
        Initialize the orientation map plotter.

        Args:
            style: Styling configuration. Uses defaults if not provided.
        """
        self.style = style or FigureStyle()
        logger.debug("OrientationMapPlotter initialized")

    @staticmethod
    def _block_average_orientation_vectors(
        angle_map: np.ndarray,
        valid_mask: np.ndarray,
        *,
        block_size: int = 8,
        min_valid_fraction: float = 0.5,
        arrow_length_fraction: float = 0.35,
        orientation_mode: str = "nematic",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reduce an orientation map to one vector per block."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if orientation_mode not in {"nematic", "directed"}:
            raise ValueError(
                "orientation_mode must be 'nematic' or 'directed'"
            )
        angles = np.asarray(angle_map, dtype=float)
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(angles)
        height, width = angles.shape
        min_valid_count = max(
            1, int(np.ceil(block_size * block_size * min_valid_fraction))
        )
        arrow_length = float(block_size) * float(arrow_length_fraction)
        xs: List[float] = []
        ys: List[float] = []
        us: List[float] = []
        vs: List[float] = []
        for row_start in range(0, height, block_size):
            row_end = min(height, row_start + block_size)
            for col_start in range(0, width, block_size):
                col_end = min(width, col_start + block_size)
                block_valid = valid[row_start:row_end, col_start:col_end]
                valid_count = int(np.count_nonzero(block_valid))
                if valid_count < min_valid_count:
                    continue
                block_angles = np.radians(
                    angles[row_start:row_end, col_start:col_end][block_valid]
                )
                if orientation_mode == "directed":
                    mean_cos = float(np.mean(np.cos(block_angles)))
                    mean_sin = float(np.mean(np.sin(block_angles)))
                    coherence = float(np.hypot(mean_cos, mean_sin))
                    theta = np.arctan2(mean_sin, mean_cos)
                else:
                    mean_cos2 = float(np.mean(np.cos(2.0 * block_angles)))
                    mean_sin2 = float(np.mean(np.sin(2.0 * block_angles)))
                    coherence = float(np.hypot(mean_cos2, mean_sin2))
                    theta = 0.5 * np.arctan2(mean_sin2, mean_cos2)
                if coherence <= 1e-09:
                    continue
                xs.append((col_start + col_end - 1) / 2.0)
                ys.append((row_start + row_end - 1) / 2.0)
                us.append(arrow_length * float(np.cos(theta)))
                vs.append(arrow_length * float(np.sin(theta)))
        return (
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(us, dtype=float),
            np.asarray(vs, dtype=float),
        )

    def save_orientation_map(
        self,
        save_path: Union[str, Path],
        title: str,
        angle_map: np.ndarray,
        r2_map: np.ndarray,
        threshold: float = 0.8,
        figsize: Tuple[float, float] = MAP_FIGSIZE,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colormap: Optional[str] = None,
        cbar_label: str = "Orientation (Degrees)",
    ) -> PlotResult:
        """
        Save a 2D orientation map with R² masking.

        Args:
            save_path: Output file path.
            title: Plot title.
            angle_map: 2D array of orientation angles (degrees).
            r2_map: 2D array of R² values for quality masking.
            threshold: R² threshold below which pixels are masked.
            figsize: Figure dimensions.

        Returns:
            PlotResult with success status and filepath.
        """
        save_path = _prepare_save_path(save_path)
        try:
            r2_clean = np.nan_to_num(r2_map, nan=0.0)
            mask = r2_clean < threshold
            masked_angles = np.ma.masked_where(mask, angle_map)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor(self.style.background_color)
            cmap = plt.get_cmap(colormap or self.style.colormap).copy()
            cmap.set_bad(self.style.background_color)
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
            _apply_figure_layout(
                fig, left=0.08, right=0.92, bottom=0.08, top=0.94
            )
            fig.savefig(save_path, dpi=self.style.dpi)
            plt.close(fig)
            logger.debug(f"Saved orientation map: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save orientation map: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))

    def save_orientation_vector_field(
        self,
        save_path: Union[str, Path],
        title: str,
        angle_map: np.ndarray,
        r2_map: np.ndarray,
        threshold: float = 0.8,
        *,
        background_image: Optional[np.ndarray] = None,
        background_cmap: str = "gray",
        background_vmin: Optional[float] = None,
        background_vmax: Optional[float] = None,
        background_cbar_label: Optional[str] = None,
        block_size: int = 8,
        min_valid_fraction: float = 0.5,
        arrow_length_fraction: float = 0.35,
        orientation_mode: str = "nematic",
        arrow_color: str = "white",
        arrow_outline: bool = False,
        arrow_outline_color: str = "black",
        arrow_width: float = 0.004,
        arrow_outline_width: float = 0.007,
        headwidth: float = 3.5,
        headlength: float = 4.5,
        headaxislength: float = 4.0,
        vectors_only: bool = False,
        figsize: Tuple[float, float] = MAP_FIGSIZE,
    ) -> PlotResult:
        """Save a block-averaged vector field over an image or colormap background."""
        save_path = _prepare_save_path(save_path)
        try:
            angles = np.asarray(angle_map, dtype=float)
            r2_clean = np.nan_to_num(r2_map, nan=0.0)
            valid_mask = (r2_clean >= threshold) & np.isfinite(angles)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor(
                "black" if vectors_only else self.style.background_color
            )
            if vectors_only:
                im = None
            elif background_image is None:
                background = np.ma.masked_where(~valid_mask, angles)
                cmap = plt.get_cmap(self.style.colormap).copy()
                cmap.set_bad(self.style.background_color)
                im = ax.imshow(
                    background,
                    cmap=cmap,
                    origin="upper",
                    vmin=background_vmin,
                    vmax=background_vmax,
                )
            else:
                background_array = np.asarray(background_image, dtype=float)
                if background_cbar_label is None:
                    im = ax.imshow(
                        background_array,
                        cmap=background_cmap,
                        origin="upper",
                        vmin=background_vmin,
                        vmax=background_vmax,
                    )
                else:
                    background = np.ma.masked_where(
                        ~valid_mask, background_array
                    )
                    cmap = plt.get_cmap(background_cmap).copy()
                    cmap.set_bad(self.style.background_color)
                    im = ax.imshow(
                        background,
                        cmap=cmap,
                        origin="upper",
                        vmin=background_vmin,
                        vmax=background_vmax,
                    )
            if (
                background_cbar_label is not None
                and (not vectors_only)
                and (im is not None)
            ):
                cbar = plt.colorbar(
                    im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
                )
                _style_colorbar(
                    cbar,
                    label=background_cbar_label,
                    font_scale=self.style.map_font_scale,
                    nbins=9
                    if background_cbar_label == "Orientation (Degrees)"
                    else None,
                )
            xs, ys, us, vs = self._block_average_orientation_vectors(
                angles,
                valid_mask,
                block_size=block_size,
                min_valid_fraction=min_valid_fraction,
                arrow_length_fraction=arrow_length_fraction,
                orientation_mode=orientation_mode,
            )
            if xs.size:

                def _draw_quiver(
                    color: str, width: float, zorder: float
                ) -> None:
                    ax.quiver(
                        xs,
                        ys,
                        us,
                        vs,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        color=color,
                        pivot="mid",
                        width=width,
                        headwidth=headwidth,
                        headlength=headlength,
                        headaxislength=headaxislength,
                        zorder=zorder,
                    )

                if arrow_outline:
                    _draw_quiver(arrow_outline_color, arrow_outline_width, 3.0)
                _draw_quiver(arrow_color, arrow_width, 4.0)
            ax.set_title(
                title,
                fontsize=self.style.effective_map_title_fontsize,
                fontweight="bold",
            )
            ax.axis("off")
            _apply_figure_layout(
                fig, left=0.08, right=0.92, bottom=0.08, top=0.94
            )
            fig.savefig(save_path, dpi=self.style.dpi)
            plt.close(fig)
            logger.debug(f"Saved orientation vector field: {save_path}")
            return PlotResult(filepath=save_path, success=True)
        except Exception as e:
            logger.error(f"Failed to save orientation vector field: {e}")
            return PlotResult(filepath=save_path, success=False, error=str(e))
