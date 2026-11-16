"""
TIFF Analyzer - GUI Components

Provides interactive GUI tools for microscopy image analysis,
including ROI (Region of Interest) selection.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np

logger = logging.getLogger(__name__)


def parse_histogram_inspection_options(
    *,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    bins: str,
    x_min: str,
    x_max: str,
    y_min: str,
    y_max: str,
    width: str,
    height: str,
    show_title: bool,
    show_statistics: bool,
    statistics_mode: str,
    default_bins: int,
    is_color_like,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate histogram inspector fields into a render-ready options dict."""
    try:
        bin_count = max(1, int(float(bins)))
        fig_width = float(width)
        fig_height = float(height)
        if fig_width <= 0.0 or fig_height <= 0.0:
            raise ValueError("figure width and height must be positive")

        def _float_or_none(text: str) -> Optional[float]:
            text = str(text).strip()
            if not text:
                return None
            return float(text)

        xlim = (_float_or_none(x_min), _float_or_none(x_max))
        ylim = (_float_or_none(y_min), _float_or_none(y_max))
        if (
            xlim[0] is not None
            and xlim[1] is not None
            and (xlim[0] >= xlim[1])
        ):
            raise ValueError("x min must be smaller than x max")
        if (
            ylim[0] is not None
            and ylim[1] is not None
            and (ylim[0] >= ylim[1])
        ):
            raise ValueError("y min must be smaller than y max")
        if not is_color_like(color):
            raise ValueError(f"invalid histogram color: {color}")
        if statistics_mode not in {"full", "mean_std", "none"}:
            raise ValueError(f"invalid statistics mode: {statistics_mode}")
        return (
            {
                "title": title,
                "color": color,
                "bins": bin_count,
                "show_title": bool(show_title),
                "show_statistics": bool(show_statistics),
                "statistics_mode": statistics_mode,
                "x_label": x_label,
                "y_label": y_label,
                "figsize": (fig_width, fig_height),
                "xlim": xlim,
                "ylim": ylim,
                "bins_changed": bin_count != int(default_bins),
            },
            None,
        )
    except (TypeError, ValueError) as exc:
        return (None, str(exc))


def inspect_histogram_options(
    *,
    title: str,
    data,
    count_total: int,
    color: str,
    bin_fraction: float,
    n_bins_override,
    figsize,
    show_title: bool,
    show_statistics: bool,
    statistics_mode: str,
    x_label: str,
    y_label: str,
    preview_renderer=None,
):
    """Open a small blocking histogram editor and return save options."""
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
        from matplotlib.colors import is_color_like
    except Exception as exc:
        raise ImportError("matplotlib Tk backend is unavailable") from exc
    values = [float(v) for v in data if np.isfinite(v)]
    root = tk.Tk()
    root.title("Histogram Inspector")
    result = {"action": "save"}
    control = tk.Frame(root)
    control.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    default_bins = (
        int(n_bins_override)
        if n_bins_override is not None
        else max(1, int(count_total * bin_fraction))
    )
    title_var = tk.StringVar(value=title)
    x_label_var = tk.StringVar(value=x_label)
    y_label_var = tk.StringVar(value=y_label)
    color_var = tk.StringVar(value=color)
    bins_var = tk.StringVar(value=str(default_bins))
    xmin_var = tk.StringVar(value="")
    xmax_var = tk.StringVar(value="")
    ymin_var = tk.StringVar(value="")
    ymax_var = tk.StringVar(value="")
    width_var = tk.StringVar(value=str(float(figsize[0])))
    height_var = tk.StringVar(value=str(float(figsize[1])))
    show_title_var = tk.BooleanVar(value=show_title)
    show_stats_var = tk.BooleanVar(value=show_statistics)
    stats_mode_var = tk.StringVar(value=statistics_mode)
    status_var = tk.StringVar(value="")

    def _entry(label: str, variable: tk.StringVar) -> None:
        tk.Label(control, text=label).pack(anchor="w")
        tk.Entry(control, textvariable=variable, width=18).pack(
            anchor="w", pady=(0, 5)
        )

    _entry("Title", title_var)
    _entry("X label", x_label_var)
    _entry("Y label", y_label_var)
    _entry("Color", color_var)
    tk.Label(control, text="Bins").pack(anchor="w")
    tk.Spinbox(
        control, from_=1, to=10000, textvariable=bins_var, width=18
    ).pack(anchor="w", pady=(0, 5))
    _entry("X min", xmin_var)
    _entry("X max", xmax_var)
    _entry("Y min", ymin_var)
    _entry("Y max", ymax_var)
    _entry("Width", width_var)
    _entry("Height", height_var)
    tk.Checkbutton(control, text="Show title", variable=show_title_var).pack(
        anchor="w"
    )
    tk.Checkbutton(
        control, text="Show statistics", variable=show_stats_var
    ).pack(anchor="w")
    tk.Label(control, text="Statistics mode").pack(anchor="w", pady=(6, 0))
    tk.OptionMenu(control, stats_mode_var, "full", "mean_std", "none").pack(
        anchor="w"
    )
    tk.Label(
        control,
        textvariable=status_var,
        fg="red",
        wraplength=160,
        justify="left",
    ).pack(anchor="w", pady=(6, 0))

    def _validated_options():
        options, error = parse_histogram_inspection_options(
            title=title_var.get(),
            x_label=x_label_var.get(),
            y_label=y_label_var.get(),
            color=color_var.get(),
            bins=bins_var.get(),
            x_min=xmin_var.get(),
            x_max=xmax_var.get(),
            y_min=ymin_var.get(),
            y_max=ymax_var.get(),
            width=width_var.get(),
            height=height_var.get(),
            show_title=show_title_var.get(),
            show_statistics=show_stats_var.get(),
            statistics_mode=stats_mode_var.get(),
            default_bins=default_bins,
            is_color_like=is_color_like,
        )
        if error:
            status_var.set(error)
        return options

    def _redraw() -> None:
        validated = _validated_options()
        if validated is None:
            return
        status_var.set("")
        ax.clear()
        bins = validated["bins"]
        if preview_renderer is not None:
            preview_renderer(
                ax,
                title=validated["title"],
                data=values,
                color=validated["color"],
                n_bins=bins,
                show_title=validated["show_title"],
                show_statistics=validated["show_statistics"],
                statistics_mode=validated["statistics_mode"],
                x_label=validated["x_label"],
                y_label=validated["y_label"],
                xlim=validated["xlim"],
                ylim=validated["ylim"],
            )
        elif values:
            ax.hist(
                values,
                bins=bins,
                color=color_var.get(),
                alpha=0.75,
                edgecolor="black",
                linewidth=0.9,
            )
            if show_stats_var.get() and stats_mode_var.get() != "none":
                mean = float(np.mean(values))
                std = float(np.std(values))
                ax.axvline(mean, color="black", linewidth=1.5)
                ax.axvspan(mean - std, mean + std, color="#D3D3D3", alpha=0.4)
                handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=1.5,
                        label=f"Mean: {mean:.2f}",
                    ),
                    Patch(
                        facecolor="#D3D3D3",
                        edgecolor="none",
                        alpha=0.4,
                        label=f"+/-1 Std: {std:.2f}",
                    ),
                ]
                if stats_mode_var.get() == "full":
                    v_min = float(np.min(values))
                    v_max = float(np.max(values))
                    ax.axvline(
                        v_min, color="green", linestyle=":", linewidth=1.5
                    )
                    ax.axvline(
                        v_max, color="purple", linestyle=":", linewidth=1.5
                    )
                    handles.insert(
                        1,
                        Line2D(
                            [0],
                            [0],
                            color="green",
                            linestyle=":",
                            lw=1.5,
                            label=f"Min: {v_min:.2f}",
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
                            label=f"Max: {v_max:.2f}",
                        ),
                    )
                ax.legend(handles=handles, loc="upper left")
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        if show_title_var.get():
            ax.set_title(title_var.get(), fontweight="bold")
        ax.set_xlabel(x_label_var.get())
        ax.set_ylabel(y_label_var.get())
        xmin, xmax = validated["xlim"]
        ymin, ymax = validated["ylim"]
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        fig.set_size_inches(*validated["figsize"], forward=True)
        fig.tight_layout()
        status_var.set(
            f"Applied: bins={bins}, color={validated['color']}, stats={validated['statistics_mode']}"
        )
        canvas.draw_idle()

    def _save() -> None:
        validated = _validated_options()
        if validated is None:
            return
        _redraw()
        result["action"] = "save"
        root.quit()

    def _skip() -> None:
        result["action"] = "skip"
        root.quit()

    def _cancel() -> None:
        result["action"] = "cancel"
        root.quit()

    for child in control.winfo_children():
        if isinstance(child, tk.Entry):
            child.bind("<Return>", lambda _event: _redraw())
    tk.Button(control, text="Apply Preview", command=_redraw).pack(
        fill=tk.X, pady=(8, 2)
    )
    tk.Button(control, text="Save This", command=_save).pack(fill=tk.X, pady=2)
    tk.Button(control, text="Skip", command=_skip).pack(fill=tk.X, pady=2)
    tk.Button(control, text="Cancel Inspection", command=_cancel).pack(
        fill=tk.X, pady=2
    )
    root.protocol("WM_DELETE_WINDOW", _save)
    _redraw()
    root.mainloop()
    if result["action"] == "skip":
        root.destroy()
        return {"skip": True}
    if result["action"] == "cancel":
        root.destroy()
        return {"cancel_inspection": True}
    validated = _validated_options() or {
        "bins": default_bins,
        "figsize": (float(figsize[0]), float(figsize[1])),
        "xlim": (None, None),
        "ylim": (None, None),
    }
    final_options = {
        "title": validated["title"],
        "color": validated["color"],
        "bin_fraction": bin_fraction,
        "n_bins_override": validated["bins"],
        "bins_changed": validated["bins_changed"],
        "figsize": validated["figsize"],
        "show_title": validated["show_title"],
        "show_statistics": validated["show_statistics"],
        "statistics_mode": validated["statistics_mode"],
        "x_label": validated["x_label"],
        "y_label": validated["y_label"],
        "xlim": validated["xlim"],
        "ylim": validated["ylim"],
    }
    root.destroy()
    return final_options


def discover_images_gui() -> Optional[Path]:
    """
    Open GUI dialog to select parent folder and auto-discover TIFF image directory.

    Returns:
        Path to the image directory, or None if cancelled or not found.
    """
    logger.info("Opening folder selection dialog...")
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        parent_folder = filedialog.askdirectory(
            title="Select Analysis Parent Folder", parent=root
        )
        if not parent_folder:
            logger.warning("No folder selected.")
            return None
        parent_path = Path(parent_folder)
        logger.info(f"Selected: {parent_path}")
        if list(parent_path.glob("*.tif")) or list(parent_path.glob("*.tiff")):
            logger.info(f"Found TIFF files directly in: {parent_path.name}")
            return parent_path
        for subdir in parent_path.iterdir():
            if subdir.is_dir():
                if list(subdir.glob("*.tif")) or list(subdir.glob("*.tiff")):
                    logger.info(
                        f"Found TIFF files in subdirectory: {subdir.name}"
                    )
                    return subdir
        logger.error("No directory containing TIFF files found.")
        return None
    except Exception as e:
        logger.error(f"GUI error: {e}")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


class CropSelector:
    """
    Interactive tool for cropping a region of the image before ROI selection.

    Users draw a single rectangle and press 'A' or 'Enter' to confirm the crop.
    Closing the window without confirming returns None (no crop).

    Attributes:
        image: The image being displayed.
        crop_bounds: The confirmed crop bounds or None.
    """

    def __init__(self, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        self.image = image
        self.crop_bounds: Optional[Tuple[int, int, int, int]] = None
        self._current: Optional[Tuple[int, int, int, int]] = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(
            "Draw Crop Region → Press 'A' or 'Enter' to Confirm → Close to Skip"
        )
        self.ax.axis("off")
        self.rs = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        print("\n--- Crop Selector Controls ---")
        print("1. Drag to select the crop region.")
        print("2. Press 'A' or 'Enter' to CONFIRM the crop.")
        print("3. Close the window to SKIP cropping.")

    def _on_select(self, eclick, erelease) -> None:
        x1, y1 = (eclick.xdata, eclick.ydata)
        x2, y2 = (erelease.xdata, erelease.ydata)
        if any((v is None for v in [x1, y1, x2, y2])):
            return
        height, width = self.image.shape[:2]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        x_min = max(0, int(round(x_min)))
        x_max = min(width, int(round(x_max)))
        y_min = max(0, int(round(y_min)))
        y_max = min(height, int(round(y_max)))
        self._current = (y_min, y_max, x_min, x_max)

    def _on_key(self, event) -> None:
        if event.key in ["a", "enter", "A"] and self._current:
            self.crop_bounds = self._current
            y_min, y_max, x_min, x_max = self.crop_bounds
            print(
                f"  [✓] Crop confirmed: rows {y_min}-{y_max}, cols {x_min}-{x_max}"
            )
            plt.close(self.fig)

    def select_crop(self) -> Optional[Tuple[int, int, int, int]]:
        """Display, block, return (r_min, r_max, c_min, c_max) or None if skipped."""
        plt.show()
        return self.crop_bounds


class ROISelector:
    """
    Interactive tool for selecting multiple Regions of Interest (ROIs) on an image.

    Uses matplotlib's RectangleSelector widget to allow users to draw boxes
    on the image and confirm selections with keyboard shortcuts.

    Attributes:
        image: The image being displayed.
        rois: List of selected ROIs as (row_start, row_end, col_start, col_end) tuples.

    Example:
        >>> selector = ROISelector(mean_image)
        >>> rois = selector.select_rois()  # Blocks until window closed
        >>> print(f"Selected {len(rois)} ROIs")
    """

    def __init__(self, image: np.ndarray) -> None:
        """
        Initialize the ROI selector.

        Args:
            image: 2D NumPy array (grayscale image) to display.

        Raises:
            ValueError: If image is None or empty.
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        self.image = image
        self.rois: List[Tuple[int, int, int, int]] = []
        self.current_selection: Optional[Tuple[int, int, int, int]] = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(
            "Draw Box → Press 'A' or 'Enter' to Add ROI → Close to Finish"
        )
        self.ax.axis("off")
        self.rs = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        logger.info("ROI Selector initialized")
        print("\n--- ROI Selector Controls ---")
        print("1. Drag mouse to select a region.")
        print("2. Adjust corners if needed (interactive).")
        print("3. Press 'A' or 'Enter' to CONFIRM and ADD the ROI.")
        print("4. Repeat for multiple ROIs.")
        print("5. Close the window when done.")

    def _on_select(self, eclick, erelease) -> None:
        """Callback for RectangleSelector - stores current selection."""
        x1, y1 = (eclick.xdata, eclick.ydata)
        x2, y2 = (erelease.xdata, erelease.ydata)
        if any((v is None for v in [x1, y1, x2, y2])):
            return
        height, width = self.image.shape
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        x_min = max(0, int(round(x_min)))
        x_max = min(width, int(round(x_max)))
        y_min = max(0, int(round(y_min)))
        y_max = min(height, int(round(y_max)))
        self.current_selection = (y_min, y_max, x_min, x_max)
        logger.debug(
            f"Current selection: rows {y_min}-{y_max}, cols {x_min}-{x_max}"
        )

    def _on_key(self, event) -> None:
        """Handle key presses for confirming ROI selections."""
        if event.key in ["a", "enter", "A"] and self.current_selection:
            self.rois.append(self.current_selection)
            y_min, y_max, x_min, x_max = self.current_selection
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            self.ax.add_patch(rect)
            self.ax.text(
                x_min,
                y_min - 5,
                f"ROI {len(self.rois)}",
                color="red",
                fontweight="bold",
            )
            self.fig.canvas.draw()
            logger.info(
                f"Added ROI {len(self.rois)}: rows {y_min}-{y_max}, cols {x_min}-{x_max}"
            )
            print(
                f"  [+] Added ROI {len(self.rois)}: Rows {y_min}-{y_max}, Cols {x_min}-{x_max}"
            )
            self.current_selection = None

    def select_rois(self) -> List[Tuple[int, int, int, int]]:
        """
        Display the selector and block until the window is closed.

        Returns:
            List of ROI tuples as (row_start, row_end, col_start, col_end).
        """
        plt.show()
        logger.info(f"Selection complete: {len(self.rois)} ROIs selected")
        return self.rois


class POISelector:
    """
    Interactive tool for selecting multiple Points of Interest (POIs) on an image.

    Attributes:
        image: The image being displayed.
        points: List of selected points as (row, col) tuples.
    """

    def __init__(self, image: np.ndarray, title: str = "Select POIs") -> None:
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        self.image = image
        self.points: List[Tuple[int, int]] = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(
            f"{title}\nLeft Click to Add Point | Middle Click/Z to Undo Last | Close to Finish"
        )
        self.ax.axis("off")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        print("\n--- POI Selector Controls ---")
        print("1. Left Click to ADD a point.")
        print("2. Middle Click (or 'z') to REMOVE the last point.")
        print("3. Close the window when done.")

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            r, c = (int(event.ydata), int(event.xdata))
            self.points.append((r, c))
            self.ax.plot(c, r, "rx", markersize=10, markeredgewidth=2)
            self.ax.text(
                c + 2,
                r + 2,
                f"{len(self.points)}",
                color="yellow",
                fontsize=12,
            )
            self.fig.canvas.draw()
            print(f"  [+] Added POI {len(self.points)} at (r={r}, c={c})")
        elif event.button == 2:
            self._undo_last()

    def _on_key(self, event) -> None:
        if event.key == "z":
            self._undo_last()

    def _undo_last(self):
        if self.points:
            removed = self.points.pop()
            print(f"  [-] Removed POI at {removed}")
            self.ax.clear()
            self.ax.imshow(self.image, cmap="gray")
            self.ax.set_title(self.ax.get_title())
            self.ax.axis("off")
            for i, (r, c) in enumerate(self.points):
                self.ax.plot(c, r, "rx", markersize=10, markeredgewidth=2)
                self.ax.text(
                    c + 2, r + 2, f"{i + 1}", color="yellow", fontsize=12
                )
            self.fig.canvas.draw()

    def select_pois(self) -> List[Tuple[int, int]]:
        """
        Display, block, return points (r, c).
        """
        plt.show()
        return self.points


class POIPairSelector:
    """
    Interactive tkinter dialog for pairing selected POIs.

    After POIs are selected, this dialog lists them and lets the user
    define one or more (reference POI, target POI) pairs to use in the
    arithmetic analysis. Any POI can be left unpaired / skipped.

    Attributes:
        points: List of (row, col) tuples from POISelector.
        pairs: List of (i, j) zero-based index pairs chosen by the user.

    Example:
        >>> selector = POIPairSelector(points, roi_id=1)
        >>> pairs = selector.select_pairs()  # [(0, 1), (2, 1)] etc.
    """

    def __init__(self, points: List[Tuple[int, int]], roi_id: int = 1) -> None:
        """
        Args:
            points: POI coordinates as (row, col) — in ROI space.
            roi_id: ROI identifier shown in the window title.
        """
        self.points = points
        self.pairs: List[Tuple[int, int]] = []
        self._roi_id = roi_id

    def select_pairs(self) -> List[Tuple[int, int]]:
        """
        Open the dialog and block until the user closes it.

        Uses a hidden Tk root + Toplevel child to avoid conflicting with
        matplotlib's TkAgg event loop (which leaves Tk in a dirty state
        after plt.show() returns).

        Returns:
            List of (ref_idx, target_idx) zero-based pairs.
            Empty list if cancelled or no pairs added.
        """
        n = len(self.points)
        if n < 2:
            logger.info("Fewer than 2 POIs — skipping pair selection.")
            return []
        root = tk.Tk()
        root.withdraw()
        top = tk.Toplevel(root)
        top.title(f"POI Pair Selection — ROI {self._roi_id}")
        top.resizable(False, False)
        top.lift()
        top.focus_force()
        info_frame = tk.Frame(top, padx=10, pady=6)
        info_frame.pack(fill="x")
        tk.Label(
            info_frame,
            text=f"ROI {self._roi_id}: {n} POIs selected.\nChoose which POI is the REFERENCE and which is the TARGET\nfor each arithmetic pair, then click 'Add Pair'.\nClick 'Done' when finished (no pairs = skip arithmetic).",
            justify="left",
            fg="#333333",
        ).pack(anchor="w")
        list_frame = tk.LabelFrame(top, text="Selected POIs", padx=8, pady=4)
        list_frame.pack(fill="x", padx=10, pady=4)
        for idx, (r, c) in enumerate(self.points):
            tk.Label(
                list_frame,
                text=f"  POI {idx + 1}  →  row={r}, col={c}",
                anchor="w",
            ).pack(fill="x")
        builder_frame = tk.LabelFrame(top, text="Add a Pair", padx=8, pady=6)
        builder_frame.pack(fill="x", padx=10, pady=4)
        poi_labels = [f"POI {i + 1}" for i in range(n)]
        tk.Label(builder_frame, text="Reference:").grid(
            row=0, column=0, sticky="w"
        )
        ref_var = tk.StringVar(value=poi_labels[0])
        ref_menu = tk.OptionMenu(builder_frame, ref_var, *poi_labels)
        ref_menu.config(width=10)
        ref_menu.grid(row=0, column=1, padx=6, pady=2)
        tk.Label(builder_frame, text="Target:").grid(
            row=0, column=2, sticky="w"
        )
        tgt_var = tk.StringVar(value=poi_labels[1] if n > 1 else poi_labels[0])
        tgt_menu = tk.OptionMenu(builder_frame, tgt_var, *poi_labels)
        tgt_menu.config(width=10)
        tgt_menu.grid(row=0, column=3, padx=6, pady=2)
        pairs_frame = tk.LabelFrame(top, text="Queued Pairs", padx=8, pady=4)
        pairs_frame.pack(fill="both", expand=True, padx=10, pady=4)
        pairs_listbox = tk.Listbox(pairs_frame, height=5, width=40)
        pairs_listbox.pack(fill="both", expand=True)

        def add_pair():
            ref_idx = poi_labels.index(ref_var.get())
            tgt_idx = poi_labels.index(tgt_var.get())
            if ref_idx == tgt_idx:
                from tkinter import messagebox

                messagebox.showwarning(
                    "Invalid Pair",
                    "Reference and Target must be different POIs.",
                    parent=top,
                )
                return
            pair = (ref_idx, tgt_idx)
            if pair in self.pairs:
                from tkinter import messagebox

                messagebox.showinfo(
                    "Duplicate",
                    "This pair is already in the list.",
                    parent=top,
                )
                return
            self.pairs.append(pair)
            pairs_listbox.insert(
                tk.END,
                f"  Ref: POI {ref_idx + 1}  ←→  Target: POI {tgt_idx + 1}",
            )
            print(f"  [Pair] Added: POI {ref_idx + 1} ↔ POI {tgt_idx + 1}")

        def remove_last():
            if self.pairs:
                self.pairs.pop()
                pairs_listbox.delete(tk.END)

        btn_frame = tk.Frame(top, padx=10, pady=6)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Add Pair", command=add_pair, width=12).pack(
            side="left", padx=4
        )
        tk.Button(
            btn_frame, text="Remove Last", command=remove_last, width=12
        ).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Done", command=top.destroy, width=12).pack(
            side="right", padx=4
        )
        root.wait_window(top)
        root.destroy()
        logger.info(
            f"ROI {self._roi_id}: {len(self.pairs)} pair(s) selected by user."
        )
        return self.pairs
