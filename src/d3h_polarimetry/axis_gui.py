"""Interactive adapter for resolving ambiguous microscopy dimensions."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from .axis_data import (
    AxisMapping,
    AxisResolutionError,
    CANONICAL_DIMENSIONS,
    InputDescriptor,
    infer_axis_mapping,
)


def build_axis_mapping(
    descriptor: InputDescriptor,
    roles: Mapping[str, str],
    *,
    starts: Mapping[str, float] | None = None,
    steps: Mapping[str, float] | None = None,
    units: Mapping[str, str] | None = None,
) -> AxisMapping:
    """Build and validate a mapping from GUI-like field values."""
    inferred = infer_axis_mapping(descriptor)
    assignments = {**dict(inferred.assignments), **dict(roles)}
    starts = dict(starts or {})
    steps = dict(steps or {})
    requested_units = dict(units or {})
    coordinates = dict(inferred.coordinates)
    resolved_units = dict(inferred.units)
    evidence = dict(inferred.evidence)
    for axis in descriptor.axes:
        semantic = assignments.get(axis.source_name)
        if not semantic:
            continue
        if axis.source_name in starts or axis.source_name in steps:
            start = float(starts.get(axis.source_name, 0.0))
            step = float(steps.get(axis.source_name, 1.0))
            coordinates[semantic] = tuple(start + step * np.arange(axis.size))
        unit = requested_units.get(axis.source_name, "").strip()
        if unit:
            resolved_units[semantic] = unit
        evidence[axis.source_name] = "user-confirmed in axis mapping window"
    mapping = AxisMapping(assignments, coordinates, resolved_units, evidence)
    mapping.validate(descriptor)
    return mapping


def collect_axis_roles(
    descriptor: InputDescriptor, selected_roles: Mapping[str, str]
) -> dict[str, str]:
    """Resolve widget values, retaining metadata-supported axis meanings."""
    inferred = infer_axis_mapping(descriptor)
    roles = dict(inferred.assignments)
    for source_name, selected in selected_roles.items():
        value = str(selected).strip()
        if value:
            roles[source_name] = value
    missing = [
        axis.source_name
        for axis in descriptor.axes
        if not roles.get(axis.source_name)
    ]
    if missing:
        raise AxisResolutionError(
            f"Choose a meaning for: {', '.join(missing)}"
        )
    return roles


def _activate_axis_dialog(dialog: Any) -> None:
    """Center an axis dialog and bring it in front on desktop platforms."""
    dialog.update_idletasks()
    width = max(int(dialog.winfo_reqwidth()), 1)
    height = max(int(dialog.winfo_reqheight()), 1)
    x_pos = max((int(dialog.winfo_screenwidth()) - width) // 2, 0)
    y_pos = max((int(dialog.winfo_screenheight()) - height) // 2, 0)
    dialog.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
    dialog.deiconify()
    dialog.lift()
    try:
        dialog.attributes("-topmost", True)

        def release_topmost() -> None:
            try:
                if dialog.winfo_exists():
                    dialog.attributes("-topmost", False)
            except Exception:
                pass

        dialog.after(300, release_topmost)
    except Exception:
        pass
    dialog.focus_force()


def resolve_axes_interactive(descriptor: InputDescriptor) -> AxisMapping:
    """Open a blocking Tk window and return a user-confirmed axis mapping."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        raise AxisResolutionError(
            "Tk is unavailable; provide an axis mapping or preset"
        ) from exc
    inferred = infer_axis_mapping(descriptor)
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        dialog = tk.Toplevel(root)
    except Exception as exc:
        if root is not None:
            root.destroy()
        raise AxisResolutionError(
            "Cannot open the axis mapping window; provide --axis-mapping or --axis-preset"
        ) from exc
    dialog.title("Clarify microscopy dimensions")
    result: dict[str, AxisMapping | None] = {"mapping": None}
    role_vars: dict[str, Any] = {}
    role_boxes: dict[str, Any] = {}
    start_vars: dict[str, Any] = {}
    step_vars: dict[str, Any] = {}
    unit_vars: dict[str, Any] = {}
    intro = "Confirm what each stored dimension represents. Shape alone cannot distinguish time from polarization angle. Coordinates are start + index × step."
    ttk.Label(dialog, text=intro, wraplength=760, justify="left").grid(
        row=0, column=0, columnspan=7, sticky="w", padx=10, pady=(10, 8)
    )
    headers = (
        "Stored axis",
        "Length",
        "Detected evidence",
        "Meaning",
        "Start",
        "Step",
        "Unit",
    )
    for column, header in enumerate(headers):
        ttk.Label(dialog, text=header).grid(
            row=1, column=column, sticky="w", padx=5
        )

    def update_defaults(source_name: str) -> None:
        axis = descriptor.axis(source_name)
        role = role_vars[source_name].get()
        if role == "polarization_angle" and unit_vars[source_name].get() in {
            "",
            "index",
        }:
            unit_vars[source_name].set("degree")
            step_vars[source_name].set(str(360.0 / axis.size))
        elif role in {"x", "y", "z"} and (not unit_vars[source_name].get()):
            unit_vars[source_name].set("pixel")

    for offset, axis in enumerate(descriptor.axes, start=2):
        source = axis.source_name
        suggested = inferred.assignments.get(source, "")
        ttk.Label(dialog, text=f"{axis.index}: {source}").grid(
            row=offset, column=0, sticky="w", padx=5
        )
        ttk.Label(dialog, text=str(axis.size)).grid(
            row=offset, column=1, sticky="w", padx=5
        )
        ttk.Label(dialog, text=axis.evidence, wraplength=220).grid(
            row=offset, column=2, sticky="w", padx=5
        )
        role_vars[source] = tk.StringVar(master=dialog, value=suggested)
        role_values = ("", *CANONICAL_DIMENSIONS)
        role_box = ttk.Combobox(
            dialog,
            textvariable=role_vars[source],
            values=role_values,
            state="readonly",
            width=20,
        )
        if suggested in role_values:
            role_box.current(role_values.index(suggested))
        role_box.grid(row=offset, column=3, padx=5, pady=3)
        role_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, name=source: update_defaults(name),
        )
        role_boxes[source] = role_box
        start_vars[source] = tk.StringVar(master=dialog, value="0")
        step_vars[source] = tk.StringVar(master=dialog, value="1")
        default_unit = "pixel" if suggested in {"x", "y", "z"} else "index"
        unit_vars[source] = tk.StringVar(master=dialog, value=default_unit)
        ttk.Entry(dialog, textvariable=start_vars[source], width=10).grid(
            row=offset, column=4, padx=5
        )
        ttk.Entry(dialog, textvariable=step_vars[source], width=10).grid(
            row=offset, column=5, padx=5
        )
        ttk.Entry(dialog, textvariable=unit_vars[source], width=12).grid(
            row=offset, column=6, padx=5
        )
    button_row = len(descriptor.axes) + 2

    def confirm() -> None:
        try:
            dialog.focus_set()
            dialog.update_idletasks()
            roles = collect_axis_roles(
                descriptor,
                {name: box.get() for name, box in role_boxes.items()},
            )
            mapping = build_axis_mapping(
                descriptor,
                roles,
                starts={
                    name: float(variable.get())
                    for name, variable in start_vars.items()
                },
                steps={
                    name: float(variable.get())
                    for name, variable in step_vars.items()
                },
                units={
                    name: variable.get()
                    for name, variable in unit_vars.items()
                },
            )
        except (AxisResolutionError, ValueError) as exc:
            messagebox.showerror(
                "Invalid axis mapping", str(exc), parent=dialog
            )
            return
        result["mapping"] = mapping
        dialog.destroy()

    ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
        row=button_row, column=5, sticky="e", padx=5, pady=10
    )
    ttk.Button(dialog, text="Use mapping", command=confirm).grid(
        row=button_row, column=6, sticky="e", padx=5, pady=10
    )
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    _activate_axis_dialog(dialog)
    try:
        root.wait_window(dialog)
    finally:
        root.destroy()
    if result["mapping"] is None:
        raise AxisResolutionError("Axis mapping was cancelled")
    return result["mapping"]
