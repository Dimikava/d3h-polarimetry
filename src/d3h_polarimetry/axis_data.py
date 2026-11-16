"""Named-axis microscopy input model and resolution helpers.

This module keeps storage layout, scientific axis meaning, and interactive
axis selection separate.  Readers describe what they found; callers resolve
ambiguous dimensions; analysis code requests the dimensions it needs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import numpy as np
import tifffile
import xarray as xr
from .io import DEFAULT_MAX_DECODED_BYTES, read_tiff_source_traced
from .io.models import AxisDescriptor, InputDescriptor, RawMicroscopyData

CANONICAL_DIMENSIONS = ("polarization_angle", "time", "channel", "z", "y", "x")
POLARIZATION_CUBE_DIMS = ("polarization_angle", "y", "x")
_TIFF_AXIS_NAMES = {"T": "time", "C": "channel", "Z": "z", "Y": "y", "X": "x"}


class AxisResolutionError(ValueError):
    """Base error for invalid or incomplete axis semantics."""


class AmbiguousAxesError(AxisResolutionError):
    """Raised when a source dimension still needs an explicit meaning."""

    def __init__(self, unresolved: Sequence[str]) -> None:
        self.unresolved = tuple(unresolved)
        joined = ", ".join(self.unresolved)
        super().__init__(f"Axis meaning is unresolved for: {joined}")


@dataclass(frozen=True)
class AxisMapping:
    """Resolved mapping from storage dimensions to scientific dimensions."""

    assignments: Mapping[str, str]
    coordinates: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    evidence: Mapping[str, str] = field(default_factory=dict)
    name: str | None = None

    def validate(self, descriptor: InputDescriptor) -> None:
        expected = set(descriptor.source_dims)
        actual = set(self.assignments)
        missing = expected - actual
        unknown = actual - expected
        if missing:
            raise AmbiguousAxesError(sorted(missing))
        if unknown:
            raise AxisResolutionError(
                f"Mapping contains unknown source axes: {sorted(unknown)}"
            )
        semantic = tuple(
            (self.assignments[source] for source in descriptor.source_dims)
        )
        if any((not name or not str(name).strip() for name in semantic)):
            raise AxisResolutionError(
                "Every source axis must have a non-empty semantic name"
            )
        duplicates = sorted(
            {name for name in semantic if semantic.count(name) > 1}
        )
        if duplicates:
            raise AxisResolutionError(
                f"Semantic dimensions must be unique: {duplicates}"
            )
        sizes = {
            self.assignments[axis.source_name]: axis.size
            for axis in descriptor.axes
        }
        for dim, values in self.coordinates.items():
            if dim not in sizes:
                raise AxisResolutionError(
                    f"Coordinates supplied for unmapped dimension: {dim}"
                )
            if len(values) != sizes[dim]:
                raise AxisResolutionError(
                    f"Coordinate length for {dim} is {len(values)}, expected {sizes[dim]}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "assignments": dict(self.assignments),
            "coordinates": {
                key: list(values) for key, values in self.coordinates.items()
            },
            "units": dict(self.units),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AxisMapping":
        return cls(
            assignments=dict(payload.get("assignments", {})),
            coordinates={
                str(key): tuple(values)
                for key, values in dict(payload.get("coordinates", {})).items()
            },
            units={
                str(key): str(value)
                for key, value in dict(payload.get("units", {})).items()
            },
            evidence={
                str(key): str(value)
                for key, value in dict(payload.get("evidence", {})).items()
            },
            name=payload.get("name"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AxisMapping":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        return resolved


@dataclass(frozen=True)
class MicroscopyDataset:
    """Canonical named ND data and its input provenance."""

    data: xr.DataArray
    mapping: AxisMapping
    descriptor: InputDescriptor
    filenames: tuple[str, ...] = ()

    def select_analysis_cube(
        self,
        required_dims: Sequence[str] = POLARIZATION_CUBE_DIMS,
        *,
        selections: Mapping[str, Any] | None = None,
    ) -> xr.DataArray:
        """Select extra dimensions and return a cube in the requested order."""
        required = tuple(required_dims)
        missing = [dim for dim in required if dim not in self.data.dims]
        if missing:
            raise AxisResolutionError(
                f"Analysis requires dimensions {required}; dataset is missing {missing}"
            )
        selections = dict(selections or {})
        extras = [dim for dim in self.data.dims if dim not in required]
        unresolved = [dim for dim in extras if dim not in selections]
        if unresolved:
            raise AxisResolutionError(
                "Select one coordinate for every extra dimension before analysis: "
                + ", ".join(unresolved)
            )
        unknown = sorted(set(selections) - set(extras))
        if unknown:
            raise AxisResolutionError(
                f"Selections reference non-extra dimensions: {unknown}"
            )
        selected = self.data
        for dim in extras:
            try:
                selected = selected.sel({dim: selections[dim]}, drop=True)
            except (KeyError, IndexError) as exc:
                available = selected.coords[dim].values.tolist()
                raise AxisResolutionError(
                    f"Coordinate {selections[dim]!r} is not available for {dim}: {available}"
                ) from exc
        return selected.transpose(*required)

    def provenance(
        self, *, selections: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "source": str(self.descriptor.source)
            if self.descriptor.source
            else None,
            "source_kind": self.descriptor.source_kind,
            "source_shape": list(self.descriptor.shape),
            "source_axes": [axis.to_dict() for axis in self.descriptor.axes],
            "resolved_mapping": self.mapping.to_dict(),
            "canonical_dims": list(self.data.dims),
            "selections": dict(selections or {}),
        }

    def write_provenance(
        self, path: str | Path, *, selections: Mapping[str, Any] | None = None
    ) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(
                self.provenance(selections=selections),
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        return resolved


def describe_array(
    values: np.ndarray,
    *,
    source_dims: Sequence[str] | None = None,
    source: str | Path | None = None,
    source_kind: str = "array",
    suggestions: Mapping[str, str] | None = None,
    coordinates: Mapping[str, Sequence[Any]] | None = None,
    evidence: Mapping[str, str] | None = None,
) -> InputDescriptor:
    """Describe an array without assigning unsupported scientific meaning."""
    values = np.asarray(values)
    dims = tuple(
        source_dims or (f"dim_{index}" for index in range(values.ndim))
    )
    if len(dims) != values.ndim or len(set(dims)) != len(dims):
        raise ValueError(
            "source_dims must contain one unique name per array dimension"
        )
    suggestions = dict(suggestions or {})
    coordinates = dict(coordinates or {})
    evidence = dict(evidence or {})
    axes = tuple(
        (
            AxisDescriptor(
                source_name=name,
                index=index,
                size=int(values.shape[index]),
                suggested_name=suggestions.get(name),
                coordinates=tuple(coordinates[name])
                if name in coordinates
                else None,
                evidence=evidence.get(name, "unresolved"),
                confidence="high" if name in suggestions else "none",
            )
            for index, name in enumerate(dims)
        )
    )
    return InputDescriptor(
        shape=tuple((int(size) for size in values.shape)),
        axes=axes,
        source=Path(source) if source is not None else None,
        source_kind=source_kind,
    )


def infer_axis_mapping(descriptor: InputDescriptor) -> AxisMapping:
    """Return metadata-supported assignments, leaving ambiguity explicit."""
    assignments: dict[str, str] = {}
    coordinates: dict[str, tuple[Any, ...]] = {}
    units: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for axis in descriptor.axes:
        if axis.suggested_name is None:
            continue
        assignments[axis.source_name] = axis.suggested_name
        if axis.coordinates is not None:
            coordinates[axis.suggested_name] = axis.coordinates
        if axis.unit:
            units[axis.suggested_name] = axis.unit
        evidence[axis.source_name] = axis.evidence
    return AxisMapping(assignments, coordinates, units, evidence)


def complete_mapping(
    descriptor: InputDescriptor, mapping: AxisMapping | None = None
) -> AxisMapping:
    """Merge explicit assignments over metadata-supported suggestions."""
    inferred = infer_axis_mapping(descriptor)
    if mapping is None:
        merged = inferred
    else:
        merged = AxisMapping(
            assignments={**inferred.assignments, **dict(mapping.assignments)},
            coordinates={**inferred.coordinates, **dict(mapping.coordinates)},
            units={**inferred.units, **dict(mapping.units)},
            evidence={**inferred.evidence, **dict(mapping.evidence)},
            name=mapping.name,
        )
    merged.validate(descriptor)
    return merged


def canonicalize(
    raw: RawMicroscopyData, mapping: AxisMapping
) -> MicroscopyDataset:
    """Apply one validated mapping and return named canonical data."""
    mapping.validate(raw.descriptor)
    source_coords: dict[str, Sequence[Any]] = {}
    for axis in raw.descriptor.axes:
        semantic = mapping.assignments[axis.source_name]
        values = mapping.coordinates.get(semantic, axis.coordinates)
        if values is None:
            values = np.arange(axis.size)
        values = np.asarray(values)
        unit = mapping.units.get(semantic, axis.unit)
        if semantic == "polarization_angle":
            try:
                values = values.astype(float)
            except (TypeError, ValueError) as exc:
                raise AxisResolutionError(
                    "polarization_angle requires numeric coordinates; provide angle values in the mapping instead of relying on filenames"
                ) from exc
            normalized = (
                unit.lower().replace("°", "degree")
                if unit is not None
                else "radian"
            )
            if normalized in {"degree", "degrees", "deg"}:
                values = np.deg2rad(values)
                unit = "radian"
            elif normalized not in {"radian", "radians", "rad"}:
                raise AxisResolutionError(
                    f"Unsupported polarization-angle unit {unit!r}; use degree or radian"
                )
            else:
                unit = "radian"
        source_coords[axis.source_name] = values
    array = xr.DataArray(
        raw.values,
        dims=raw.descriptor.source_dims,
        coords=source_coords,
        name="intensity",
        attrs={"source_kind": raw.descriptor.source_kind},
    ).rename(dict(mapping.assignments))
    for dim in array.dims:
        unit = mapping.units.get(dim)
        if (
            dim == "polarization_angle"
            and unit
            and (unit.lower() in {"degree", "degrees", "deg", "°"})
        ):
            unit = "radian"
        if unit:
            array.coords[dim].attrs["units"] = unit
    return MicroscopyDataset(array, mapping, raw.descriptor, raw.filenames)


def read_microscopy_source(
    source: str | Path,
    *,
    sort_by: str = "name",
    tiff_series: int | None = None,
    max_frames: int | None = None,
    max_decoded_bytes: int | None = DEFAULT_MAX_DECODED_BYTES,
) -> RawMicroscopyData:
    """Decode one TIFF file or a folder of TIFF files without guessing sequence meaning."""
    return read_tiff_source_traced(
        source,
        sort_by=sort_by,
        tiff_series=tiff_series,
        max_frames=max_frames,
        max_decoded_bytes=max_decoded_bytes,
    ).raw


def load_dataset(
    source: str | Path,
    *,
    axis_mapping: AxisMapping | str | Path | None = None,
    mapping_preset: str | None = None,
    preset_store: "AxisPresetStore | None" = None,
    sort_by: str = "name",
    tiff_series: int | None = None,
    max_frames: int | None = None,
    max_decoded_bytes: int | None = DEFAULT_MAX_DECODED_BYTES,
) -> MicroscopyDataset:
    """Load and canonicalize a TIFF source, failing on unresolved semantics."""
    raw = read_microscopy_source(
        source,
        sort_by=sort_by,
        tiff_series=tiff_series,
        max_frames=max_frames,
        max_decoded_bytes=max_decoded_bytes,
    )
    mapping: AxisMapping | None
    if isinstance(axis_mapping, (str, Path)):
        mapping = AxisMapping.load(axis_mapping)
    else:
        mapping = axis_mapping
    if mapping_preset is not None:
        if mapping is not None:
            raise AxisResolutionError(
                "Use either axis_mapping or mapping_preset, not both"
            )
        mapping = (preset_store or AxisPresetStore.default()).load(
            mapping_preset
        )
    return canonicalize(raw, complete_mapping(raw.descriptor, mapping))


class AxisPresetStore:
    """Named reusable axis mappings stored outside raw datasets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def default(cls) -> "AxisPresetStore":
        return cls(Path.home() / ".d3h_polarimetry" / "axis_presets.json")

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "presets": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload.get("presets", {}), dict):
            raise AxisResolutionError(
                f"Invalid axis preset store: {self.path}"
            )
        return payload

    def load(self, name: str) -> AxisMapping:
        payload = self._read()
        try:
            preset = payload["presets"][name]
        except KeyError as exc:
            raise AxisResolutionError(
                f"Axis preset not found: {name}"
            ) from exc
        return AxisMapping.from_dict(preset)

    def save(self, name: str, mapping: AxisMapping) -> Path:
        if not name.strip():
            raise AxisResolutionError("Axis preset name cannot be empty")
        payload = self._read()
        named = AxisMapping(
            assignments=mapping.assignments,
            coordinates=mapping.coordinates,
            units=mapping.units,
            evidence=mapping.evidence,
            name=name,
        )
        payload.setdefault("presets", {})[name] = named.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._read().get("presets", {})))


def parse_axis_selections(items: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeated ``DIM=COORDINATE`` CLI selections."""
    selections: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise AxisResolutionError(
                f"Axis selection must be DIM=VALUE: {item}"
            )
        dim, raw_value = (part.strip() for part in item.split("=", 1))
        if not dim or not raw_value:
            raise AxisResolutionError(
                f"Axis selection must be DIM=VALUE: {item}"
            )
        selections[dim] = _parse_scalar(raw_value)
    return selections


def _tiff_axes(path: Path, ndim: int) -> str | None:
    try:
        with tifffile.TiffFile(path) as tif:
            axes = tif.series[0].axes
        return axes if len(axes) == ndim else None
    except (tifffile.TiffFileError, IndexError):
        return None


def _source_axis_metadata(
    axes_text: str | None, ndim: int | None = None
) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    count = len(axes_text) if axes_text is not None else int(ndim or 0)
    names: list[str] = []
    suggestions: dict[str, str] = {}
    evidence: dict[str, str] = {}
    used: set[str] = set()
    for index in range(count):
        symbol = axes_text[index].upper() if axes_text is not None else ""
        base = _TIFF_AXIS_NAMES.get(symbol, f"dim_{index}")
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
        if symbol in _TIFF_AXIS_NAMES:
            suggestions[name] = _TIFF_AXIS_NAMES[symbol]
            evidence[name] = f"TIFF series axis {symbol}"
    return (tuple(names), suggestions, evidence)


def _natural_sort_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split("(\\d+)", path.name)
    ]


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
