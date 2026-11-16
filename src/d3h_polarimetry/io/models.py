"""Xarray-free models for decoded microscopy inputs and reader traces."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class AxisDescriptor:
    """Observed information about one storage dimension."""

    source_name: str
    index: int
    size: int
    suggested_name: str | None = None
    coordinates: tuple[Any, ...] | None = None
    unit: str | None = None
    evidence: str = "unresolved"
    confidence: str = "none"

    def __post_init__(self) -> None:
        if not self.source_name or not self.source_name.strip():
            raise ValueError("axis source_name must be non-empty")
        if self.index < 0:
            raise ValueError("axis index must be non-negative")
        if self.size <= 0:
            raise ValueError("axis size must be greater than zero")
        if self.coordinates is not None and len(self.coordinates) != self.size:
            raise ValueError(
                f"axis {self.source_name!r} has {len(self.coordinates)} coordinates, expected {self.size}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "index": self.index,
            "size": self.size,
            "suggested_name": self.suggested_name,
            "coordinates": list(self.coordinates)
            if self.coordinates is not None
            else None,
            "unit": self.unit,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class InputDescriptor:
    """Shape and metadata observed before scientific interpretation."""

    shape: tuple[int, ...]
    axes: tuple[AxisDescriptor, ...]
    source: Path | None = None
    source_kind: str = "array"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.shape or any((int(size) <= 0 for size in self.shape)):
            raise ValueError("input shape must contain positive dimensions")
        if len(self.axes) != len(self.shape):
            raise ValueError(
                "input descriptor must contain one axis per array dimension"
            )
        names = tuple((axis.source_name for axis in self.axes))
        if len(set(names)) != len(names):
            raise ValueError(f"input axis names must be unique: {names}")
        for index, (size, axis) in enumerate(zip(self.shape, self.axes)):
            if axis.index != index:
                raise ValueError(
                    f"axis {axis.source_name!r} has index {axis.index}, expected {index}"
                )
            if axis.size != int(size):
                raise ValueError(
                    f"axis {axis.source_name!r} has size {axis.size}, expected {size}"
                )

    @property
    def source_dims(self) -> tuple[str, ...]:
        return tuple((axis.source_name for axis in self.axes))

    def axis(self, source_name: str) -> AxisDescriptor:
        for axis in self.axes:
            if axis.source_name == source_name:
                return axis
        raise KeyError(source_name)


@dataclass(frozen=True)
class RawMicroscopyData:
    """Decoded NumPy pixels plus an uninterpreted input descriptor."""

    values: np.ndarray
    descriptor: InputDescriptor
    filenames: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("raw microscopy values must be a NumPy array")
        if self.values.dtype.hasobject:
            raise ValueError(
                "object-dtype microscopy arrays are not supported"
            )
        if tuple(self.values.shape) != tuple(self.descriptor.shape):
            raise ValueError(
                f"raw array shape {self.values.shape} does not match descriptor shape {self.descriptor.shape}"
            )


@dataclass(frozen=True)
class SourceFileRecord:
    """Stable identity for one decoded source file."""

    relative_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.relative_path:
            raise ValueError("source file relative_path must be non-empty")
        if self.size_bytes < 0 or self.mtime_ns < 0:
            raise ValueError(
                "source file size and modification time must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    """One ordered reader stage."""

    stage: str
    status: str = "ok"
    detail: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: value for key, value in payload.items() if value is not None
        }


@dataclass(frozen=True)
class ReaderTrace:
    """Versioned provenance for decoding performed before canonicalization."""

    source: str
    source_sha256: str
    adapter: str
    backend: str
    backend_versions: Mapping[str, str]
    selected_scene: str | None
    available_scenes: tuple[str, ...]
    source_dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    pixel_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mosaic_tile_positions: tuple[tuple[int, int], ...] = ()
    fallback_from: str | None = None
    fallback_reason: str | None = None
    warnings: tuple[str, ...] = ()
    events: tuple[TraceEvent, ...] = ()
    source_files: tuple[SourceFileRecord, ...] = ()
    selected_tiff_series: int | None = None
    observed_tiff_axes: str | None = None
    estimated_decoded_bytes: int | None = None
    max_decoded_bytes: int | None = None
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "adapter": self.adapter,
            "backend": self.backend,
            "backend_versions": dict(self.backend_versions),
            "selected_scene": self.selected_scene,
            "available_scenes": list(self.available_scenes),
            "source_dims": list(self.source_dims),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "pixel_sha256": self.pixel_sha256,
            "metadata": dict(self.metadata),
            "mosaic_tile_positions": [
                list(item) for item in self.mosaic_tile_positions
            ],
            "fallback_from": self.fallback_from,
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
            "events": [event.to_dict() for event in self.events],
            "source_files": [record.to_dict() for record in self.source_files],
            "selected_tiff_series": self.selected_tiff_series,
            "observed_tiff_axes": self.observed_tiff_axes,
            "estimated_decoded_bytes": self.estimated_decoded_bytes,
            "max_decoded_bytes": self.max_decoded_bytes,
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return destination


@dataclass(frozen=True)
class ReadResult:
    """Decoded raw data and its format-reader trace."""

    raw: RawMicroscopyData
    trace: ReaderTrace

    def __post_init__(self) -> None:
        if tuple(self.raw.values.shape) != tuple(self.trace.shape):
            raise ValueError(
                "reader trace shape does not match decoded pixels"
            )
        if str(self.raw.values.dtype) != self.trace.dtype:
            raise ValueError(
                "reader trace dtype does not match decoded pixels"
            )
        if self.raw.descriptor.source_dims != tuple(self.trace.source_dims):
            raise ValueError(
                "reader trace dimensions do not match the input descriptor"
            )
