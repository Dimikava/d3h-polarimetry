"""Format-specific microscopy readers with explicit, reproducible tracing."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
from math import prod
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable
import numpy as np
import tifffile
from .models import (
    AxisDescriptor,
    InputDescriptor,
    RawMicroscopyData,
    ReaderTrace,
    ReadResult,
    SourceFileRecord,
    TraceEvent,
)

DEFAULT_MAX_DECODED_BYTES = 8 * 1024**3


class MicroscopyReaderError(RuntimeError):
    """Base error for traced microscopy decoding."""


class ReaderCapabilityError(MicroscopyReaderError):
    """Raised when an installed reader cannot support an input feature."""


class ReaderValidationError(MicroscopyReaderError):
    """Raised when decoded pixels and observed metadata are inconsistent."""


class ReaderDecodeError(MicroscopyReaderError):
    """Raised when an identified TIFF source cannot be decoded."""


class ReaderResourceError(MicroscopyReaderError):
    """Raised before decoding when an input exceeds the configured size limit."""


_SUGGESTIONS = {
    "T": "time",
    "C": "channel",
    "Z": "z",
    "Y": "y",
    "X": "x",
    "M": "mosaic_tile",
    "S": "sample",
}


def read_tiff_source_traced(
    source: str | Path,
    *,
    sort_by: str = "name",
    tiff_series: int | None = None,
    max_frames: int | None = None,
    max_decoded_bytes: int | None = DEFAULT_MAX_DECODED_BYTES,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> ReadResult:
    """Strictly decode one TIFF file or a flat folder of TIFF files."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    _validate_resource_options(
        max_frames=max_frames, max_decoded_bytes=max_decoded_bytes
    )
    if path.is_dir():
        return _read_tiff_directory_strict(
            path,
            sort_by=sort_by,
            tiff_series=tiff_series,
            max_frames=max_frames,
            max_decoded_bytes=max_decoded_bytes,
            on_event=on_event,
        )
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ReaderCapabilityError(
            f"Expected a .tif or .tiff input, got {path.name}"
        )
    if max_frames is not None:
        raise ReaderValidationError(
            "max_frames is supported only for TIFF folders"
        )
    return _read_tiff_file_strict(
        path,
        tiff_series=tiff_series,
        max_decoded_bytes=max_decoded_bytes,
        on_event=on_event,
    )


@dataclass(frozen=True)
class _TiffSeriesInfo:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    axes: str
    series_index: int
    series_count: int
    is_ome: bool

    @property
    def decoded_bytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


def _read_tiff_file_strict(
    path: Path,
    *,
    tiff_series: int | None,
    max_decoded_bytes: int | None,
    on_event: Callable[[TraceEvent], None] | None,
) -> ReadResult:
    events: list[TraceEvent] = []

    def record(event: TraceEvent) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    record(TraceEvent("discover", detail=path.name))
    inspected = perf_counter()
    info = _inspect_tiff(path, tiff_series)
    record(
        TraceEvent(
            "inspect-header",
            duration_ms=_elapsed_ms(inspected),
            detail=f"series={info.series_index} axes={info.axes} shape={info.shape} dtype={info.dtype}",
        )
    )
    limit = _normalized_limit(max_decoded_bytes)
    _enforce_decoded_limit(info.decoded_bytes, limit, path)
    record(
        TraceEvent(
            "preflight",
            detail=f"decoded_bytes={info.decoded_bytes} limit={limit}",
        )
    )
    hashed = perf_counter()
    source_record = _source_file_record(path, path.parent)
    record(TraceEvent("hash-source", duration_ms=_elapsed_ms(hashed)))
    decoded = perf_counter()
    try:
        with tifffile.TiffFile(path) as tif:
            values = np.asarray(tif.series[info.series_index].asarray())
    except Exception as exc:
        raise ReaderDecodeError(
            f"Failed to decode TIFF {path}: {exc}"
        ) from exc
    record(TraceEvent("read-pixels", duration_ms=_elapsed_ms(decoded)))
    _validate_decoded_array(values, info, path)
    descriptor = InputDescriptor(
        shape=tuple((int(size) for size in values.shape)),
        axes=_tiff_axis_descriptors(info.axes, values.shape),
        source=path,
        source_kind="tiff-file",
        metadata={
            "tiff_series_count": info.series_count,
            "selected_tiff_series": info.series_index,
            "tiff_axes": info.axes,
            "ome_metadata_present": info.is_ome,
        },
    )
    raw = RawMicroscopyData(values, descriptor, (path.name,))
    record(
        TraceEvent(
            "validate-raw", detail=f"{info.axes} {values.shape} {values.dtype}"
        )
    )
    pixels_hashed = perf_counter()
    pixel_sha256 = _array_sha256(values)
    record(TraceEvent("hash-pixels", duration_ms=_elapsed_ms(pixels_hashed)))
    trace = ReaderTrace(
        source=str(path.resolve()),
        source_sha256=source_record.sha256,
        adapter="tiff-file",
        backend="tifffile",
        backend_versions=_backend_versions("tifffile"),
        selected_scene=None,
        available_scenes=(),
        source_dims=descriptor.source_dims,
        shape=descriptor.shape,
        dtype=str(values.dtype),
        pixel_sha256=pixel_sha256,
        metadata=descriptor.metadata,
        events=tuple(events),
        source_files=(source_record,),
        selected_tiff_series=info.series_index,
        observed_tiff_axes=info.axes,
        estimated_decoded_bytes=info.decoded_bytes,
        max_decoded_bytes=limit,
    )
    return ReadResult(raw, trace)


def _read_tiff_directory_strict(
    path: Path,
    *,
    sort_by: str,
    tiff_series: int | None,
    max_frames: int | None,
    max_decoded_bytes: int | None,
    on_event: Callable[[TraceEvent], None] | None,
) -> ReadResult:
    events: list[TraceEvent] = []

    def record(event: TraceEvent) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    discovered = perf_counter()
    files = _discover_tiff_files(path, sort_by=sort_by)
    if max_frames is not None:
        files = files[:max_frames]
    if not files:
        raise ReaderCapabilityError(f"No TIFF images found in {path}")
    record(
        TraceEvent(
            "discover",
            duration_ms=_elapsed_ms(discovered),
            detail=f"{len(files)} TIFF files",
        )
    )
    inspected = perf_counter()
    infos = tuple((_inspect_tiff(item, tiff_series) for item in files))
    reference = infos[0]
    for item, info in zip(files[1:], infos[1:]):
        if (info.shape, info.dtype, info.axes) != (
            reference.shape,
            reference.dtype,
            reference.axes,
        ):
            raise ReaderValidationError(
                f"TIFF {item} has series shape/dtype/axes {info.shape}/{info.dtype}/{info.axes}, expected {reference.shape}/{reference.dtype}/{reference.axes}"
            )
    record(
        TraceEvent(
            "inspect-header",
            duration_ms=_elapsed_ms(inspected),
            detail=f"series={reference.series_index} axes={reference.axes} frame_shape={reference.shape} dtype={reference.dtype}",
        )
    )
    decoded_bytes = reference.decoded_bytes * len(files)
    limit = _normalized_limit(max_decoded_bytes)
    _enforce_decoded_limit(decoded_bytes, limit, path)
    record(
        TraceEvent(
            "preflight", detail=f"decoded_bytes={decoded_bytes} limit={limit}"
        )
    )
    hashed = perf_counter()
    source_files = tuple((_source_file_record(item, path) for item in files))
    source_sha256 = _folder_source_sha256(source_files)
    record(TraceEvent("hash-source", duration_ms=_elapsed_ms(hashed)))
    values = np.empty((len(files), *reference.shape), dtype=reference.dtype)
    decoded = perf_counter()
    for index, (item, info) in enumerate(zip(files, infos)):
        try:
            with tifffile.TiffFile(item) as tif:
                frame = np.asarray(tif.series[info.series_index].asarray())
        except Exception as exc:
            raise ReaderDecodeError(
                f"Failed to decode TIFF {item}: {exc}"
            ) from exc
        _validate_decoded_array(frame, info, item)
        values[index] = frame
    record(TraceEvent("read-pixels", duration_ms=_elapsed_ms(decoded)))
    inner_axes = _tiff_axis_descriptors(
        reference.axes, reference.shape, index_offset=1
    )
    axes = (
        AxisDescriptor(
            "sequence",
            0,
            len(files),
            coordinates=tuple((item.name for item in files)),
            evidence="deterministic TIFF directory order",
            confidence="none",
        ),
        *inner_axes,
    )
    descriptor = InputDescriptor(
        tuple((int(size) for size in values.shape)),
        axes,
        path,
        "tiff-directory",
        metadata={
            "tiff_series_count": reference.series_count,
            "selected_tiff_series": reference.series_index,
            "tiff_axes": reference.axes,
        },
    )
    raw = RawMicroscopyData(
        values, descriptor, tuple((item.name for item in files))
    )
    record(
        TraceEvent(
            "validate-raw",
            detail=f"{descriptor.source_dims} {values.shape} {values.dtype}",
        )
    )
    pixels_hashed = perf_counter()
    pixel_sha256 = _array_sha256(values)
    record(TraceEvent("hash-pixels", duration_ms=_elapsed_ms(pixels_hashed)))
    trace = ReaderTrace(
        source=str(path.resolve()),
        source_sha256=source_sha256,
        adapter="tiff-directory",
        backend="tifffile",
        backend_versions=_backend_versions("tifffile"),
        selected_scene=None,
        available_scenes=(),
        source_dims=descriptor.source_dims,
        shape=descriptor.shape,
        dtype=str(values.dtype),
        pixel_sha256=pixel_sha256,
        metadata=descriptor.metadata,
        events=tuple(events),
        source_files=source_files,
        selected_tiff_series=reference.series_index,
        observed_tiff_axes=reference.axes,
        estimated_decoded_bytes=decoded_bytes,
        max_decoded_bytes=limit,
    )
    return ReadResult(raw, trace)


def _validate_resource_options(
    *, max_frames: int | None, max_decoded_bytes: int | None
) -> None:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if max_decoded_bytes is not None and max_decoded_bytes < 0:
        raise ValueError("max_decoded_bytes must be non-negative or None")


def _normalized_limit(max_decoded_bytes: int | None) -> int | None:
    return None if max_decoded_bytes in {None, 0} else int(max_decoded_bytes)


def _enforce_decoded_limit(
    estimated: int, limit: int | None, source: Path
) -> None:
    if limit is not None and estimated > limit:
        raise ReaderResourceError(
            f"Decoded TIFF data for {source} is estimated at {estimated / 1024**3:.3f} GiB, above the configured {limit / 1024**3:.3f} GiB limit"
        )


def _discover_tiff_files(path: Path, *, sort_by: str) -> list[Path]:
    files = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}
    ]
    if sort_by == "name":
        files.sort(
            key=lambda item: (
                _natural_sort_key(item),
                item.name.casefold(),
                item.name,
            )
        )
    elif sort_by == "date":
        files.sort(
            key=lambda item: (
                item.stat().st_mtime_ns,
                _natural_sort_key(item),
                item.name.casefold(),
                item.name,
            )
        )
    else:
        raise ValueError("sort_by must be 'name' or 'date'")
    return files


def _inspect_tiff(path: Path, requested_series: int | None) -> _TiffSeriesInfo:
    try:
        with tifffile.TiffFile(path) as tif:
            count = len(tif.series)
            if count == 0:
                raise ReaderValidationError(
                    f"TIFF {path} contains no image series"
                )
            if requested_series is None:
                if count != 1:
                    raise ReaderValidationError(
                        f"TIFF {path} contains {count} series; select one explicitly"
                    )
                index = 0
            else:
                index = int(requested_series)
                if index < 0 or index >= count:
                    raise ReaderValidationError(
                        f"TIFF series {index} is unavailable for {path}; valid range is 0..{count - 1}"
                    )
            series = tif.series[index]
            shape = tuple((int(size) for size in series.shape))
            dtype = np.dtype(series.dtype)
            axes = str(series.axes or "")
            if not shape or any((size <= 0 for size in shape)):
                raise ReaderValidationError(
                    f"TIFF {path} has invalid series shape {shape}"
                )
            if dtype.hasobject:
                raise ReaderValidationError(
                    f"TIFF {path} uses unsupported object dtype"
                )
            if len(axes) != len(shape):
                axes = "Q" * len(shape)
            return _TiffSeriesInfo(
                shape, dtype, axes, index, count, bool(tif.is_ome)
            )
    except ReaderValidationError:
        raise
    except Exception as exc:
        raise ReaderDecodeError(
            f"Failed to inspect TIFF {path}: {exc}"
        ) from exc


def _validate_decoded_array(
    values: np.ndarray, info: _TiffSeriesInfo, path: Path
) -> None:
    if tuple(values.shape) != info.shape:
        raise ReaderValidationError(
            f"TIFF {path} decoded to shape {values.shape}, expected {info.shape}"
        )
    if values.dtype != info.dtype:
        raise ReaderValidationError(
            f"TIFF {path} decoded to dtype {values.dtype}, expected {info.dtype}"
        )
    if values.dtype.hasobject:
        raise ReaderValidationError(
            f"TIFF {path} decoded to unsupported object dtype"
        )


def _tiff_axis_descriptors(
    axes_text: str, shape: tuple[int, ...], *, index_offset: int = 0
) -> tuple[AxisDescriptor, ...]:
    used: dict[str, int] = {}
    descriptors: list[AxisDescriptor] = []
    for local_index, (symbol, size) in enumerate(zip(axes_text, shape)):
        base = _SUGGESTIONS.get(symbol, symbol.lower() or f"dim_{local_index}")
        used[base] = used.get(base, 0) + 1
        name = base if used[base] == 1 else f"{base}_{used[base]}"
        descriptors.append(
            AxisDescriptor(
                name,
                local_index + index_offset,
                int(size),
                suggested_name=_SUGGESTIONS.get(symbol),
                evidence=f"tifffile series axis {symbol}",
                confidence="high" if symbol in _SUGGESTIONS else "none",
            )
        )
    return tuple(descriptors)


def _source_file_record(path: Path, root: Path) -> SourceFileRecord:
    stat = path.stat()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    return SourceFileRecord(
        relative, stat.st_size, stat.st_mtime_ns, _file_sha256(path)
    )


def _folder_source_sha256(records: tuple[SourceFileRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(record.size_bytes).encode("ascii"))
        digest.update(b"\x00")
        digest.update(bytes.fromhex(record.sha256))
    return digest.hexdigest()


def _backend_versions(*modules: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for module in modules:
        package = module
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    if values.flags.c_contiguous:
        raw = memoryview(values).cast("B")
        for offset in range(0, len(raw), 1024 * 1024):
            digest.update(raw[offset : offset + 1024 * 1024])
    else:
        for index in range(values.shape[0]):
            digest.update(np.ascontiguousarray(values[index]).view(np.uint8))
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


def _natural_sort_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split("(\\d+)", path.name)
    ]
