"""Xarray-free microscopy decoding and provenance tracing."""

from .models import (
    AxisDescriptor,
    InputDescriptor,
    RawMicroscopyData,
    ReaderTrace,
    ReadResult,
    SourceFileRecord,
    TraceEvent,
)
from .readers import (
    DEFAULT_MAX_DECODED_BYTES,
    MicroscopyReaderError,
    ReaderCapabilityError,
    ReaderDecodeError,
    ReaderResourceError,
    ReaderValidationError,
    read_tiff_source_traced,
)

__all__ = [
    "AxisDescriptor",
    "InputDescriptor",
    "RawMicroscopyData",
    "ReaderTrace",
    "ReadResult",
    "SourceFileRecord",
    "TraceEvent",
    "DEFAULT_MAX_DECODED_BYTES",
    "MicroscopyReaderError",
    "ReaderCapabilityError",
    "ReaderDecodeError",
    "ReaderResourceError",
    "ReaderValidationError",
    "read_tiff_source_traced",
]
