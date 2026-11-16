"""
TIFF Analyzer - Data Reader Module

This module provides classes for loading microscopy data:
- DatParser: Parses metadata from .dat configuration files.
- ImageLoader: Loads and stacks TIFF image sequences.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .io import (
    DEFAULT_MAX_DECODED_BYTES,
    MicroscopyReaderError,
    ReaderCapabilityError,
    read_tiff_source_traced,
)

logger = logging.getLogger(__name__)


class DatParseError(Exception):
    """Raised when DAT file parsing fails."""

    pass


class ImageLoadError(Exception):
    """Raised when image loading fails."""

    pass


class DatParser:
    """
    Parses metadata from microscopy .dat configuration files.

    The parser handles section-based key-value files, automatically detecting
    section headers and extracting numeric values from strings.

    Attributes:
        dat_path: Path to the .dat file.
        metadata: Parsed metadata dictionary, organized by section.

    Example:
        >>> parser = DatParser("experiment_Info.dat")
        >>> meta = parser.get_metadata()
        >>> print(meta['Scan Settings Kami']['Number of lines'])
        1200.0
    """

    def __init__(self, dat_path: Union[str, Path]) -> None:
        """
        Initialize the parser and load metadata.

        Args:
            dat_path: Path to the .dat file.

        Raises:
            FileNotFoundError: If the file does not exist.
            DatParseError: If the file cannot be parsed.
        """
        self.dat_path = Path(dat_path)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse the DAT file into sections and key-value pairs."""
        if not self.dat_path.exists():
            raise FileNotFoundError(f"DAT file not found: {self.dat_path}")
        try:
            with open(self.dat_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except IOError as e:
            raise DatParseError(f"Failed to read DAT file: {e}") from e
        current_section = "Global"
        self.metadata[current_section] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":") and ":" in line:
                parts = line.split(":")
                if len(parts) == 2 and (not parts[1].strip()):
                    current_section = parts[0].strip()
                    self.metadata[current_section] = {}
                    logger.debug(f"Found section: {current_section}")
                    continue
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value_str = parts[1].strip()
                self.metadata[current_section][key] = self._parse_value(
                    value_str
                )
        logger.info(
            f"Parsed {len(self.metadata)} sections from {self.dat_path.name}"
        )

    def get_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parsed metadata.

        Returns:
            Dictionary of sections, each containing key-value pairs.
        """
        return self.metadata

    @staticmethod
    def _parse_value(value_str: str) -> Any:
        """
        Parse leading numeric metadata values while preserving free text.

        Examples:
            "1200" -> 1200.0
            "150 um" -> 150.0
            "-1,25 V" -> -1.25
            "mode 2" -> "mode 2"
        """
        numeric_match = re.match(
            "\\s*([-+]?(?:\\d+(?:[.,]\\d*)?|[.,]\\d+)(?:[eE][-+]?\\d+)?)",
            value_str,
        )
        if not numeric_match:
            return value_str
        num_str = numeric_match.group(1).replace(",", ".")
        try:
            return float(num_str)
        except ValueError:
            return value_str

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a specific section from the metadata.

        Args:
            section_name: Name of the section to retrieve.

        Returns:
            Dictionary of key-value pairs for the section.

        Raises:
            KeyError: If the section does not exist.
        """
        if section_name not in self.metadata:
            raise KeyError(f"Section '{section_name}' not found in metadata")
        return self.metadata[section_name]

    def get_scale_info(
        self,
        scan_section: str = "Scan Settings Kami",
        field_key: str = "Scan field X",
        lines_key: str = "Number of lines",
    ) -> Tuple[float, float, float]:
        """
        Extract physical scale information from DAT metadata.

        Args:
            scan_section: Section name containing scan settings.
            field_key: Key for the scan field width (microns).
            lines_key: Key for number of pixels/lines.

        Returns:
            Tuple of (scan_width_microns, num_pixels, microns_per_pixel).

        Raises:
            KeyError: If required keys are not found.
            ValueError: If values are not numeric.
        """
        section = self.get_section(scan_section)
        scan_width = section.get(field_key)
        if scan_width is None:
            raise KeyError(
                f"'{field_key}' not found in section '{scan_section}'"
            )
        scan_width = float(scan_width)
        num_pixels = section.get(lines_key)
        if num_pixels is None:
            raise KeyError(
                f"'{lines_key}' not found in section '{scan_section}'"
            )
        num_pixels = float(num_pixels)
        if num_pixels <= 0:
            raise ValueError(f"Invalid pixel count: {num_pixels}")
        microns_per_pixel = scan_width / num_pixels
        logger.info(
            f"Scale info: {scan_width:.2f} μm across {int(num_pixels)} px → {microns_per_pixel:.4f} μm/px"
        )
        return (scan_width, num_pixels, microns_per_pixel)


class ImageLoader:
    """
    Loads TIFF image sequences from a directory.

    Supports sorting by modification date or filename (with natural number ordering).

    Attributes:
        folder_path: Path to the image directory.

    Example:
        >>> loader = ImageLoader("experiment/tif1")
        >>> stack, filenames = loader.load_images(sort_by='name')
        >>> print(stack.shape)  # (360, 1200, 1200)
    """

    def __init__(self, folder_path: Union[str, Path]) -> None:
        """
        Initialize the loader.

        Args:
            folder_path: Path to directory containing TIFF files.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

    def load_images(
        self,
        sort_by: str = "date",
        num_frames: Optional[int] = None,
        max_decoded_bytes: Optional[int] = DEFAULT_MAX_DECODED_BYTES,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Load TIFF images from the folder.

        Args:
            sort_by: Sorting method - 'date' (modification time) or 'name' (natural sort).

        Returns:
            Tuple of (image_stack, filenames):
                - image_stack: NumPy array of shape (N, H, W) or None if no images.
                - filenames: List of loaded filenames.

        Raises:
            ImageLoadError: If images cannot be stacked (different sizes).
        """
        if not self.folder_path.is_dir():
            raise ImageLoadError(
                f"ImageLoader requires a TIFF directory: {self.folder_path}"
            )
        try:
            result = read_tiff_source_traced(
                self.folder_path,
                sort_by=sort_by,
                max_frames=num_frames,
                max_decoded_bytes=max_decoded_bytes,
            )
        except ReaderCapabilityError as exc:
            if "No TIFF images found" in str(exc):
                logger.warning("No TIFF files found in %s", self.folder_path)
                return (None, [])
            raise ImageLoadError(str(exc)) from exc
        except MicroscopyReaderError as exc:
            raise ImageLoadError(str(exc)) from exc
        logger.info(
            "Loaded stack: %s, dtype=%s",
            result.raw.values.shape,
            result.raw.values.dtype,
        )
        return (result.raw.values, list(result.raw.filenames))

    @staticmethod
    def _natural_sort_key(path: Path) -> List[Union[int, str]]:
        """
        Generate a key for natural sorting (e.g., R1, R2, R10 instead of R1, R10, R2).

        Args:
            path: Path object to generate key for.

        Returns:
            List of string/int segments for sorting.
        """
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("(\\d+)", str(path))
        ]
