from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Recursively convert metadata to JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class AnalysisCache:
    """Persist and reload cached analysis arrays plus JSON metadata."""

    DEFAULT_FILENAME = "analysis_cache.npz"

    @classmethod
    def save(
        cls,
        path: Path,
        *,
        metadata: Dict[str, Any],
        arrays: Dict[str, np.ndarray],
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = dict(arrays)
        payload["_metadata_json"] = np.array(
            json.dumps(_to_jsonable(metadata), indent=2, sort_keys=True),
            dtype=str,
        )
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        path = cls.resolve_path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata_json = str(data["_metadata_json"])
            metadata = json.loads(metadata_json)
            arrays = {
                k: np.array(data[k])
                for k in data.files
                if k != "_metadata_json"
            }
        return (metadata, arrays)

    @classmethod
    def resolve_path(cls, path: Path) -> Path:
        path = Path(path)
        if path.is_dir():
            nested = path / "cache" / cls.DEFAULT_FILENAME
            if nested.exists():
                return nested
            return path / cls.DEFAULT_FILENAME
        return path
