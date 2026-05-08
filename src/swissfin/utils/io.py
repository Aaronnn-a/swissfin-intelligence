"""Lightweight JSON / transcript I/O helpers used across the pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_json(data: Any, path: Path, *, indent: int = 2) -> Path:
    """Serialize ``data`` as UTF-8 JSON to ``path``.

    Args:
        data: Any JSON-serializable Python object. Pydantic models should be
            converted via ``model.model_dump()`` before calling this helper.
        path: Destination file path. Parent directories are created on demand.
        indent: Indentation level for the JSON output. ``0`` produces compact
            output; the default is human-readable.

    Returns:
        The resolved path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    logger.debug("Wrote JSON: %s (%d bytes)", path, path.stat().st_size)
    return path


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON document from ``path``.

    Args:
        path: Source file path.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If ``path`` does not contain valid JSON.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_transcript(path: Path) -> dict[str, Any]:
    """Load a transcript JSON produced by :class:`WhisperTranscriber`.

    The expected schema is::

        {"text": str, "language": str | None,
         "segments": [{"start": float, "end": float, "text": str}, ...]}

    Plain ``.txt`` files are also accepted; they are wrapped into a single
    pseudo-segment spanning the full call.

    Args:
        path: Path to a ``.json`` or ``.txt`` transcript file.

    Returns:
        A dict with ``text``, ``segments``, ``language`` keys.
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        return {
            "text": text,
            "language": None,
            "segments": [{"start": 0.0, "end": 0.0, "text": text}],
        }
    payload = load_json(path)
    if not isinstance(payload, dict) or "segments" not in payload:
        raise ValueError(f"{path} is not a valid transcript JSON document.")
    return payload
