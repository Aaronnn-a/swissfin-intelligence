"""Split long earnings-call audio into Whisper-friendly chunks.

Whisper internally limits each forward-pass to 30 s windows but performs much
better with explicit, semantically-meaningful chunks of a few minutes that align
with silence boundaries between speakers. This module produces such chunks
using :mod:`pydub` and falls back to fixed-length splitting when no usable
silence is found.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_SILENCE_MS = 700  # Silence shorter than this is treated as a breath, not a break.
_KEEP_SILENCE_MS = 250  # Padding kept around chunk boundaries to avoid clipping words.


def split_audio(
    file_path: Path,
    chunk_seconds: int = 600,
    *,
    silence_thresh_dbfs: int = -40,
    output_dir: Path | None = None,
) -> list[Path]:
    """Split ``file_path`` into chunks of roughly ``chunk_seconds`` length.

    The algorithm prefers boundaries where the audio drops below
    ``silence_thresh_dbfs``. If no such boundary lands close to the target
    chunk length we fall back to a hard cut to keep memory usage bounded.

    Args:
        file_path: Source audio file (any format pydub/FFmpeg can decode).
        chunk_seconds: Target chunk length in seconds. Actual chunks may be
            slightly longer if a clean silence boundary is nearby.
        silence_thresh_dbfs: Loudness threshold for what counts as silence.
            ``-40`` dBFS works well for most teleconference audio.
        output_dir: Where to write the chunks. Defaults to a sibling
            ``<file_path>.chunks/`` directory.

    Returns:
        Ordered list of chunk file paths (``<stem>_000.<ext>``,
        ``<stem>_001.<ext>``, …).
    """
    from pydub import AudioSegment, silence

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    output_dir = Path(output_dir) if output_dir else file_path.with_suffix("").parent / (
        file_path.stem + ".chunks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)
    chunk_ms = chunk_seconds * 1000
    logger.info(
        "Splitting %s (%.1fs) into ~%ds chunks", file_path.name, total_ms / 1000, chunk_seconds
    )

    if total_ms <= chunk_ms:
        out = output_dir / f"{file_path.stem}_000{file_path.suffix}"
        audio.export(out, format=file_path.suffix.lstrip("."))
        logger.info("Audio is shorter than chunk size; produced single chunk %s", out)
        return [out]

    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=_MIN_SILENCE_MS,
        silence_thresh=silence_thresh_dbfs,
    )
    boundaries = _pick_boundaries(silent_ranges, total_ms, chunk_ms)
    logger.debug("Boundary cuts (ms): %s", boundaries)

    chunks: list[Path] = []
    suffix = file_path.suffix.lstrip(".") or "m4a"
    prev = 0
    for idx, cut in enumerate(boundaries + [total_ms]):
        start = max(prev - _KEEP_SILENCE_MS, 0)
        end = min(cut + _KEEP_SILENCE_MS, total_ms)
        segment = audio[start:end]
        out = output_dir / f"{file_path.stem}_{idx:03d}.{suffix}"
        segment.export(out, format=suffix)
        chunks.append(out)
        prev = cut

    logger.info("Wrote %d chunks → %s", len(chunks), output_dir)
    return chunks


def _pick_boundaries(
    silent_ranges: list[list[int]], total_ms: int, chunk_ms: int
) -> list[int]:
    """Pick mid-points of silences closest to each multiple of ``chunk_ms``."""
    targets = list(range(chunk_ms, total_ms, chunk_ms))
    if not targets:
        return []
    if not silent_ranges:
        return targets

    midpoints = [(s + e) // 2 for s, e in silent_ranges]
    cuts: list[int] = []
    for target in targets:
        nearest = min(midpoints, key=lambda m: abs(m - target))
        # Only snap to silence if it's within +/- 25% of the target window.
        if abs(nearest - target) <= chunk_ms * 0.25:
            cuts.append(nearest)
        else:
            cuts.append(target)
    # Ensure monotonically increasing (silence snapping can cause duplicates).
    cuts = sorted({c for c in cuts if 0 < c < total_ms})
    return cuts
