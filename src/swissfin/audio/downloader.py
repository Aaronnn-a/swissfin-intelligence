"""Earnings-call audio acquisition via :mod:`yt_dlp`.

This module is intentionally a thin wrapper: yt-dlp is the right tool for the
job (handles YouTube, generic IR pages, podcast feeds), and we only standardize
the output naming and audio format so the rest of the pipeline can rely on a
consistent on-disk layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_AUDIO_FORMAT = "m4a"
_DEFAULT_AUDIO_QUALITY = "192"


def download_audio(
    url: str,
    output_dir: Path,
    filename_stem: str,
    *,
    audio_format: str = _DEFAULT_AUDIO_FORMAT,
    audio_quality: str = _DEFAULT_AUDIO_QUALITY,
) -> Path:
    """Download the audio track from ``url`` into ``output_dir``.

    The resulting file is mono, 192 kbps by default, named
    ``<filename_stem>.<audio_format>``. yt-dlp transcodes through FFmpeg, which
    must be available on ``PATH`` at runtime.

    Args:
        url: Source URL (YouTube, Vimeo, generic IR site, etc.).
        output_dir: Destination directory; created if missing.
        filename_stem: File name without extension. Used as the on-disk handle
            for the rest of the pipeline (``ubs_q3_2025`` → ``ubs_q3_2025.m4a``).
        audio_format: Target codec/extension. Anything supported by FFmpeg.
        audio_quality: Audio bitrate hint passed to yt-dlp's postprocessor.

    Returns:
        Absolute path to the saved audio file.

    Raises:
        RuntimeError: If yt-dlp completes without producing the expected file.
    """
    # Imported lazily so unit tests can stub the module without paying the
    # yt-dlp import cost (and so `pip install` failures remain local).
    from yt_dlp import YoutubeDL

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{filename_stem}.{audio_format}"

    ydl_opts: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{filename_stem}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": audio_quality,
            }
        ],
        # Force mono — sentiment models don't need stereo and Whisper
        # downmixes anyway, so save half the disk space.
        "postprocessor_args": ["-ac", "1"],
    }

    logger.info("Downloading audio: url=%s → %s", url, target)
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not target.exists():
        # Some extractors produce a slightly different extension; fall back to
        # any file in the directory matching the stem.
        candidates = sorted(output_dir.glob(f"{filename_stem}.*"))
        if not candidates:
            raise RuntimeError(
                f"yt-dlp finished but no file matching '{filename_stem}.*' "
                f"was produced in {output_dir}."
            )
        target = candidates[0]
        logger.warning("Expected %s.%s, fell back to %s", filename_stem, audio_format, target)

    logger.info("Audio saved: %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target
