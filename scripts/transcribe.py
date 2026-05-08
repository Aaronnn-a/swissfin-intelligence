"""CLI: transcribe a raw audio file into ``data/transcripts/<name>.json``.

Example:
    python scripts/transcribe.py --name ubs_q3_2025 --model small
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swissfin.config import settings  # noqa: E402
from swissfin.transcription.whisper_runner import WhisperTranscriber  # noqa: E402
from swissfin.utils.io import save_json  # noqa: E402

logger = logging.getLogger("swissfin.scripts.transcribe")


def _resolve_audio(name: str) -> Path:
    """Find an audio file in ``raw_dir`` whose stem matches ``name``."""
    raw_dir = settings.resolved_raw_dir
    matches = sorted(p for p in raw_dir.glob(f"{name}.*") if p.is_file())
    if not matches:
        raise FileNotFoundError(
            f"No audio file matching '{name}.*' under {raw_dir}. "
            "Run scripts/download_call.py first."
        )
    return matches[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a downloaded earnings call with Whisper.",
    )
    parser.add_argument("--name", required=True, help="Audio stem (matches data/raw/<name>.*).")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Whisper model size. Default: {settings.whisper_model}.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (de/en/fr/it). Default: auto-detect.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings.ensure_dirs()
    audio_path = _resolve_audio(args.name)
    transcriber = WhisperTranscriber(model_size=args.model, language=args.language)
    result = transcriber.transcribe(audio_path)

    out_path = settings.resolved_transcripts_dir / f"{args.name}.json"
    save_json(result.model_dump(), out_path)
    logger.info(
        "Transcribed %s → %s (%d segments, language=%s)",
        audio_path.name,
        out_path,
        len(result.segments),
        result.language,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
