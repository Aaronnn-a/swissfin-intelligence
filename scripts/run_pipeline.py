"""End-to-end pipeline: URL → audio → transcript → sentiment JSON.

Example:
    python scripts/run_pipeline.py \
        --url https://www.youtube.com/watch?v=XXXX \
        --name ubs_q3_2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swissfin.audio.downloader import download_audio  # noqa: E402
from swissfin.config import settings  # noqa: E402
from swissfin.sentiment.analyzer import SentimentAnalyzer  # noqa: E402
from swissfin.transcription.whisper_runner import WhisperTranscriber  # noqa: E402
from swissfin.utils.io import save_json  # noqa: E402

logger = logging.getLogger("swissfin.scripts.run_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Phase-1 pipeline: download → transcribe → sentiment.",
    )
    parser.add_argument("--url", required=True, help="Earnings-call source URL.")
    parser.add_argument("--name", required=True, help="Filename stem for all artefacts.")
    parser.add_argument("--whisper-model", default=None, help="Override Whisper model size.")
    parser.add_argument("--sentiment-model", default=None, help="Override sentiment model.")
    parser.add_argument(
        "--language",
        default=None,
        help="Force Whisper language code (de/en/fr/it).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing audio file in data/raw/ instead of re-downloading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings.ensure_dirs()

    # 1) Download
    if args.skip_download:
        candidates = sorted(settings.resolved_raw_dir.glob(f"{args.name}.*"))
        if not candidates:
            raise FileNotFoundError(
                f"--skip-download set but no audio matching '{args.name}.*' "
                f"in {settings.resolved_raw_dir}."
            )
        audio_path = candidates[0]
        logger.info("Skipping download; reusing %s", audio_path)
    else:
        audio_path = download_audio(
            url=args.url,
            output_dir=settings.resolved_raw_dir,
            filename_stem=args.name,
        )

    # 2) Transcribe
    transcriber = WhisperTranscriber(
        model_size=args.whisper_model, language=args.language
    )
    transcription = transcriber.transcribe(audio_path)
    transcript_path = settings.resolved_transcripts_dir / f"{args.name}.json"
    save_json(transcription.model_dump(), transcript_path)
    logger.info("Saved transcript → %s", transcript_path)

    # 3) Sentiment
    analyzer = SentimentAnalyzer(model_name=args.sentiment_model)
    sentiments = analyzer.analyze_segments(
        [s.model_dump() for s in transcription.segments]
    )
    payload = {
        "name": args.name,
        "url": args.url,
        "language": transcription.language,
        "whisper_model": transcriber.model_size,
        "sentiment_model": analyzer.model_name,
        "segments": [s.model_dump() for s in sentiments],
    }
    out_path = settings.resolved_outputs_dir / f"{args.name}.sentiment.json"
    save_json(payload, out_path)
    logger.info(
        "Pipeline complete: %d segments scored → %s", len(sentiments), out_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
