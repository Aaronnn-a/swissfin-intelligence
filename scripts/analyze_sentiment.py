"""CLI: score sentiment for a transcript and write results to ``data/outputs/``.

Example:
    python scripts/analyze_sentiment.py --name ubs_q3_2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swissfin.config import settings  # noqa: E402
from swissfin.sentiment.analyzer import SentimentAnalyzer  # noqa: E402
from swissfin.utils.io import load_transcript, save_json  # noqa: E402

logger = logging.getLogger("swissfin.scripts.analyze_sentiment")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-segment sentiment analysis on a saved transcript.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Transcript stem under data/transcripts/ (e.g. 'ubs_q3_2025').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"HF sentiment model id. Default: {settings.sentiment_model}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Inference batch size. Default: {settings.sentiment_batch_size}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings.ensure_dirs()

    transcript_path = settings.resolved_transcripts_dir / f"{args.name}.json"
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {transcript_path}. Run scripts/transcribe.py first."
        )

    transcript = load_transcript(transcript_path)
    analyzer = SentimentAnalyzer(model_name=args.model, batch_size=args.batch_size)
    results = analyzer.analyze_segments(transcript["segments"])

    payload = {
        "name": args.name,
        "language": transcript.get("language"),
        "sentiment_model": analyzer.model_name,
        "segments": [r.model_dump() for r in results],
    }
    out_path = settings.resolved_outputs_dir / f"{args.name}.sentiment.json"
    save_json(payload, out_path)
    logger.info("Wrote %d scored segments → %s", len(results), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
