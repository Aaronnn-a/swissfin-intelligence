"""CLI: download an earnings-call audio file into ``data/raw/``.

Example:
    python scripts/download_call.py \
        --url https://www.youtube.com/watch?v=XXXX \
        --name ubs_q3_2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the `swissfin` package importable when running the script directly
# (i.e. without `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swissfin.audio.downloader import download_audio  # noqa: E402
from swissfin.config import settings  # noqa: E402

logger = logging.getLogger("swissfin.scripts.download_call")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download earnings-call audio into the raw data directory.",
    )
    parser.add_argument("--url", required=True, help="Source URL (YouTube, IR site, podcast).")
    parser.add_argument(
        "--name",
        required=True,
        help="Filename stem, e.g. 'ubs_q3_2025'. The extension is appended automatically.",
    )
    parser.add_argument(
        "--format", default="m4a", help="Target audio format (m4a, mp3, wav). Default: m4a."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings.ensure_dirs()
    out = download_audio(
        url=args.url,
        output_dir=settings.resolved_raw_dir,
        filename_stem=args.name,
        audio_format=args.format,
    )
    logger.info("Done. File saved to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
