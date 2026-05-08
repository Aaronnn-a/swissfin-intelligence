"""Centralised, environment-driven configuration for SwissFin Intelligence.

All runtime knobs live here. Values are loaded from environment variables (or a
local ``.env`` file) via :mod:`pydantic_settings`. Importing :data:`settings`
also has the side-effect of configuring root logging exactly once, so library
modules can simply do ``logger = logging.getLogger(__name__)`` and rely on
formatting being in place.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root (``swissfin-intelligence/``)."""

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
WhisperModelSize = Literal["tiny", "base", "small", "medium", "large-v3", "turbo"]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / ``.env``.

    Attributes are immutable once instantiated. Override any field by setting
    the corresponding ``SWISSFIN_*`` environment variable, e.g.::

        SWISSFIN_WHISPER_MODEL=medium python scripts/transcribe.py --name foo
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="SWISSFIN_",
        extra="ignore",
        frozen=True,
    )

    # --- Paths ----------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_dir: Path | None = Field(default=None)
    transcripts_dir: Path | None = Field(default=None)
    outputs_dir: Path | None = Field(default=None)

    # --- Whisper --------------------------------------------------------
    whisper_model: WhisperModelSize = Field(default="small")
    whisper_language: str | None = Field(default=None)

    # --- Sentiment ------------------------------------------------------
    sentiment_model: str = Field(default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    sentiment_batch_size: int = Field(default=16, ge=1, le=128)

    # --- Audio chunking -------------------------------------------------
    chunk_seconds: int = Field(default=600, ge=30, le=3600)
    silence_thresh_dbfs: int = Field(default=-40, le=0)

    # --- Logging --------------------------------------------------------
    log_level: LogLevel = Field(default="INFO")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_raw_dir(self) -> Path:
        """Directory holding downloaded raw audio files."""
        return self.raw_dir or (self.data_dir / "raw")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_transcripts_dir(self) -> Path:
        """Directory holding Whisper transcript JSONs."""
        return self.transcripts_dir or (self.data_dir / "transcripts")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_outputs_dir(self) -> Path:
        """Directory holding sentiment / analysis output JSONs."""
        return self.outputs_dir or (self.data_dir / "outputs")

    def ensure_dirs(self) -> None:
        """Create all artefact directories if they do not yet exist."""
        for path in (
            self.resolved_raw_dir,
            self.resolved_transcripts_dir,
            self.resolved_outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _configure_logging(level: LogLevel) -> None:
    """Idempotently configure root logging with a portfolio-grade format."""
    if getattr(_configure_logging, "_done", False):
        return

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": (
                        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
                    ),
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                }
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {
                "level": level,
                "handlers": ["stderr"],
            },
            "loggers": {
                # Tame noisy third-party loggers.
                "urllib3": {"level": "WARNING"},
                "huggingface_hub": {"level": "WARNING"},
                "matplotlib": {"level": "WARNING"},
            },
        }
    )
    _configure_logging._done = True  # type: ignore[attr-defined]


settings = Settings()
"""Singleton :class:`Settings` instance used across the codebase."""

_configure_logging(settings.log_level)
