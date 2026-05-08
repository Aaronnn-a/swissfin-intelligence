"""Local Whisper transcription with chunked, timestamp-aware aggregation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from swissfin.audio.chunker import split_audio
from swissfin.config import settings

logger = logging.getLogger(__name__)


class TranscriptSegment(BaseModel):
    """A single time-aligned transcript segment as emitted by Whisper."""

    start: float = Field(..., ge=0.0, description="Segment start (seconds, global timeline).")
    end: float = Field(..., ge=0.0, description="Segment end (seconds, global timeline).")
    text: str = Field(..., description="Transcribed text for this segment.")


class TranscriptionResult(BaseModel):
    """Full transcription output: concatenated text + per-segment timestamps."""

    text: str = Field(..., description="Concatenated transcript across all chunks.")
    language: str | None = Field(default=None, description="Detected language code (e.g. 'de').")
    segments: list[TranscriptSegment] = Field(default_factory=list)
    audio_path: str | None = Field(default=None, description="Source audio file path.")
    model_size: str | None = Field(default=None, description="Whisper model size used.")


class WhisperTranscriber:
    """Thin wrapper around :func:`whisper.load_model` with chunking support.

    Audio shorter than :attr:`Settings.chunk_seconds` is transcribed in one
    pass; longer files are split via :func:`swissfin.audio.chunker.split_audio`
    and the per-chunk segment timestamps are shifted into a single global
    timeline before being returned.
    """

    def __init__(
        self,
        model_size: str | None = None,
        language: str | None = None,
        *,
        device: str | None = None,
    ) -> None:
        """Initialise the transcriber but defer model loading until first use.

        Args:
            model_size: One of Whisper's published model sizes. Defaults to the
                value configured in :data:`swissfin.config.settings`.
            language: ISO-639 language code (``de``, ``en``, ``fr``, ``it``).
                ``None`` triggers Whisper's auto-detection.
            device: ``"cuda"`` / ``"cpu"`` / ``"mps"``. ``None`` auto-detects.
        """
        self.model_size = model_size or settings.whisper_model
        self.language = language if language is not None else settings.whisper_language
        self._device = device
        self._model: Any | None = None

    @property
    def device(self) -> str:
        """Lazy-resolve the inference device (CUDA when available)."""
        if self._device is not None:
            return self._device
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dep
            self._device = "cpu"
        return self._device

    def _ensure_model(self) -> Any:
        """Load the Whisper model on first use and cache it on the instance."""
        if self._model is None:
            import whisper  # local import — heavy dependency

            logger.info("Loading Whisper model: size=%s device=%s", self.model_size, self.device)
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe ``audio_path``, chunking long files transparently.

        Args:
            audio_path: Path to an audio file decodable by FFmpeg.

        Returns:
            A :class:`TranscriptionResult` with global-timeline segments.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        try:
            from pydub import AudioSegment

            duration_ms = len(AudioSegment.from_file(audio_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not probe duration for %s (%s); assuming long audio.", audio_path, exc)
            duration_ms = settings.chunk_seconds * 1000 + 1

        chunks: list[Path]
        offsets_ms: list[int]
        if duration_ms <= settings.chunk_seconds * 1000:
            chunks = [audio_path]
            offsets_ms = [0]
        else:
            logger.info("Audio length %.1fs exceeds chunk size; splitting first.", duration_ms / 1000)
            chunks = split_audio(
                audio_path,
                chunk_seconds=settings.chunk_seconds,
                silence_thresh_dbfs=settings.silence_thresh_dbfs,
            )
            offsets_ms = self._estimate_offsets(chunks)

        all_segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        detected_language: str | None = None

        model = self._ensure_model()
        for idx, (chunk_path, offset_ms) in enumerate(zip(chunks, offsets_ms, strict=True)):
            logger.info("Transcribing chunk %d/%d (%s)", idx + 1, len(chunks), chunk_path.name)
            kwargs: dict[str, Any] = {"fp16": self.device == "cuda"}
            if self.language:
                kwargs["language"] = self.language
            result = model.transcribe(str(chunk_path), **kwargs)

            detected_language = detected_language or result.get("language")
            full_text_parts.append((result.get("text") or "").strip())
            offset_s = offset_ms / 1000.0
            for seg in result.get("segments", []) or []:
                all_segments.append(
                    TranscriptSegment(
                        start=float(seg["start"]) + offset_s,
                        end=float(seg["end"]) + offset_s,
                        text=str(seg.get("text", "")).strip(),
                    )
                )

        return TranscriptionResult(
            text=" ".join(p for p in full_text_parts if p).strip(),
            language=detected_language,
            segments=all_segments,
            audio_path=str(audio_path),
            model_size=self.model_size,
        )

    @staticmethod
    def _estimate_offsets(chunks: list[Path]) -> list[int]:
        """Compute cumulative offsets (ms) for each chunk relative to chunk 0."""
        from pydub import AudioSegment

        offsets: list[int] = []
        running = 0
        for chunk in chunks:
            offsets.append(running)
            running += len(AudioSegment.from_file(chunk))
        return offsets
