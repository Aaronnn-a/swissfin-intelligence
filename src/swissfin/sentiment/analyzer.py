"""Per-segment sentiment scoring via HuggingFace ``transformers``.

The default model is multilingual (``cardiffnlp/twitter-xlm-roberta-base-sentiment``)
which gives a usable baseline for German-, French-, Italian- and English-language
calls. Phase 3 will swap this out for a German-finance-tuned model trained on
SIX-listed earnings transcripts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from swissfin.config import settings

logger = logging.getLogger(__name__)

# Map common HF sentiment label vocabularies to a consistent scheme.
_LABEL_NORMALIZATION: dict[str, str] = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


class SegmentSentiment(BaseModel):
    """Sentiment annotation for one transcript segment."""

    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str
    label: str = Field(..., description="One of 'positive' / 'neutral' / 'negative'.")
    score: float = Field(..., ge=0.0, le=1.0, description="Model confidence in ``label``.")


class SentimentAnalyzer:
    """Batched sentiment classifier wrapping a HuggingFace text-classification pipeline."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int | None = None,
        device: int | str | None = None,
    ) -> None:
        """Initialise the analyzer (model loading is deferred until first call).

        Args:
            model_name: HuggingFace model id. Defaults to
                :data:`Settings.sentiment_model`.
            batch_size: Inference batch size. Defaults to
                :data:`Settings.sentiment_batch_size`.
            device: ``transformers.pipeline`` device argument. ``None``
                auto-detects (CUDA → ``0``, otherwise CPU).
        """
        self.model_name = model_name or settings.sentiment_model
        self.batch_size = batch_size or settings.sentiment_batch_size
        self._device = device
        self._pipeline: Any | None = None

    def _resolve_device(self) -> int | str:
        if self._device is not None:
            return self._device
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except ImportError:  # pragma: no cover
            return -1

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            logger.info("Loading sentiment model: %s", self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self._resolve_device(),
                truncation=True,
                top_k=1,
            )
        return self._pipeline

    def analyze_segments(self, segments: Iterable[dict[str, Any]]) -> list[SegmentSentiment]:
        """Score each segment in ``segments``.

        Args:
            segments: Iterable of dicts with at least ``start``, ``end``,
                ``text`` keys (matches the schema produced by
                :class:`swissfin.transcription.WhisperTranscriber`).

        Returns:
            One :class:`SegmentSentiment` per non-empty segment. Empty-text
            segments are skipped silently.
        """
        seg_list = [s for s in segments if (s.get("text") or "").strip()]
        if not seg_list:
            logger.warning("No non-empty segments supplied to SentimentAnalyzer.")
            return []

        texts = [s["text"].strip() for s in seg_list]
        pipe = self._ensure_pipeline()
        logger.info("Scoring %d segments (batch_size=%d)", len(texts), self.batch_size)
        raw = pipe(texts, batch_size=self.batch_size)

        results: list[SegmentSentiment] = []
        for seg, prediction in zip(seg_list, raw, strict=True):
            top = prediction[0] if isinstance(prediction, list) else prediction
            label = _LABEL_NORMALIZATION.get(str(top["label"]).lower(), str(top["label"]).lower())
            results.append(
                SegmentSentiment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"].strip(),
                    label=label,
                    score=float(top["score"]),
                )
            )
        return results
