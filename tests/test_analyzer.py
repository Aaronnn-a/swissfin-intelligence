"""Unit tests for :mod:`swissfin.sentiment.analyzer`.

The HuggingFace pipeline is mocked so the suite stays fast and offline-safe.
"""

from __future__ import annotations

from typing import Any

import pytest

from swissfin.sentiment.analyzer import SegmentSentiment, SentimentAnalyzer


class _FakePipeline:
    """Stand-in for ``transformers.pipeline('text-classification', ...)``."""

    def __init__(self, mapping: dict[str, tuple[str, float]]) -> None:
        self._mapping = mapping
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], batch_size: int = 16) -> list[list[dict[str, Any]]]:
        self.calls.append(list(texts))
        out: list[list[dict[str, Any]]] = []
        for t in texts:
            label, score = self._mapping.get(t, ("LABEL_1", 0.5))
            out.append([{"label": label, "score": score}])
        return out


@pytest.fixture
def fake_segments() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 5.0, "text": "Revenue grew strongly this quarter."},
        {"start": 5.0, "end": 10.0, "text": "Margins compressed across the investment bank."},
        {"start": 10.0, "end": 15.0, "text": "We reiterate our full-year guidance."},
        {"start": 15.0, "end": 16.0, "text": "   "},  # empty — should be skipped
    ]


def test_analyze_segments_normalizes_labels(monkeypatch, fake_segments) -> None:
    fake = _FakePipeline(
        {
            "Revenue grew strongly this quarter.": ("Positive", 0.92),
            "Margins compressed across the investment bank.": ("LABEL_0", 0.81),
            "We reiterate our full-year guidance.": ("neutral", 0.66),
        }
    )

    analyzer = SentimentAnalyzer(model_name="dummy/model")
    monkeypatch.setattr(analyzer, "_ensure_pipeline", lambda: fake)

    results = analyzer.analyze_segments(fake_segments)

    assert len(results) == 3
    assert all(isinstance(r, SegmentSentiment) for r in results)
    assert [r.label for r in results] == ["positive", "negative", "neutral"]
    assert results[0].score == pytest.approx(0.92)
    # Empty segment was skipped.
    assert all(r.text.strip() for r in results)
    # Pipeline was called exactly once with the three non-empty texts.
    assert len(fake.calls) == 1
    assert len(fake.calls[0]) == 3


def test_analyze_segments_handles_empty_input() -> None:
    analyzer = SentimentAnalyzer(model_name="dummy/model")
    assert analyzer.analyze_segments([]) == []
    assert analyzer.analyze_segments([{"start": 0, "end": 1, "text": "  "}]) == []


def test_segment_sentiment_validation() -> None:
    s = SegmentSentiment(start=0.0, end=2.5, text="hello", label="positive", score=0.8)
    assert s.score == 0.8
    with pytest.raises(ValueError):
        SegmentSentiment(start=0.0, end=1.0, text="x", label="positive", score=1.5)
