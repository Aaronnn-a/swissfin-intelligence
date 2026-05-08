"""Aggregate per-segment sentiment into a smooth time series and detect
the prepared-remarks → Q&A boundary that's typical of earnings calls."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from swissfin.sentiment.analyzer import SegmentSentiment

logger = logging.getLogger(__name__)

_LABEL_TO_VALUE: dict[str, float] = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}


def _signed_score(label: str, score: float) -> float:
    """Map (label, confidence) → continuous sentiment score in ``[-1, 1]``."""
    base = _LABEL_TO_VALUE.get(label.lower(), 0.0)
    # Confidence weighting: a positive label at 0.95 → 0.95, negative at 0.6 → -0.6.
    return float(base * score) if base != 0.0 else 0.0


def build_tone_timeline(
    segment_sentiments: Iterable[SegmentSentiment | dict],
    window_seconds: int = 60,
) -> pd.DataFrame:
    """Convert per-segment sentiment into a per-second tone time series.

    Each second of the call is given a sentiment score by averaging the
    contributions of all segments that overlap it. A rolling average over
    ``window_seconds`` smooths the signal for plotting.

    Args:
        segment_sentiments: Output of :meth:`SentimentAnalyzer.analyze_segments`
            (or already-deserialized dicts).
        window_seconds: Width of the rolling-mean window in seconds.

    Returns:
        DataFrame with columns ``t_seconds``, ``sentiment_score``, ``rolling_avg``,
        ``coverage`` (fraction of the second covered by segments).
    """
    items = list(segment_sentiments)
    if not items:
        logger.warning("build_tone_timeline received no segments.")
        return pd.DataFrame(columns=["t_seconds", "sentiment_score", "rolling_avg", "coverage"])

    rows = [
        s.model_dump() if isinstance(s, SegmentSentiment) else dict(s)
        for s in items
    ]

    end_total = max(int(round(r["end"])) for r in rows)
    if end_total <= 0:
        end_total = len(rows)

    score_acc = np.zeros(end_total + 1, dtype=np.float64)
    cover_acc = np.zeros(end_total + 1, dtype=np.float64)

    for r in rows:
        start = max(int(round(r["start"])), 0)
        end = max(int(round(r["end"])), start + 1)
        end = min(end, end_total)
        if end <= start:
            continue
        signed = _signed_score(r["label"], r["score"])
        score_acc[start:end] += signed
        cover_acc[start:end] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        sentiment = np.where(cover_acc > 0, score_acc / cover_acc, 0.0)

    df = pd.DataFrame(
        {
            "t_seconds": np.arange(end_total + 1),
            "sentiment_score": sentiment,
            "coverage": cover_acc,
        }
    )
    window = max(1, int(window_seconds))
    df["rolling_avg"] = (
        df["sentiment_score"].rolling(window=window, min_periods=1, center=True).mean()
    )
    return df


def detect_section_break(
    timeline: pd.DataFrame,
    threshold: float = 0.4,
    *,
    min_offset_seconds: int = 120,
) -> int | None:
    """Heuristically find the prepared-remarks → Q&A transition.

    The prepared portion of an earnings call is scripted and emotionally flat;
    Q&A introduces variance as analysts probe difficult topics. We compute the
    first second at which the rolling standard deviation of sentiment exceeds
    ``threshold × global_std``.

    Args:
        timeline: Output of :func:`build_tone_timeline`.
        threshold: Multiplier on the call-wide std-dev that defines a break.
            ``0.4`` works well empirically on Swiss IR calls.
        min_offset_seconds: Minimum offset before a break may be reported. The
            first ~2 minutes are almost always still scripted.

    Returns:
        The estimated break time in seconds, or ``None`` if no clear shift is
        detected.
    """
    if timeline.empty or "rolling_avg" not in timeline.columns:
        return None

    series = timeline["rolling_avg"].fillna(0.0)
    if len(series) <= min_offset_seconds + 60:
        return None

    global_std = float(series.std())
    if global_std == 0.0:
        return None

    rolling_std = series.rolling(window=60, min_periods=10).std().fillna(0.0)
    bar = threshold * global_std
    candidates = rolling_std.iloc[min_offset_seconds:]
    above = candidates[candidates > bar]
    if above.empty:
        return None
    return int(timeline.loc[above.index[0], "t_seconds"])
