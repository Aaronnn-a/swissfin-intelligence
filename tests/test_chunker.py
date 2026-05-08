"""Unit tests for the silence-based chunk-boundary picker.

We deliberately exercise the pure helper :func:`_pick_boundaries` so that the
test suite stays fast (no real audio decoding, no FFmpeg dependency on CI).
"""

from __future__ import annotations

import pytest

from swissfin.audio.chunker import _pick_boundaries


class TestPickBoundaries:
    def test_returns_empty_when_audio_shorter_than_target(self) -> None:
        cuts = _pick_boundaries(silent_ranges=[[1_000, 2_000]], total_ms=30_000, chunk_ms=60_000)
        assert cuts == []

    def test_falls_back_to_target_when_no_silences(self) -> None:
        cuts = _pick_boundaries(silent_ranges=[], total_ms=180_000, chunk_ms=60_000)
        assert cuts == [60_000, 120_000]

    def test_snaps_to_nearby_silence(self) -> None:
        # Silence centred at 58s — within the 25% tolerance of the 60s target.
        cuts = _pick_boundaries(
            silent_ranges=[[57_000, 59_000]], total_ms=180_000, chunk_ms=60_000
        )
        assert 58_000 in cuts

    def test_ignores_silence_far_from_target(self) -> None:
        # Silence at 10s is too far from any target (60s, 120s).
        cuts = _pick_boundaries(
            silent_ranges=[[9_500, 10_500]], total_ms=180_000, chunk_ms=60_000
        )
        assert 10_000 not in cuts
        assert 60_000 in cuts

    def test_drops_duplicate_or_out_of_range_cuts(self) -> None:
        cuts = _pick_boundaries(
            silent_ranges=[[59_000, 61_000], [59_500, 60_500]],
            total_ms=130_000,
            chunk_ms=60_000,
        )
        assert sorted(cuts) == cuts
        assert all(0 < c < 130_000 for c in cuts)
        assert len(cuts) == len(set(cuts))


@pytest.mark.parametrize(
    ("total_ms", "chunk_ms", "expected_n"),
    [
        (60_000, 60_000, 0),
        (120_001, 60_000, 1),
        (300_000, 60_000, 4),
    ],
)
def test_target_count_matches_total_length(total_ms: int, chunk_ms: int, expected_n: int) -> None:
    cuts = _pick_boundaries(silent_ranges=[], total_ms=total_ms, chunk_ms=chunk_ms)
    assert len(cuts) == expected_n
