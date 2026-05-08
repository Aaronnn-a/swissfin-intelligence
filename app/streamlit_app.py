"""SwissFin Intelligence — Streamlit demo UI.

Lets a user pick a previously-scored transcript, visualize tone over time, and
inspect individual segments. Designed to be the user-facing surface of the
Phase-1 pipeline; runs entirely on the local machine, no external API calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from swissfin.analysis.tone_timeline import build_tone_timeline, detect_section_break  # noqa: E402
from swissfin.config import settings  # noqa: E402
from swissfin.sentiment.analyzer import SegmentSentiment, SentimentAnalyzer  # noqa: E402
from swissfin.utils.io import load_json, load_transcript  # noqa: E402

logger = logging.getLogger("swissfin.app")

st.set_page_config(
    page_title="SwissFin Intelligence",
    page_icon="🇨🇭",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _list_transcripts(transcripts_dir: Path) -> list[Path]:
    return sorted(p for p in transcripts_dir.glob("*.json") if p.is_file())


@st.cache_data(show_spinner=False)
def _list_outputs(outputs_dir: Path) -> dict[str, Path]:
    return {
        p.stem.removesuffix(".sentiment"): p
        for p in outputs_dir.glob("*.sentiment.json")
    }


def _render_sidebar() -> dict:
    st.sidebar.title("SwissFin Intelligence")
    st.sidebar.caption("Phase 1 — Earnings Call Intelligence")

    transcripts = _list_transcripts(settings.resolved_transcripts_dir)
    if not transcripts:
        st.sidebar.warning(
            "No transcripts found. Run "
            "`python scripts/run_pipeline.py` first."
        )
        return {}

    transcript_path = st.sidebar.selectbox(
        "Transcript",
        transcripts,
        format_func=lambda p: p.stem,
    )
    sentiment_model = st.sidebar.text_input(
        "Sentiment model",
        value=settings.sentiment_model,
        help="HuggingFace model id used if cached scores aren't available.",
    )
    window_seconds = st.sidebar.slider(
        "Rolling window (seconds)", min_value=10, max_value=300, value=60, step=10
    )
    threshold = st.sidebar.slider(
        "Section-break sensitivity", min_value=0.1, max_value=1.0, value=0.4, step=0.05,
        help="Higher = stricter; lower = more eagerly flag a Q&A transition.",
    )
    return {
        "transcript_path": transcript_path,
        "sentiment_model": sentiment_model,
        "window_seconds": window_seconds,
        "threshold": threshold,
    }


def _load_or_score(transcript_path: Path, sentiment_model: str) -> tuple[list[SegmentSentiment], dict]:
    """Use a cached sentiment file when available; otherwise score on-the-fly."""
    name = transcript_path.stem
    outputs = _list_outputs(settings.resolved_outputs_dir)
    transcript = load_transcript(transcript_path)
    meta = {
        "name": name,
        "language": transcript.get("language"),
        "n_segments": len(transcript["segments"]),
    }
    if name in outputs:
        cached = load_json(outputs[name])
        meta["sentiment_model"] = cached.get("sentiment_model", sentiment_model)
        meta["source"] = "cached"
        scored = [SegmentSentiment(**s) for s in cached["segments"]]
        return scored, meta

    with st.spinner(f"Scoring sentiment with {sentiment_model}…"):
        analyzer = SentimentAnalyzer(model_name=sentiment_model)
        scored = analyzer.analyze_segments(transcript["segments"])
    meta["sentiment_model"] = sentiment_model
    meta["source"] = "live"
    return scored, meta


def _plot_timeline(timeline: pd.DataFrame, break_t: int | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["t_seconds"],
            y=timeline["sentiment_score"],
            mode="lines",
            name="raw",
            line=dict(width=1, color="rgba(120,120,120,0.4)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeline["t_seconds"],
            y=timeline["rolling_avg"],
            mode="lines",
            name="rolling avg",
            line=dict(width=2.5, color="#cc0000"),
        )
    )
    fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
    if break_t is not None:
        fig.add_vline(
            x=break_t,
            line=dict(color="#0033a0", dash="dash"),
            annotation_text=f"Q&A starts ≈ {break_t // 60}m{break_t % 60:02d}s",
            annotation_position="top",
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time (s)",
        yaxis_title="Sentiment (-1 … +1)",
        yaxis=dict(range=[-1.05, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _segments_dataframe(scored: list[SegmentSentiment]) -> pd.DataFrame:
    df = pd.DataFrame([s.model_dump() for s in scored])
    if df.empty:
        return df
    df["start_hms"] = df["start"].apply(lambda s: f"{int(s) // 60:02d}:{int(s) % 60:02d}")
    df["score"] = df["score"].round(3)
    return df[["start_hms", "start", "end", "label", "score", "text"]]


def _color_label(label: str) -> str:
    return {
        "positive": "background-color: #d6f5d6;",
        "negative": "background-color: #f7d6d6;",
        "neutral": "background-color: #f0f0f0;",
    }.get(label, "")


def main() -> None:
    settings.ensure_dirs()
    cfg = _render_sidebar()
    if not cfg:
        st.title("SwissFin Intelligence")
        st.info(
            "Run the pipeline first:\n\n"
            "```bash\n"
            "python scripts/run_pipeline.py --url <URL> --name <stem>\n"
            "```"
        )
        return

    scored, meta = _load_or_score(cfg["transcript_path"], cfg["sentiment_model"])
    timeline = build_tone_timeline(scored, window_seconds=cfg["window_seconds"])
    break_t = detect_section_break(timeline, threshold=cfg["threshold"])

    st.title(f"📞 {meta['name']}")
    col1, col2, col3, col4 = st.columns(4)
    duration_s = int(timeline["t_seconds"].max()) if not timeline.empty else 0
    col1.metric("Duration", f"{duration_s // 60}m {duration_s % 60:02d}s")
    col2.metric("Segments", meta["n_segments"])
    col3.metric("Language", meta.get("language") or "auto")
    col4.metric("Sentiment model", meta["sentiment_model"].split("/")[-1])

    st.subheader("Tone timeline")
    if timeline.empty:
        st.warning("Not enough data to plot a timeline.")
    else:
        st.plotly_chart(_plot_timeline(timeline, break_t), use_container_width=True)

    st.subheader("Segments")
    seg_df = _segments_dataframe(scored)
    if seg_df.empty:
        st.info("No scored segments available.")
    else:
        styled = seg_df.style.map(_color_label, subset=["label"])
        st.dataframe(styled, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
