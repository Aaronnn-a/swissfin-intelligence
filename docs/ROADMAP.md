# SwissFin Intelligence — Roadmap

This roadmap lays out a four-phase plan toward a production-grade, locally-runnable AI stack for Swiss financial content. Each phase is independently demonstrable: it produces a runnable artefact, a measurable outcome, and a piece of the long-term architecture.

> Estimates assume part-time work alongside vocational training and (later) a Data Science degree at ZHAW. Dates are deliberately rough.

---

## Phase 1 — Earnings Call Intelligence 🚧 *In progress*

**Goal:** ship an end-to-end local pipeline that turns a Swiss earnings-call URL into a per-segment sentiment timeline, viewable in a demo UI.

### Scope
- yt-dlp-based audio downloader (YouTube, IR sites, podcast feeds)
- Silence-aware chunker (pydub) for long-form audio
- Local Whisper transcription with chunked, timestamp-aware aggregation
- Multilingual sentiment baseline via `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Tone-timeline aggregation + heuristic prepared-remarks → Q&A break detector
- Streamlit demo UI: timeline plot, segment table, section-break overlay
- Test suite (mocked HF, no network), pinned deps, ruff + black, Makefile

### Deliverables
- `src/swissfin/` package, `scripts/run_pipeline.py`, `app/streamlit_app.py`
- `data/outputs/<call>.sentiment.json` files for at least three real Swiss calls (UBS, Roche, Nestlé)
- A README screenshot (`docs/screenshots/streamlit_demo.png`) showing the working UI
- A short blog post on Substack walking through the architecture

### Success metric
- A reviewer can clone the repo, run `make setup && make pipeline URL=… NAME=…`, and see a tone timeline within ~10 minutes on a CPU-only laptop.
- ≥ 90% of transcript segments receive a non-neutral, plausible sentiment label on hand-curated test calls.

### Estimated duration
~3–4 weeks of focused part-time work.

---

## Phase 2 — Annual Report RAG ⏳ *Planned*

**Goal:** extend the pipeline to PDF annual reports of SIX-listed companies, and let users ask analytical questions that cross-reference earnings calls and the matching annual report.

### Scope
- PDF ingestion (PyMuPDF / pdfplumber) with section-aware splitting (MD&A, segment reporting, risk factors)
- Local embeddings (`bge-m3` or `multilingual-e5-large`) via sentence-transformers
- Local vector store (ChromaDB or Qdrant in Docker), persisted under `data/vectorstore/`
- Hybrid retrieval (BM25 + dense) with metadata filters (company, fiscal year, section)
- Local LLM Q&A via `llama-cpp-python` or `ollama`, default model `llama-3.1-8b-instruct` (Q4 quantized)
- New CLI: `scripts/ingest_report.py` and `scripts/ask.py`
- Streamlit tab: ask a question; answer is grounded with PDF page anchors and transcript timestamps

### Deliverables
- A second `swissfin/rag/` subpackage with retriever, indexer, and answerer modules
- An ingested corpus of ~10 SIX annual reports (2023–2024)
- A small evaluation set (~30 hand-written Q/A pairs) and a Jupyter notebook benchmarking retrieval Hit@5 and answer faithfulness

### Success metric
- Hit@5 ≥ 0.75 on the eval set with default retrieval settings.
- For a question like *"How did Roche describe Diagnostics-segment headwinds in 2024?"* the system returns an answer citing both the Q3 call segment **and** the matching MD&A paragraph.

### Estimated duration
~6–8 weeks.

---

## Phase 3 — German-Finance Sentiment Fine-Tune ⏳ *Planned*

**Goal:** replace the multilingual baseline with a model that actually understands the register of Swiss-German earnings calls — and publish the dataset and weights as portfolio artefacts.

### Scope
- Curate a labelled dataset (~5–10k segments) from publicly available transcripts of DACH-listed issuers, with three-way sentiment + an additional `forward_looking` flag
- Publish dataset as `aaronashraf/de-finance-sentiment` on HuggingFace Datasets
- Fine-tune XLM-RoBERTa-base (and a smaller distil variant for CPU inference) using `transformers` Trainer
- Evaluate against the multilingual baseline on a held-out test set: macro-F1, per-class precision/recall, calibration
- Publish weights as `aaronashraf/swissfin-sentiment-de` on HuggingFace Hub
- Wire the new model into `swissfin.config.Settings` so the existing pipeline picks it up without code changes

### Deliverables
- HF dataset card + model card (with limitations, intended use, ethical notes)
- A `notebooks/fine_tune_sentiment.ipynb` reproducing the run end-to-end
- An updated README screenshot showing improved results on a Swiss call

### Success metric
- Macro-F1 ≥ +5 absolute points over the multilingual baseline on the held-out test split.
- The HF dataset receives at least one external download/clone (proof it's discoverable).

### Estimated duration
~8–10 weeks (dominated by data labelling).

---

## Phase 4 — Production Deployment ⏳ *Planned*

**Goal:** wrap the pipeline as a service that an internal team at a Swiss bank could run behind their VPN, no internet egress required after image pull.

### Scope
- FastAPI service exposing: `POST /pipeline` (URL → job), `GET /jobs/{id}`, `GET /transcripts/{name}`, `GET /sentiment/{name}`
- Background worker (RQ or Celery + Redis) for long-running transcription jobs
- Multi-stage Dockerfile: CPU and CUDA variants, both pre-baking model weights to make the image fully air-gapped
- `docker compose up` brings up API + worker + Redis + (optional) vector store
- Auth: simple bearer-token middleware (real deployments would integrate with the bank's IdP)
- Structured JSON logging, Prometheus metrics, basic request tracing
- GitHub Actions CI: lint + test on every PR; image build + smoke test on tag

### Deliverables
- `docker/Dockerfile.api`, `docker/Dockerfile.worker`, `compose.yml`
- A short architecture doc (`docs/DEPLOYMENT.md`) with sequence diagrams
- A signed v1.0.0 release on GitHub

### Success metric
- An external reviewer can run `docker compose up` and POST a URL to `/pipeline` to receive a sentiment JSON within the documented latency budget (≤ 1.5× real-time for `whisper-small` on a modern CPU).
- The image runs with `--network=none` after model pre-baking — no internet calls during inference.

### Estimated duration
~6–8 weeks.

---

## Beyond Phase 4 (open ideas)

- Quarter-over-quarter tone-shift dashboard (Δ sentiment, Δ topic frequencies)
- Speaker-aware diarization (CEO vs CFO vs analyst) using `pyannote.audio`
- Question-quality scoring on the Q&A section (challenging vs softball analyst questions)
- Multilingual extension: French earnings calls (Lonza, Givaudan) and Italian (Ti.fin)
- Cross-asset signal study: does tone Δ predict the next 5d return on the underlying SIX-listed equity?

These remain explicitly out of scope for the v1 portfolio milestone but are tracked in GitHub issues as the project matures.
