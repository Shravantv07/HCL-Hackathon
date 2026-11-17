# ResearchIQ – Multimodal AI Research Assistant

Hackathon project for the ResearchIQ challenge.

**Topic:** Multimodal Large Language Models (Topic 2)

Sub-areas covered: Vision-language models, document intelligence, visual question answering, chart and table understanding, scientific figure understanding.

## Paper Collection (9 papers)

| ID | Title | Year | Source |
|---|---|---|---|
| p01 | LLaVA-OneVision-1.5 | 2025 | arXiv |
| p02 | VisuLogic | 2025 | arXiv |
| p03 | SimpleDoc | 2025 | ACL Anthology |
| p04 | ChartMind | 2025 | ACL Anthology |
| p05 | MISS-QA | 2025 | ACL Anthology |
| p06 | LongTableBench | 2025 | ACL Anthology |
| p07 | SciFigBench | 2026 | ACL Anthology |
| p08 | M3DocVQA | 2025 | CVF Open Access |
| p09 | Table as a Modality for LLMs | 2025 | NeurIPS |

Full metadata in [paper_catalogue.csv](paper_catalogue.csv). Users can add their own papers from the chat (see *Uploads*).

## Features

One chat box; each question is routed automatically:

| You ask… | ResearchIQ does |
|---|---|
| a factual question | Evidence-grounded answer with labelled citations `[S1]` → paper, page, section and the original passage |
| "Compare X and Y" (named or selected papers) | Side-by-side table across objective, method, datasets, setup, metrics, results, limitations; "Not reported in the supplied content" where a paper is silent |
| "research gaps / across all papers / overview" | Per-paper summaries → cross-paper synthesis; research gaps labelled as AI suggestions |
| about a figure / chart / table | Sends the actual figure image to Gemini vision (named figures such as "Figure 1" are matched by caption) |
| with an image attached | Answers from the uploaded image itself |
| a follow-up ("what about its limitations?") | Resolves the reference from the recent conversation first |

Plus:

- **Evidence check** (sidebar toggle): every claim is checked against its cited passage — supported / partially supported / unsupported / contradicted / interpretation.
- **Uploads:** attach PDF, `.txt` or `.md` files in the chat. They are parsed (with OCR for scanned pages), their figures described, indexed and registered automatically; duplicates are detected; uploads can be removed from the sidebar.
- **Paper library** in the sidebar: limit answers to chosen papers.
- **Scanned PDFs:** Tesseract OCR, with Gemini vision as a fallback for pages it cannot read.
- **Searchable figures:** every extracted figure gets an AI-generated description at indexing time, so chart values and diagram contents can be found by search.
- Prompt-injection defence and safe secret handling.

## Team

| Member | Area |
|---|---|
| Murshid-Khalil | Document processing & ingestion (Member 1) |
| Riithvic | Retrieval: embeddings, hybrid search, reranking (Member 2) |
| Manavalan | LLM reasoning: Q&A, comparison, multimodal, evidence validation (Member 3) |
| Shravan | Streamlit UI & FastAPI backend (Member 4) |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Shravantv07/HCL-Hackathon.git
cd HCL-Hackathon

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies (includes sentence-transformers for the reranker)
pip install -r requirements.txt

# 4. Install Tesseract for OCR of scanned PDFs (optional; Gemini vision is the fallback)
winget install UB-Mannheim.TesseractOCR   # Windows
# sudo apt install tesseract-ocr          # Linux / brew install tesseract on macOS

# 5. Configure environment
copy .env.example .env
# Edit .env: set GEMINI_API_KEY, and use LLM_MODEL=gemini-3.8-flash

# 6. Add papers: place the PDFs listed in paper_catalogue.csv in data/papers/

# 7. Build the index (Gemini embeddings + figure descriptions; ~10 min the first time)
python -m src.ingestion.indexer --catalogue paper_catalogue.csv

# 8. Run the backend and the UI (two terminals, from the project root)
uvicorn backend.main:app --port 8000
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The first question after a restart takes about a minute (the retriever builds its BM25 index and loads the reranker); later questions take ~15–60 s depending on the mode.

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required |
| `LLM_MODEL` | `gemini-3.8-flash` (code default) | Answering, comparison, synthesis, vision |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Must be the same at indexing and query time |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 256 / 32 | Rebuild the index after changing |
| `TOP_K` | 6 | Passages per question |
| `RERANK_ENABLED` | true | Cross-encoder reranking for Q&A (comparison/synthesis skip it for speed) |
| `VISION_FIGURE_DESCRIPTIONS` | true | Describe figures with Gemini while indexing |
| `VISION_OCR_FALLBACK` | true | Gemini OCR for scanned pages Tesseract cannot read |
| `VISION_WORKERS` | 8 | Parallel Gemini calls for figure descriptions (lower if rate-limited) |
| `TESSERACT_CMD` | auto-detected | Path to `tesseract` if it is not found automatically |

### API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{question, selected_papers, uploaded_pdf_ids, image_ids, history, verify}` → `{answer, route, status, sources, warnings, verification}` |
| `POST /upload` | PDF / `.txt` / `.md` → indexed and registered (`paper_id`, title, page/chunk/figure counts) |
| `POST /upload-image` | PNG / JPG for image questions |
| `GET /papers` | Catalogue papers + uploads with status |
| `DELETE /papers/{paper_id}` | Remove an upload (catalogue papers cannot be deleted) |
| `GET /health` | Liveness |

## Architecture

```
PDFs / .txt / .md (catalogue or chat upload)
 └─► uploads.py           (validate, de-duplicate by hash, register in data/uploads/registry.json)
 └─► pdf_parser.py        (PyMuPDF text; Tesseract OCR; Gemini OCR fallback)
 └─► figure_extractor.py  (embedded images + caption linking)
 └─► vision.py            (Gemini figure descriptions, cached by image hash)
 └─► chunker.py           (256-token chunks with overlap + metadata)
 └─► indexer.py           (Gemini embeddings → ChromaDB, figures_index.json)
          │
          ▼
     retriever.py          (dense + BM25 → RRF → cross-encoder reranker)
     router.py             (query classifier)
          │
          ▼
     backend/dispatch.py   (routes /ask to a chain, renders the answer)
     qa_chain.py           (Q&A + labelled citations)
     comparison_chain.py   (cross-paper comparison table)
     synthesis_chain.py    (per-paper summaries → synthesis + gaps)
     visual_chain.py       (Gemini vision on the actual figure)
     verification_chain.py (claim-level evidence check)
          │
          ▼
     FastAPI backend  ◄──►  Streamlit UI
```

## Evaluation

[evaluation/EVALUATION.md](evaluation/EVALUATION.md) is the latest report: method, results by question type, per-question outcomes, failures and limitations. The question set is [evaluation/eval_questions.csv](evaluation/eval_questions.csv) (26 questions: factual, methodology, results, limitations, cross-paper, synthesis, figure, unanswerable, prompt injection); every expected fact is checked to occur in its paper's indexed text before scoring. Re-run with:

```bash
python -m evaluation.run_eval
```

Tests: `pytest -m "not integration"` (offline, no API calls). Live end-to-end tests: `RUN_LIVE_TESTS=1 pytest -m integration`.

## Design Decisions and Limitations

**Decisions**

- **Citations cannot be invented.** Passages are given short labels (`[S1]`…) and the model cites labels only; the code maps labels back to the real paper, page and chunk, and drops any label that was not supplied. In comparisons, a cell may only cite its own paper's passages.
- **Answer only from evidence.** When the passages do not cover a question the system says so instead of guessing; a question asked with an unreadable attachment is not answered from unrelated papers.
- **Interpretation is labelled.** Model inferences are prefixed "Interpretation:", research gaps are marked as AI suggestions, and figure descriptions are stored as `[AI-generated figure description]` so they are never mistaken for the authors' words.
- **Hybrid retrieval + reranking** for precision on specific facts; comparison and synthesis retrieve per paper and per dimension and skip the reranker to keep latency reasonable.
- **One integration layer** (`src/chains/sources.py`) isolates the reasoning code from ingestion/retrieval formats.
- **Vision at ingestion and at question time:** figures are described once when indexed (searchable, cached), and the actual image is sent again when a user asks about it.

**Limitations**

- Vector-drawn figures (lines and shapes, not embedded images) are not extracted, so they get no image description; questions about them rely on captions and text.
- Comparison and synthesis take ~1 minute; uploads with many figures take longer because each figure is described.
- The evidence check depends on an LLM judge; deterministic checks (quotes must occur in the cited passage, numbers must match, wrong-paper attribution) can only downgrade a verdict.
- The evaluation set is small (26 questions) and keyword-scored; see the report for details.
- No authentication — intended for local use.

## Security Notes

- API keys loaded from `.env` via python-dotenv; never committed to git
- Retrieved paper content wrapped in `<source>` tags to prevent prompt injection; the evaluation includes a direct and an embedded injection test
- File uploads validated for type (PDF, .txt, .md, PNG/JPG), signature and size
- Uploaded files, the index and generated figures stay local (git-ignored)
