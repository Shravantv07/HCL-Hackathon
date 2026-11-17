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

Full metadata in [paper_catalogue.csv](paper_catalogue.csv).

## Features

- Ingest 8–10 research PDFs with figure and table extraction
- Evidence-grounded Q&A with page-level citations
- Cross-paper comparison (methodology, results, architecture)
- Literature synthesis with AI-labeled research gap suggestions
- Multimodal visual Q&A using Gemini vision
- Prompt-injection defence and safe secret handling

## Team

<!-- Add team member names here -->

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Shravantv07/HCL-Hackathon.git
cd HCL-Hackathon

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Add papers
# Place PDF files in data/papers/
# Fill in paper_catalogue.csv with metadata

# 6. Ingest papers
python -m src.ingestion.indexer --catalogue paper_catalogue.csv

# 7. Run the app
streamlit run app/streamlit_app.py
```

## Architecture

```
PDFs
 └─► pdf_parser.py        (text extraction, OCR fallback)
 └─► figure_extractor.py  (image extraction + caption linking)
 └─► chunker.py           (512-token chunks with overlap + metadata)
 └─► indexer.py           (Gemini embeddings → ChromaDB)
          │
          ▼
     ChromaDB + figures_index.json
          │
          ▼
     retriever.py          (hybrid BM25 + dense, RRF merge)
     router.py             (query classifier)
          │
          ▼
     qa_chain.py           (factual Q&A + citations)
     comparison_chain.py   (cross-paper comparison)
     synthesis_chain.py    (literature synthesis + gap detection)
     visual_chain.py       (Gemini vision for figures/charts)
          │
          ▼
     Streamlit UI
```

## Evaluation

See [evaluation/eval_questions.csv](evaluation/eval_questions.csv) for the full 15-question evaluation set and results.

## Design Decisions and Limitations

<!-- Fill in after implementation -->

## Security Notes

- API keys loaded from `.env` via python-dotenv; never committed to git
- Retrieved paper content wrapped in `<source>` tags to prevent prompt injection
- File uploads validated for type (PDF only) and size (max 50 MB)
- Adversarial prompt-injection test documented in `tests/test_security.py`
