# ResearchIQ – Multimodal AI Research Assistant

Hackathon project for the ResearchIQ challenge.

**Topic:** Advanced Retrieval-Augmented Generation (RAG)

## Features

- Ingest 8–10 research PDFs with figure and table extraction
- Evidence-grounded Q&A with page-level citations
- Cross-paper comparison (methodology, results, architecture)
- Literature synthesis with AI-labeled research gap suggestions
- Multimodal visual Q&A using GPT-4o vision
- Prompt-injection defence and safe secret handling

## Team

<!-- Add team member names here -->

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd researchiq

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Add papers
# Place PDF files in data/papers/
# Fill in paper_catalogue.csv with metadata

# 6. Ingest papers
python -m src.ingestion.indexer

# 7. Run the app
streamlit run app/streamlit_app.py
```

## Architecture

<!-- Add architecture diagram here -->

## Paper Catalogue

See [paper_catalogue.csv](paper_catalogue.csv).

## Evaluation

See [evaluation/eval_questions.csv](evaluation/eval_questions.csv) for the full question set and results.

## Design Decisions and Limitations

<!-- Fill in after implementation -->

## Security Notes

- API keys are loaded from `.env` via python-dotenv; never committed to git
- Retrieved paper content is wrapped in `<source>` tags to prevent prompt injection
- File uploads are validated for type and size before processing
- Adversarial prompt-injection test documented in `tests/test_security.py`
