"""
ResearchIQ API. Run from the project root:  uvicorn backend.main:app --reload
"""
import csv

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from uuid import uuid4

from src.config import PROJECT_ROOT
from src.chains import qa_chain, sources as chain_sources
from src.ingestion.uploads import UPLOAD_DIR, UploadError, ingest_upload, load_registry

app = FastAPI(title="ResearchIQ API")



class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    selected_papers: list[str] = Field(default_factory=list)
    uploaded_pdf_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "Backend is running!"}



@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Papers attached to this message (or selected in the UI) scope the search;
    # with none, the whole library is searched.
    paper_ids = list(dict.fromkeys(request.selected_papers + request.uploaded_pdf_ids)) or None
    result = qa_chain.answer(request.question, paper_ids=paper_ids)

    answer = result["answer"]
    if result["status"] == "error":
        answer = f"⚠️ {result['error']}"

    return {
        "answer": answer,
        "status": result["status"],
        "sources": result["citations"],   # each has paper, page, paper_id, chunk_id, section, text
        "warnings": result["warnings"],
        "received_attachments": {
            "pdf_ids": request.uploaded_pdf_ids,
            "image_ids": request.image_ids
        }
    }

# Directory for uploaded PDFs (shared with src/ingestion/uploads.py)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    # Check file extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Read file with size limit
    try:
        content = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 20 MB"
        )

    # Basic PDF signature validation
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file"
        )

    # Save, index (text + figures) and register the paper. Indexing is blocking work
    # (PDF parsing + embedding calls), so it runs in a worker thread.
    try:
        record = await run_in_threadpool(ingest_upload, content, filename)
    except UploadError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # The shared retriever's BM25 index was built before this paper existed.
    if not record.get("duplicate"):
        chain_sources.refresh_retriever()

    return {
        "paper_id": record["paper_id"],
        "filename": filename,
        "title": record.get("title"),
        "status": record["status"],            # "indexed"
        "duplicate": record.get("duplicate", False),
        "pages": record.get("pages"),
        "text_chunks": record.get("text_chunks"),
        "figures": record.get("figures"),
    }


@app.get("/papers")
def list_papers():
    """Every registered paper: the team catalogue plus user uploads (with their status)."""
    papers = []

    catalogue = PROJECT_ROOT / "paper_catalogue.csv"
    if catalogue.exists():
        with open(catalogue, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id", "").strip():
                    papers.append({
                        "paper_id": row["id"].strip(),
                        "title": row.get("title", "").strip(),
                        "filename": row.get("local_filename", "").strip(),
                        "status": "indexed",
                        "source": "catalogue",
                    })

    for record in load_registry().values():
        papers.append({k: record.get(k) for k in (
            "paper_id", "title", "filename", "status", "error",
            "pages", "text_chunks", "figures", "uploaded_at")} | {"source": "upload"})

    return {
        "papers": papers,
        "total": len(papers)
    }

# Directory for uploaded images
IMAGE_DIR = UPLOAD_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    # Check supported formats
    extension = Path(filename).suffix.lower()

    if extension not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPG images are allowed"
        )

    try:
        content = await file.read(MAX_IMAGE_SIZE + 1)
    finally:
        await file.close()

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB"
        )

    # Check actual file signature
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = content.startswith(b"\xff\xd8\xff")

    if not (
        (extension == ".png" and is_png)
        or (extension in [".jpg", ".jpeg"] and is_jpeg)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format"
        )

    # Save image
    image_id = uuid4().hex
    image_path = IMAGE_DIR / f"{image_id}{extension}"
    image_path.write_bytes(content)
    
    return {
        "image_id": image_id,
        "filename": filename,
        "file_type": "image",
        "status": "uploaded_not_processed"
    }


