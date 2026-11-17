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
from src.chains import sources as chain_sources
from src.ingestion.uploads import (
    UPLOAD_DIR, UploadError, delete_upload, ingest_text_upload, ingest_upload, load_registry,
)
from backend.dispatch import route_question

app = FastAPI(title="ResearchIQ API")



class HistoryTurn(BaseModel):
    role: str
    content: str = ""


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    selected_papers: list[str] = Field(default_factory=list)
    uploaded_pdf_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    history: list[HistoryTurn] = Field(default_factory=list)   # recent turns, for follow-ups
    verify: bool = False                                        # claim-level evidence check


@app.get("/health")
def health():
    return {"status": "Backend is running!"}



@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Route the question to QA, comparison, synthesis or figure analysis
    (see backend/dispatch.py). Papers attached to this message or selected in the
    UI scope the search; with none, the whole library is searched.
    """
    images = [p for image_id in request.image_ids
              for p in IMAGE_DIR.glob(f"{Path(image_id).name}.*")]
    if request.image_ids and not images:
        raise HTTPException(status_code=404, detail="Uploaded image not found.")

    result = route_question(
        request.question,
        paper_ids=request.selected_papers + request.uploaded_pdf_ids,
        image_paths=images,
        history=[t.model_dump() for t in request.history],
        verify=request.verify,
    )
    result["received_attachments"] = {
        "pdf_ids": request.uploaded_pdf_ids,
        "image_ids": request.image_ids,
    }
    return result

# Directory for uploaded PDFs (shared with src/ingestion/uploads.py)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


TEXT_EXTENSIONS = (".txt", ".md")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    is_text = filename.lower().endswith(TEXT_EXTENSIONS)

    # Check file extension
    if not (filename.lower().endswith(".pdf") or is_text):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, .txt and .md files are allowed"
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
    if not is_text and not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file"
        )

    # Save, index (text + figures) and register the paper. Indexing is blocking work
    # (PDF parsing + embedding calls), so it runs in a worker thread. Text files are
    # converted to PDF first so they follow exactly the same pipeline.
    try:
        ingest = ingest_text_upload if is_text else ingest_upload
        record = await run_in_threadpool(ingest, content, filename)
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


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    """Remove a user upload from the index, the registry and disk (team papers cannot be deleted)."""
    try:
        removed = delete_upload(paper_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Upload not found (catalogue papers cannot be deleted).")
    chain_sources.refresh_retriever()
    return {"paper_id": paper_id, "deleted": True, "chunks_removed": removed}

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


