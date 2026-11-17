
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, HTTPException
from pathlib import Path
from uuid import uuid4
import json

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
    return {
        "answer": f"Received your question: {request.question}. RAG integration pending.",
        "sources": [],
        "received_attachments": {
            "pdf_ids": request.uploaded_pdf_ids,
            "image_ids": request.image_ids
        }
    }

# Directory for uploaded PDFs
UPLOAD_DIR = Path("data/uploads")
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

    # Save using a unique filename
    paper_id = uuid4().hex
    file_path = UPLOAD_DIR / f"{paper_id}.pdf"

    file_path.write_bytes(content)

    # Store original paper information
    metadata = {
        "paper_id": paper_id,
        "filename": filename,
        "status": "uploaded_not_indexed"
    }

    metadata_path = UPLOAD_DIR / f"{paper_id}.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    return {
        "paper_id": paper_id,
        "filename": filename,
        "status": "uploaded_not_indexed"
    }


@app.get("/papers")
def list_papers():
    papers = []

    for pdf_path in UPLOAD_DIR.glob("*.pdf"):
        papers.append({
            "paper_id": pdf_path.stem,
            "filename": pdf_path.name,
            "status": "uploaded_not_indexed"
        })

    return {
        "papers": papers,
        "total": len(papers)
    }


@app.get("/papers")
def list_papers():
    papers = []

    for pdf_path in UPLOAD_DIR.glob("*.pdf"):
        paper_id = pdf_path.stem
        metadata_path = UPLOAD_DIR / f"{paper_id}.json"

        if metadata_path.exists():
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        else:
            # Handle PDFs uploaded before metadata was added
            metadata = {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "status": "uploaded_not_indexed"
            }

        papers.append(metadata)

    return {
        "papers": papers,
        "total": len(papers)
    }

# Directory for uploaded images
IMAGE_DIR = Path("data/uploads/images")
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


