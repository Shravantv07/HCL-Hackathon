
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, HTTPException
from pathlib import Path
from uuid import uuid4


app = FastAPI(title="ResearchIQ API")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    selected_papers: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "Backend is running!"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Temporary response until RAG is connected
    return {
        "answer": f"Received your question: {request.question}. RAG integration pending.",
        "sources": []
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

    return {
        "paper_id": paper_id,
        "filename": filename,
        "status": "uploaded_not_indexed"
    }
