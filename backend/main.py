
from fastapi import FastAPI
from pydantic import BaseModel, Field

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
