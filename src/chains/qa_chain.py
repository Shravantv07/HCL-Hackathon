"""
Core Q&A chain: retrieve → prompt → LLM → answer + citations.

    answer(question)                         # retrieves via Member 2, then answers
    answer_from_passages(question, records)  # answers from passages you already have

Both return the same dict:
    {
        status: "ok" | "insufficient_evidence" | "error",
        answer: str,
        citations: [ {source_label, chunk_id, paper_id, paper, page, section,
                      fig_ref, text, display} ],
        chunks_used: [chunk_id],     # passages sent to the model
        chunks_omitted: [chunk_id],  # passages left out to fit the context budget
        warnings: [str],
        error: str | None,
        verification: dict | None,   # claim-level check (verification_chain) when verify=True
    }
Citations are built from the supplied passages only, never from model output.
"""

import json
from typing import Any, Dict, List, Optional

from src.chains import evidence
from src.chains.llm import LLMError, get_client
from src.chains.sources import SourceError, get_passages

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The selected papers do not contain enough evidence to answer this question."
)

QA_INSTRUCTIONS = """Answer the question using only the research passages below.

Citation format for this task (this replaces the [Paper Title, p.X] format): cite the
label of each supporting passage in square brackets right after the claim, e.g. [S2]
or [S1, S3]. Use only labels that appear below. Do not write paper titles, page
numbers, URLs, or chunk ids as citations; the application adds them from the labels.

Rules:
- Do not invent facts, numbers, datasets, methods, or conclusions.
- Distinguish what the authors state from your own explanation: prefix any sentence
  that is your interpretation or inference with "Interpretation:".
- If the passages only partly answer the question, answer that part and say what is missing.
- If the passages do not answer the question at all, set "insufficient_evidence" to true
  and set "answer" to exactly: "{insufficient}"

Respond with JSON only: {{"answer": "<text with [S#] citations>", "insufficient_evidence": <true|false>}}

Question: {question}

Passages:
{context}"""


def _result(status: str, answer: str = "", **extra: Any) -> Dict[str, Any]:
    out = {
        "status": status,
        "answer": answer,
        "citations": [],
        "chunks_used": [],
        "chunks_omitted": [],
        "warnings": [],
        "error": None,
        "verification": None,
    }
    out.update(extra)
    return out


def _parse_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):  # tolerate fenced JSON
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise ValueError("expected an object with a string 'answer'")
    return data


def answer_from_passages(
    question: str,
    passages: List[Dict[str, Any]],
    llm: Optional[Any] = None,
    max_context_tokens: int = evidence.DEFAULT_MAX_CONTEXT_TOKENS,
    verify: bool = False,
) -> Dict[str, Any]:
    """
    Answer `question` from retrieved passages (raw retriever records or normalized dicts).
    With verify=True, a successful answer also gets claim-level `verification` (one extra LLM call).
    """
    if not question or not question.strip():
        return _result("error", error="Question is empty.")

    usable, warnings = evidence.prepare_passages(passages)
    if not usable:
        return _result("insufficient_evidence", INSUFFICIENT_EVIDENCE_MESSAGE,
                       warnings=warnings + ["No usable passages were supplied; the model was not called."])

    kept, omitted = evidence.fit_to_budget(usable, max_context_tokens)
    if omitted:
        warnings.append(f"{len(omitted)} lower-ranked passage(s) were left out to fit the context budget.")
    labelled = evidence.label_passages(kept)
    bookkeeping = {
        "chunks_used": [p["chunk_id"] for p in kept],
        "chunks_omitted": [p["chunk_id"] for p in omitted],
    }

    prompt = QA_INSTRUCTIONS.format(
        insufficient=INSUFFICIENT_EVIDENCE_MESSAGE,
        question=question.strip(),
        context=evidence.build_context(labelled),
    )

    try:
        raw = (llm or get_client()).generate(prompt, json_output=True)
    except LLMError as e:
        return _result("error", error=str(e), warnings=warnings, **bookkeeping)

    try:
        data = _parse_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return _result("error", error=f"The model returned malformed output ({e}).",
                       warnings=warnings, **bookkeeping)

    if data.get("insufficient_evidence") is True:
        return _result("insufficient_evidence", INSUFFICIENT_EVIDENCE_MESSAGE,
                       warnings=warnings, **bookkeeping)

    citations, unknown = evidence.extract_citations(data["answer"], labelled)
    answer_text = evidence.strip_labels(data["answer"], unknown)
    if unknown:
        warnings.append(f"Removed citation label(s) not matching any supplied passage: {', '.join(unknown)}.")
    if not citations:
        warnings.append("The answer does not cite any supplied passage; treat it as unverified.")

    result = _result("ok", answer_text, citations=citations, warnings=warnings, **bookkeeping)
    if verify:
        from src.chains.verification_chain import verify_answer
        result["verification"] = verify_answer(result, llm=llm,
                                               papers={p["paper_id"]: p["title"] for p in kept})
    return result


def answer(question: str, top_k: Optional[int] = None, llm: Optional[Any] = None,
           verify: bool = False, paper_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Retrieve passages for `question` via Member 2's retriever, then answer from them.
    `paper_ids` (e.g. the UI's selected papers) limits retrieval to those papers.
    """
    if not question or not question.strip():
        return _result("error", error="Question is empty.")
    try:
        passages = get_passages(question, top_k=top_k, paper_ids=paper_ids)
    except SourceError as e:
        return _result("error", error=str(e))
    return answer_from_passages(question, passages, llm=llm, verify=verify)
