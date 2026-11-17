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
import re
from typing import Any, Dict, List, Optional

from src.chains import evidence
from src.chains.llm import LLMError, get_client
from src.chains.sources import SourceError, _catalogue_ids, get_passages

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
- If the question asks about "a" / "an" / "a key" figure, chart or table without naming one,
  choose the most relevant figure or table among the passages, name it (e.g. "Figure 7"),
  and explain it from its caption and surrounding text: what is measured, the axes or
  columns, the legend or rows, the compared settings, and the main trend or result.
  If the passages lack exact values, describe what they do state and say what is missing.
- If the question spans several papers (e.g. "across the papers", "the nine papers"),
  cover every paper that appears in the passages, attribute each point to its paper by
  name, and end by listing the papers for which no relevant passage was supplied.
- If asked for inconsistent or contradictory findings, compare what the papers report on
  the same topic (e.g. model rankings, effect of resolution, OCR, model size, prompting)
  and explain whether differences in benchmarks, models or settings account for them.
  If no direct contradiction is visible, say so and describe the closest tensions instead.
- If the passages only partly answer the question, answer that part and say what is missing.
  Partial answers are preferred over refusing.
- Set "insufficient_evidence" to true only if no passage is related to the question's
  topic at all; then set "answer" to exactly: "{insufficient}"

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
    budget = evidence.DEFAULT_MAX_CONTEXT_TOKENS
    try:
        targets = _cross_paper_targets(question, paper_ids)
        if targets:
            passages = _passages_per_paper(question, targets)
            budget = CROSS_PAPER_MAX_CONTEXT_TOKENS
        else:
            passages = get_passages(question, top_k=top_k, paper_ids=paper_ids)
    except SourceError as e:
        return _result("error", error=str(e))
    return answer_from_passages(question, passages, llm=llm, verify=verify,
                                max_context_tokens=budget)


# Questions about the whole collection need evidence from every paper; a single global
# top-k search tends to return passages from only one or two of them.
_CROSS_PAPER_RE = re.compile(
    r"\b(across|all (?:the |nine |\d+ )?papers|(?:nine|\d+|several|multiple|these|the) papers"
    r"|(?:paper )?collection|each paper|every paper|corpus|literature|synthesi[sz]e|synthesis"
    r"|research directions?|future (?:work|research)|research gaps?|recurring"
    r"|inconsisten\w*|contradict\w*)\b",
    re.IGNORECASE,
)
PASSAGES_PER_PAPER = 3
CROSS_PAPER_MAX_CONTEXT_TOKENS = 12000


def _cross_paper_targets(question: str, paper_ids: Optional[List[str]]) -> List[str]:
    """Papers to search one by one for a collection-wide question, or [] for a normal search."""
    if not _CROSS_PAPER_RE.search(question):
        return []
    if paper_ids:
        return list(paper_ids) if len(paper_ids) > 1 else []
    # No selection: "the papers" means the team catalogue, not ad-hoc user uploads.
    return sorted(_catalogue_ids() or [])


def _passages_per_paper(question: str, paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Top passages from each paper, interleaved by rank so the budget trims evenly."""
    per_paper = [get_passages(question, top_k=PASSAGES_PER_PAPER, paper_ids=[pid])
                 for pid in paper_ids]
    longest = max((len(p) for p in per_paper), default=0)
    return [p[rank] for rank in range(longest) for p in per_paper if rank < len(p)]
