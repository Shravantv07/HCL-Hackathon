"""
Structured cross-paper comparison chain.

    compare_papers(paper_id_a, paper_id_b, aspect)       # original two-paper interface
    compare(paper_ids, dimensions=None, focus=None)      # two or more papers, retrieves per paper
    compare_from_passages(paper_ids, passages_by_paper)  # same, from passages you already have

Result:
    {
        status: "ok" | "insufficient_evidence" | "error",
        papers: [{paper_id, title, summary: Cell}],
        dimensions: [{key, label}],
        table: [{dimension, label, cells: [Cell, ...]}],   # one Cell per paper, in paper order
        similarities | differences | trade_offs: [{text, citations, status}],
        research_gaps: [{text, citations, label}],         # labelled as AI inference
        citations: [citation, ...],                        # every citation used, deduplicated
        chunks_used, chunks_omitted, papers_without_evidence, warnings, error,
        verification: dict | None,                         # when verify=True (verification_chain)
    }
    Cell = {paper_id, text, status: "reported" | "not_reported" | "uncited", citations}

Every passage keeps its paper_id end to end. A cell may only cite passages from
its own paper; any other citation is removed and reported in `warnings`.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.chains import evidence
from src.chains.llm import LLMError, get_client
from src.chains.sources import SourceError, get_paper_passages, known_paper_ids, paper_title

NOT_REPORTED = "Not reported in the supplied content."
GAP_LABEL = "(AI-generated suggestion, not established fact)"
DEFAULT_MAX_CONTEXT_TOKENS = 8000

# key -> (display label, retrieval query)
DIMENSIONS: Dict[str, Tuple[str, str]] = {
    "objective": ("Research problem / objective", "research problem objective motivation contributions"),
    "method": ("Proposed method or architecture", "proposed method model architecture approach"),
    "dataset": ("Datasets", "datasets benchmark data used for training and evaluation"),
    "setup": ("Training / experimental setup", "training details experimental setup implementation"),
    "metrics": ("Evaluation metrics", "evaluation metrics measured"),
    "results": ("Reported results", "results performance compared with baselines"),
    "limitations": ("Limitations", "limitations weaknesses future work"),
}

# Aspect names used by the UI (app/streamlit_app.py) and common variants
_ALIASES = {
    "methodology": "method", "methods": "method", "architecture": "method", "model": "method",
    "datasets": "dataset", "data": "dataset", "result": "results", "performance": "results",
    "limitation": "limitations", "objectives": "objective", "problem": "objective",
    "evaluation": "metrics", "metric": "metrics", "experiments": "setup", "training": "setup",
}

COMPARE_INSTRUCTIONS = """Compare the research papers below using only the supplied passages.

Papers (use these exact ids as JSON keys):
{paper_list}

Comparison dimensions (use these exact keys):
{dimension_list}
{focus_line}
Citation format for this task (this replaces the [Paper Title, p.X] format): cite passage
labels in square brackets right after each claim, e.g. [S2] or [S1, S4]. A summary or table
cell about a paper may cite ONLY passages from that paper. Use only labels that appear below.

Rules:
- Do not invent facts, numbers, datasets, methods, results, or conclusions.
- If a paper's passages say nothing about a dimension, write exactly "{not_reported}"
  for that cell. Do not guess or fill in typical values.
- Similarities and differences must be based on cited passages from the papers involved.
- Include a trade-off only if the passages support it; otherwise return an empty list.
- Research gaps are your own analysis, not the authors' claims; base them on what the
  passages show is missing or limited.
- Keep each cell to 1-3 sentences.

Respond with JSON only, in this shape:
{{
  "summaries": {{"<paper_id>": "<2-3 sentence overview with [S#] citations>"}},
  "table": {{"<dimension_key>": {{"<paper_id>": "<finding with [S#] citations, or the not-reported text>"}}}},
  "similarities": ["<statement with [S#] citations>"],
  "differences": ["<statement with [S#] citations>"],
  "trade_offs": ["<statement with [S#] citations>"],
  "research_gaps": ["<statement>"]
}}

Passages:
{context}"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _result(status: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "status": status, "papers": [], "dimensions": [], "table": [],
        "similarities": [], "differences": [], "trade_offs": [], "research_gaps": [],
        "citations": [], "chunks_used": [], "chunks_omitted": [],
        "papers_without_evidence": [], "warnings": [], "error": None, "verification": None,
    }
    out.update(extra)
    return out


def _validate_ids(paper_ids: Any, check_catalogue: bool) -> Tuple[List[str], Optional[str]]:
    if isinstance(paper_ids, str) or not isinstance(paper_ids, Iterable):
        return [], "paper_ids must be a list of paper ids."
    ids: List[str] = []
    for raw in paper_ids:
        pid = str(raw).strip() if raw is not None else ""
        if not pid:
            return [], "paper_ids contains an empty id."
        if pid not in ids:
            ids.append(pid)
    if len(ids) < 2:
        return ids, "Select at least two different papers to compare."
    if check_catalogue:
        known = known_paper_ids()
        unknown = [pid for pid in ids if known is not None and pid not in known]
        if unknown:
            return ids, f"Unknown paper id(s): {', '.join(unknown)}."
    return ids, None


def _resolve_dimensions(
    dimensions: Optional[List[str]], focus: Optional[str]
) -> List[Tuple[str, str, str]]:
    """Return [(key, label, query)], focus first. Unknown names become custom dimensions."""
    def one(name: str) -> Tuple[str, str, str]:
        raw = name.strip()
        key = _ALIASES.get(raw.lower(), raw.lower())
        if key in DIMENSIONS:
            return (key, *DIMENSIONS[key])
        slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_") or "custom"
        return (slug, raw, raw)

    names = [focus] if focus and focus.strip() else []
    names += [d for d in (dimensions or list(DIMENSIONS)) if d and d.strip()]
    resolved, seen = [], set()
    for name in names:
        dim = one(name)
        if dim[0] not in seen:
            seen.add(dim[0])
            resolved.append(dim)
    return resolved


def _load_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output (tolerates ```json fences); None if malformed."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_GAP_LABEL_RE = re.compile(
    r"\(\s*AI[- ]generated suggestion[^)]*\)\s*[:.\-–]?\s*"                       # "(AI-generated ...)"
    r"|^\s*AI[- ]generated suggestion(?:,\s*not established fact)?\s*[:.\-–]\s*",  # "AI-generated suggestion: "
    re.IGNORECASE)


def strip_gap_label(text: str) -> str:
    """Remove a copy of the AI-suggestion label the model may write into the gap text itself
    (the system prompt asks it to); the label is returned once, in the `label` field."""
    return _GAP_LABEL_RE.sub("", text or "").strip()


def _is_not_reported(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.lower()).strip().startswith("not reported")


def _cell(value: Any, pid: str, labelled: Dict[str, Dict[str, Any]], where: str,
          warnings: List[str]) -> Dict[str, Any]:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        warnings.append(f"{where}: the model returned nothing; shown as not reported.")
    if not text or _is_not_reported(text):
        return {"paper_id": pid, "text": NOT_REPORTED, "status": "not_reported", "citations": []}

    clean, citations, unknown, wrong = evidence.resolve_citations(text, labelled, [pid])
    if unknown:
        warnings.append(f"{where}: removed label(s) not matching any supplied passage: {', '.join(unknown)}.")
    if wrong:
        warnings.append(f"{where}: removed citation(s) to another paper's passages: {', '.join(wrong)}.")
    if not citations:
        warnings.append(f"{where}: finding has no citation to this paper's passages; treat as unverified.")
    return {"paper_id": pid, "text": clean, "status": "reported" if citations else "uncited",
            "citations": citations}


def _statements(values: Any, labelled: Dict[str, Dict[str, Any]], allowed: List[str],
                where: str, warnings: List[str], require_citation: bool = False) -> List[Dict[str, Any]]:
    out = []
    for i, value in enumerate(values if isinstance(values, list) else []):
        if not isinstance(value, str) or not value.strip():
            continue
        clean, citations, unknown, wrong = evidence.resolve_citations(value.strip(), labelled, allowed)
        if unknown or wrong:
            warnings.append(f"{where}[{i}]: removed invalid citation label(s): {', '.join(unknown + wrong)}.")
        if require_citation and not citations:
            warnings.append(f"{where}[{i}]: dropped because it cites no supplied passage.")
            continue
        out.append({"text": clean, "citations": citations, "status": "cited" if citations else "uncited"})
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def compare_from_passages(
    paper_ids: List[str],
    passages_by_paper: Dict[str, List[Dict[str, Any]]],
    dimensions: Optional[List[str]] = None,
    focus: Optional[str] = None,
    llm: Optional[Any] = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    verify: bool = False,
) -> Dict[str, Any]:
    """
    Compare papers using passages already grouped by paper id.
    With verify=True, a successful comparison also gets claim-level `verification`.
    """
    ids, error = _validate_ids(paper_ids, check_catalogue=False)
    if error:
        return _result("error", error=error)
    dims = _resolve_dimensions(dimensions, focus)
    dim_info = [{"key": k, "label": label} for k, label, _ in dims]
    warnings: List[str] = []

    # Collect each paper's passages; drop any passage whose paper_id disagrees with its group.
    kept_by_paper: Dict[str, List[Dict[str, Any]]] = {}
    omitted: List[Dict[str, Any]] = []
    usable_by_paper: Dict[str, List[Dict[str, Any]]] = {}
    for pid in ids:
        usable, w = evidence.prepare_passages((passages_by_paper or {}).get(pid) or [])
        warnings += [f"{pid}: {msg}" for msg in w]
        mismatched = [p for p in usable if p["paper_id"] != pid]
        if mismatched:
            warnings.append(f"{pid}: dropped {len(mismatched)} passage(s) belonging to other papers.")
        usable_by_paper[pid] = [p for p in usable if p["paper_id"] == pid]

    with_evidence = [pid for pid in ids if usable_by_paper[pid]]
    without = [pid for pid in ids if not usable_by_paper[pid]]
    if without:
        warnings.append(f"No passages found for: {', '.join(without)}; excluded from the comparison.")
    if len(with_evidence) < 2:
        return _result("insufficient_evidence", dimensions=dim_info, papers_without_evidence=without,
                       warnings=warnings,
                       error="At least two of the selected papers need retrievable content to compare.")

    per_paper_budget = max_context_tokens // len(with_evidence)
    for pid in with_evidence:
        kept, dropped = evidence.fit_to_budget(usable_by_paper[pid], per_paper_budget)
        kept_by_paper[pid] = kept
        omitted += dropped
    if omitted:
        warnings.append(f"{len(omitted)} lower-ranked passage(s) were left out to fit the context budget.")

    all_kept = [p for pid in with_evidence for p in kept_by_paper[pid]]
    labelled = evidence.label_passages(all_kept)
    titles = {pid: kept_by_paper[pid][0]["title"] for pid in with_evidence}
    bookkeeping = {
        "dimensions": dim_info,
        "papers_without_evidence": without,
        "chunks_used": [p["chunk_id"] for p in all_kept],
        "chunks_omitted": [p["chunk_id"] for p in omitted],
    }

    focus_key = dims[0][0] if focus and focus.strip() else None
    prompt = COMPARE_INSTRUCTIONS.format(
        paper_list="\n".join(f"- {pid}: {titles[pid]}" for pid in with_evidence),
        dimension_list="\n".join(f"- {k}: {label}" for k, label, _ in dims),
        focus_line=f"Pay particular attention to the '{focus_key}' dimension.\n" if focus_key else "",
        not_reported=NOT_REPORTED,
        context=evidence.build_context(labelled),
    )

    try:
        raw = (llm or get_client()).generate(prompt, json_output=True)
    except LLMError as e:
        return _result("error", error=str(e), warnings=warnings, **bookkeeping)

    data = _load_json_object(raw)
    if data is None or not isinstance(data.get("table"), dict) \
            or not isinstance(data.get("summaries", {}), dict):
        return _result("error", error="The model returned malformed comparison output.",
                       warnings=warnings, **bookkeeping)

    extra_ids = set(data.get("summaries", {})) | {p for row in data["table"].values()
                                                  if isinstance(row, dict) for p in row}
    extra_ids -= set(with_evidence)
    extra_dims = set(data["table"]) - {k for k, _, _ in dims}
    if extra_ids:
        warnings.append(f"Ignored output for paper id(s) not in the comparison: {', '.join(sorted(extra_ids))}.")
    if extra_dims:
        warnings.append(f"Ignored unrequested dimension(s): {', '.join(sorted(extra_dims))}.")

    papers = [{"paper_id": pid, "title": titles[pid],
               "summary": _cell(data.get("summaries", {}).get(pid), pid, labelled, f"summary[{pid}]", warnings)}
              for pid in with_evidence]

    table = []
    for key, label, _ in dims:
        row = data["table"].get(key)
        row = row if isinstance(row, dict) else {}
        table.append({"dimension": key, "label": label,
                      "cells": [_cell(row.get(pid), pid, labelled, f"table[{key}][{pid}]", warnings)
                                for pid in with_evidence]})

    similarities = _statements(data.get("similarities"), labelled, with_evidence, "similarities", warnings)
    differences = _statements(data.get("differences"), labelled, with_evidence, "differences", warnings)
    trade_offs = _statements(data.get("trade_offs"), labelled, with_evidence, "trade_offs", warnings,
                             require_citation=True)
    gaps = [{**g, "text": strip_gap_label(g["text"]), "label": GAP_LABEL}
            for g in _statements(data.get("research_gaps"), labelled, with_evidence, "research_gaps", warnings)]

    all_citations: Dict[str, Dict[str, Any]] = {}
    for item in [p["summary"] for p in papers] + [c for r in table for c in r["cells"]] \
            + similarities + differences + trade_offs + gaps:
        for c in item["citations"]:
            all_citations.setdefault(c["source_label"], c)

    result = _result(
        "ok", papers=papers, table=table, similarities=similarities, differences=differences,
        trade_offs=trade_offs, research_gaps=gaps,
        citations=sorted(all_citations.values(), key=lambda c: int(c["source_label"][1:])),
        warnings=warnings, **bookkeeping,
    )
    if verify:
        from src.chains.verification_chain import verify_comparison
        result["verification"] = verify_comparison(result, llm=llm)
    return result


def compare(
    paper_ids: List[str],
    dimensions: Optional[List[str]] = None,
    focus: Optional[str] = None,
    llm: Optional[Any] = None,
    top_k: Optional[int] = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    verify: bool = False,
) -> Dict[str, Any]:
    """Compare two or more papers, retrieving evidence for each dimension from each paper."""
    ids, error = _validate_ids(paper_ids, check_catalogue=True)
    if error:
        return _result("error", error=error)
    dims = _resolve_dimensions(dimensions, focus)

    passages_by_paper: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for pid in ids:
            title = paper_title(pid)
            queries = [f"{title} {query}".strip() for _, _, query in dims]
            passages_by_paper[pid] = get_paper_passages(pid, queries, top_k=top_k)
    except SourceError as e:
        return _result("error", error=str(e))

    return compare_from_passages(ids, passages_by_paper, dimensions=dimensions, focus=focus,
                                 llm=llm, max_context_tokens=max_context_tokens, verify=verify)


def compare_papers(paper_id_a: str, paper_id_b: str, aspect: str = "methodology",
                   llm: Optional[Any] = None, verify: bool = False) -> Dict[str, Any]:
    """
    Compare two papers. `aspect` (e.g. "methodology", "results", "datasets") is placed
    first and emphasised; the other standard dimensions are still compared.
    Returns the structured dict described in the module docstring.
    """
    return compare(paper_ids=[paper_id_a, paper_id_b], focus=aspect, llm=llm, verify=verify)
