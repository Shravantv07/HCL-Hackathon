"""
Evidence handling shared by the chains.

Passages are given short labels ([S1], [S2], ...) in the prompt. The model cites
labels only; this module maps labels back to the real passage records, so a
citation can never point to a paper, page, or chunk that was not supplied.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.chains.sources import normalize_passage
from src.utils.citations import format_citation
from src.utils.security import wrap_source_content

DEFAULT_MAX_CONTEXT_TOKENS = 6000

_CITE_GROUP_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_LABEL_RE = re.compile(r"S\d+")


def prepare_passages(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize records, dropping unusable or duplicate ones with a warning for each."""
    passages, warnings, seen = [], [], set()
    for i, record in enumerate(records or []):
        try:
            p = normalize_passage(record)
        except (ValueError, AttributeError, TypeError) as e:
            warnings.append(f"Skipped retrieved record #{i + 1}: {e}.")
            continue
        key = p["chunk_id"] or (p["paper_id"], p["page"], p["text"])
        if key in seen:
            continue
        seen.add(key)
        passages.append(p)
    return passages, warnings


def fit_to_budget(
    passages: List[Dict[str, Any]], max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Keep passages in their given (retrieval-rank) order until the approximate
    token budget is reached. Returns (kept, omitted); callers report omissions.
    The first passage is always kept so a single long chunk is never dropped.
    """
    kept, omitted, used = [], [], 0
    for p in passages:
        cost = len(p["text"].split())
        if kept and used + cost > max_tokens:
            omitted.append(p)
            continue
        kept.append(p)
        used += cost
    return kept, omitted


def label_passages(passages: List[Dict[str, Any]], prefix: str = "S") -> Dict[str, Dict[str, Any]]:
    return {f"{prefix}{i}": p for i, p in enumerate(passages, start=1)}


def build_context(labelled: Dict[str, Dict[str, Any]]) -> str:
    """Render labelled passages as <source> blocks with their provenance."""
    blocks = []
    for label, p in labelled.items():
        header = [f"label: {label}", f"paper: {p['title']} (id: {p['paper_id']})"]
        header.append(f"page: {p['page'] if p['page'] is not None else 'unknown'}")
        if p["section"]:
            header.append(f"section: {p['section']}")
        if p["chunk_type"] != "text":
            header.append(f"type: {p['chunk_type']}" + (f" ({p['figure_ref']})" if p["figure_ref"] else ""))
        blocks.append(wrap_source_content("\n".join(header) + "\n---\n" + p["text"]))
    return "\n\n".join(blocks)


def citation_record(label: str, p: Dict[str, Any]) -> Dict[str, Any]:
    """UI-facing citation for one supplied passage (`paper`/`page` keys match the frontend)."""
    return {
        "source_label": label,
        "chunk_id": p["chunk_id"],
        "paper_id": p["paper_id"],
        "paper": p["title"],
        "page": p["page"],
        "section": p["section"],
        "fig_ref": p["figure_ref"] or None,
        "text": p["text"],
        "display": format_citation({"title": p["title"], "page": p["page"] if p["page"] is not None else "?",
                                    "figure_ref": p["figure_ref"]}),
    }


def cited_labels(text: str) -> List[str]:
    """Labels cited in `text`, in first-appearance order, e.g. '[S1] ... [S2, S3]'."""
    labels: List[str] = []
    for group in _CITE_GROUP_RE.findall(text or ""):
        for label in _LABEL_RE.findall(group):
            if label not in labels:
                labels.append(label)
    return labels


def extract_citations(
    text: str, labelled: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (citations for valid labels, labels the model used that were never supplied)."""
    citations, unknown = [], []
    for label in cited_labels(text):
        if label in labelled:
            citations.append(citation_record(label, labelled[label]))
        else:
            unknown.append(label)
    return citations, unknown


def resolve_citations(
    text: str,
    labelled: Dict[str, Dict[str, Any]],
    allowed_paper_ids: Optional[Iterable[str]] = None,
) -> Tuple[str, List[Dict[str, Any]], List[str], List[str]]:
    """
    Map the labels in `text` to citations and clean the text.

    If `allowed_paper_ids` is given, citations to passages from any other paper are
    rejected (used to keep comparison cells attributed to the right paper).
    Returns (clean_text, citations, unknown_labels, wrong_paper_labels).
    """
    citations, unknown = extract_citations(text, labelled)
    wrong: List[str] = []
    if allowed_paper_ids is not None:
        allowed = set(allowed_paper_ids)
        wrong = [c["source_label"] for c in citations if c["paper_id"] not in allowed]
        citations = [c for c in citations if c["source_label"] not in wrong]
    return strip_labels(text, unknown + wrong), citations, unknown, wrong


def strip_labels(text: str, labels: List[str]) -> str:
    """Remove specific (invalid) labels from answer prose, leaving valid ones intact."""
    if not labels:
        return text
    bad = set(labels)

    def _clean(match: "re.Match[str]") -> str:
        keep = [l for l in _LABEL_RE.findall(match.group(1)) if l not in bad]
        return f"[{', '.join(keep)}]" if keep else ""

    return re.sub(r"[ \t]{2,}", " ", _CITE_GROUP_RE.sub(_clean, text)).strip()
