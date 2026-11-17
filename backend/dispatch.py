"""
Routes a chat question to the right Member 3 chain and renders the result for the UI.

    route_question(question, paper_ids, image_paths, history, verify) -> response dict

Routing (src/retrieval/router.classify_query + attachments):
    image attached  -> visual_chain.answer_about_figure   (the uploaded image itself)
    "visual"        -> visual_chain.answer_visual         (figures in the papers; falls back to QA)
    "comparison"    -> comparison_chain.compare           (selected or named papers; QA if < 2)
    "synthesis"     -> synthesis_chain.synthesize         (selected papers, else the catalogue)
    "factual"       -> qa_chain.answer

Every response has the same shape:
    {answer (markdown), status, route, resolved_question, sources: [citation...],
     warnings, verification}
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chains import comparison_chain, qa_chain, synthesis_chain, visual_chain
from src.chains.llm import LLMError, get_client
from src.chains.sources import _catalogue_ids, find_mentioned_papers
from src.retrieval.router import classify_query

REWRITE_SYSTEM = "You rewrite follow-up questions into standalone questions. Output only the question."
REWRITE_PROMPT = """Conversation so far (most recent last):
{history}

Follow-up question: {question}

If the follow-up depends on the conversation (pronouns such as "it", "they", "this paper",
"the second one", or omitted subjects), rewrite it as one standalone question that names
what it refers to. If it is already standalone, return it unchanged. Output only the question."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resolve_followup(question: str, history: List[Dict[str, str]]) -> str:
    """Rewrite a follow-up into a standalone question using recent turns (best effort)."""
    turns = [h for h in (history or []) if (h.get("content") or "").strip()][-4:]
    if not turns:
        return question
    rendered = "\n".join(f"{h.get('role', 'user')}: {h['content'].strip()[:800]}" for h in turns)
    try:
        rewritten = get_client().generate(REWRITE_PROMPT.format(history=rendered, question=question),
                                          system=REWRITE_SYSTEM).strip().strip('"')
    except LLMError:
        return question
    return rewritten if 3 <= len(rewritten) <= 500 else question


def _cell_md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", "<br>")


def _response(route: str, question: str, result: Dict[str, Any], answer_md: str,
              extra_warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    status = result.get("status", "error")
    if status == "error":
        answer_md = f"⚠️ {result.get('error') or 'Something went wrong.'}"
    elif not answer_md.strip():
        answer_md = result.get("error") or "No answer was produced."
    return {
        "answer": answer_md,
        "status": status,
        "route": route,
        "resolved_question": question,
        "sources": result.get("citations", []),
        "warnings": (extra_warnings or []) + result.get("warnings", []),
        "verification": result.get("verification"),
    }


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

def _render_comparison(r: Dict[str, Any]) -> str:
    if r["status"] != "ok":
        return r.get("error") or ""
    papers = r["papers"]
    lines = ["**Overview**"]
    lines += [f"- **{p['title']}** — {p['summary']['text']}" for p in papers]
    header = "| | " + " | ".join(_cell_md(p["title"].split(":")[0]) for p in papers) + " |"
    lines += ["", header, "|" + "---|" * (len(papers) + 1)]
    for row in r["table"]:
        lines.append(f"| **{row['label']}** | " + " | ".join(_cell_md(c["text"]) for c in row["cells"]) + " |")
    for key, title in (("similarities", "Similarities"), ("differences", "Differences"),
                       ("trade_offs", "Trade-offs")):
        if r.get(key):
            lines += ["", f"**{title}**"] + [f"- {s['text']}" for s in r[key]]
    if r.get("research_gaps"):
        lines += ["", f"**Possible research gaps** _{comparison_chain.GAP_LABEL}_"]
        lines += [f"- {g['text']}" for g in r["research_gaps"]]
    return "\n".join(lines)


def _render_synthesis(r: Dict[str, Any]) -> str:
    if r["status"] != "ok":
        return r.get("error") or ""
    lines = [r["synthesis"]["text"], "", "**Per-paper summaries**"]
    lines += [f"- **{p['title']}** — {p['summary']['text']}" for p in r["papers"]]
    if r.get("research_gaps"):
        lines += ["", f"**Possible research gaps** _{comparison_chain.GAP_LABEL}_"]
        lines += [f"- {g['text']}" for g in r["research_gaps"]]
    lines += ["", f"_{r['disclaimer']}_"]
    return "\n".join(lines)


def _render_visual(r: Dict[str, Any], uploaded: bool) -> str:
    if r["status"] == "error":
        return ""
    src = r.get("source") or {}
    lines = [r.get("answer", "")]
    if not uploaded and (src.get("fig_id") or src.get("paper")):
        where = ", ".join(x for x in (src.get("paper"), f"p.{src['page']}" if src.get("page") else None,
                                      src.get("fig_id")) if x)
        lines += ["", f"🖼️ _Figure: {where}_" + (f" — “{src['caption'][:200]}”" if src.get("caption") else "")]
    if r.get("readability") and r["readability"] != "clear":
        lines.append(f"_Image readability: {r['readability'].replace('_', ' ')}_")
    if r.get("unclear_elements"):
        lines.append("_Could not read: " + "; ".join(r["unclear_elements"]) + "_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def route_question(question: str, paper_ids: Optional[List[str]] = None,
                   image_paths: Optional[List[Path]] = None,
                   history: Optional[List[Dict[str, str]]] = None,
                   verify: bool = False) -> Dict[str, Any]:
    q = resolve_followup(question, history or [])
    paper_ids = list(dict.fromkeys(paper_ids or [])) or None

    # 1. An attached image: answer from the image itself.
    if image_paths:
        path = image_paths[0]
        r = visual_chain.answer_about_figure(q, {"image_bytes": path.read_bytes()})
        warn = [f"Only the first of {len(image_paths)} attached images was analysed."] if len(image_paths) > 1 else []
        return _response("visual", q, r, _render_visual(r, uploaded=True), warn)

    route = classify_query(q)

    if route == "comparison":
        ids = paper_ids if paper_ids and len(paper_ids) >= 2 else \
            list(dict.fromkeys((paper_ids or []) + find_mentioned_papers(q)))
        if len(ids) >= 2:
            r = comparison_chain.compare(ids, verify=verify)
            return _response("comparison", q, r, _render_comparison(r))
        note = ["Name (or select) at least two papers for a side-by-side comparison; "
                "answered as a regular question instead."]
        r = qa_chain.answer(q, paper_ids=paper_ids, verify=verify)
        return _response("factual", q, r, r.get("answer", ""), note)

    if route == "synthesis":
        ids = paper_ids if paper_ids and len(paper_ids) >= 2 else sorted(_catalogue_ids() or [])
        r = synthesis_chain.synthesize(ids, topic=q, verify=verify)
        return _response("synthesis", q, r, _render_synthesis(r))

    if route == "visual":
        scope = paper_ids or find_mentioned_papers(q) or None
        r = visual_chain.answer_visual(q, paper_ids=scope)
        if r["status"] == "ok":
            return _response("visual", q, r, _render_visual(r, uploaded=False))
        # No usable figure image: the QA chain can still explain figures from captions/text.
        note = [f"No figure image could be analysed ({r.get('error') or r.get('answer') or r['status']}); "
                "answered from the paper text instead."]
        r = qa_chain.answer(q, paper_ids=scope, verify=verify)
        return _response("factual", q, r, r.get("answer", ""), note)

    r = qa_chain.answer(q, paper_ids=paper_ids, verify=verify)
    return _response("factual", q, r, r.get("answer", ""))
