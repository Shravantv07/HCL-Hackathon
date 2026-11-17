"""
Deterministic query router for ResearchIQ.

Classifies incoming research queries and dispatches them to the existing
hybrid `Retriever` with route-appropriate parameters and metadata.
"""
import re
from typing import Any, Dict, List, Literal, Optional

from src.config import get_settings
from src.retrieval.retriever import Retriever
from src.utils.citations import format_citation

QueryType = Literal["factual", "comparison", "visual", "synthesis"]
VALID_ROUTES = frozenset({"factual", "comparison", "visual", "synthesis"})

_VISUAL_RE = re.compile(
    r"\b(fig(?:ure)?s?|charts?|diagrams?|plots?|graphs?|tables?|images?|visuals?|illustrations?|screenshots?|depicted|shown in)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differences? between|differ|contrast|similarities|outperform|better than)\b",
    re.IGNORECASE,
)
_SYNTHESIS_RE = re.compile(
    r"\b(synthesi[sz]e|synthesis|across (?:all )?papers|overview|survey|literature|common themes|research gaps?|future work|trends|state of the art|all papers)\b",
    re.IGNORECASE,
)


def classify_query(question: str) -> QueryType:
    """
    Classify a research question using deterministic keyword heuristics.

    Returns one of: "factual" | "comparison" | "visual" | "synthesis".
    """
    if not question or not question.strip():
        raise ValueError("Query cannot be empty.")

    if _VISUAL_RE.search(question):
        return "visual"
    if _COMPARISON_RE.search(question):
        return "comparison"
    if _SYNTHESIS_RE.search(question):
        return "synthesis"
    return "factual"


class QueryRouter:
    """
    Routes research queries to the existing `Retriever` and returns a
    structured response with preserved metadata, citations, and warnings.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        self.retriever = retriever
        self.default_top_k = (
            default_top_k
            if default_top_k is not None
            else get_settings().top_k
        )
        if self.default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero.")

    def _get_retriever(self) -> Retriever:
        if self.retriever is None:
            try:
                self.retriever = Retriever()
            except Exception as exc:
                raise RuntimeError(f"Retriever unavailable: {exc}") from exc
        return self.retriever

    def route(
        self,
        query: str,
        top_k: Optional[int] = None,
        paper_ids: Optional[List[str]] = None,
        route_override: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Classify the query (unless overridden), retrieve evidence via `Retriever`,
        and return a structured result dictionary.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if route_override is not None:
            if route_override not in VALID_ROUTES:
                raise ValueError(
                    f"Unsupported route: {route_override!r}. "
                    f"Must be one of {sorted(VALID_ROUTES)}."
                )
            selected_route: QueryType = route_override  # type: ignore[assignment]
        else:
            selected_route = classify_query(query)

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if top_k is not None:
            effective_top_k = top_k
        elif selected_route == "comparison":
            effective_top_k = max(self.default_top_k, 8)
        elif selected_route == "synthesis":
            effective_top_k = max(self.default_top_k, 10)
        else:
            effective_top_k = self.default_top_k

        warnings: List[str] = []
        if selected_route == "visual":
            warnings.append(
                "Visual understanding is not implemented in the retrieval layer; "
                "returning text and figure/table caption chunks only."
            )

        effective_rerank = (
            rerank if rerank is not None else get_settings().rerank_enabled
        )

        retriever = self._get_retriever()
        try:
            raw_results = retriever.retrieve(
                query=query,
                top_k=effective_top_k,
                paper_ids=paper_ids,
                rerank=effective_rerank,
            )
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        results: List[Dict[str, Any]] = []
        seen_papers: List[str] = []

        for chunk in raw_results:
            item = chunk.copy()
            if "citation" not in item:
                item["citation"] = format_citation(item)
            results.append(item)

            pid = item.get("paper_id")
            if pid and pid not in seen_papers:
                seen_papers.append(pid)

        if not results:
            warnings.append("No matching evidence found for the query.")

        has_visual = any(
            item.get("chunk_type") in ("figure", "table") or bool(item.get("figure_ref"))
            for item in results
        )

        return {
            "query": query,
            "route": selected_route,
            "results": results,
            "paper_ids": seen_papers,
            "requested_paper_ids": paper_ids,
            "metadata": {
                "top_k": effective_top_k,
                "rerank": effective_rerank,
                "result_count": len(results),
                "has_visual_candidates": has_visual,
                "warnings": warnings,
            },
        }
