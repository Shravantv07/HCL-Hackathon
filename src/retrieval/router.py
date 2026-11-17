"""
Classifies an incoming question into a query type so the right chain is used.
Types: factual | comparison | visual | synthesis
"""
from typing import Literal

QueryType = Literal["factual", "comparison", "visual", "synthesis"]


def classify_query(question: str) -> QueryType:
    """
    Use keyword heuristics (fast) or a cheap LLM call to classify the query.
    Returns one of: factual | comparison | visual | synthesis
    """
    # TODO: implement
    raise NotImplementedError
