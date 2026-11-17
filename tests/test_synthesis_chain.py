"""
Deterministic tests for the literature synthesis chain: input validation only.

No LLM or retrieval is involved — NeverCalledLLM fails the test if the model is reached.
"""
import pytest

from src.chains import synthesis_chain as sc


class NeverCalledLLM:
    def generate(self, *args, **kwargs):
        raise AssertionError("the model must not be called for invalid input")


@pytest.mark.parametrize("ids,message", [
    (["p01"], "at least two"), (["p01", "p01"], "at least two"),
    ("p01,p02", "list"), (["p01", ""], "empty"),
])
def test_invalid_selections_are_rejected_before_any_call(ids, message):
    res = sc.synthesize_from_passages(ids, {}, llm=NeverCalledLLM())
    assert res["status"] == "error" and message in res["error"].lower()
    assert "synthesize" in res["error"] or message != "at least two"
    assert res["disclaimer"] == sc.DISCLAIMER


def test_unknown_catalogue_id_is_rejected(monkeypatch):
    monkeypatch.setattr(sc, "known_paper_ids", lambda: frozenset({"p01", "p02"}))
    res = sc.synthesize(["p01", "p99"], llm=NeverCalledLLM())
    assert res["status"] == "error" and "p99" in res["error"]


def test_papers_without_passages_give_insufficient_evidence():
    res = sc.synthesize_from_passages(["p01", "p02"], {"p01": [], "p02": []}, llm=NeverCalledLLM())
    assert res["status"] == "insufficient_evidence"
    assert res["papers_without_evidence"] == ["p01", "p02"]
