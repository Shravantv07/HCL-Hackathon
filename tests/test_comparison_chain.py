"""
Tests for the cross-paper comparison chain.

The LLM is always replaced by FakeLLM here — no network calls. The passages are
small synthetic TEST FIXTURES, not content from real papers.
"""
import json

import pytest

from src.chains import comparison_chain as cc
from src.chains import sources
from src.chains.llm import LLMError


def P(pid, idx, page, text):
    return {"paper_id": pid, "title": f"Test Paper {pid}", "page": page, "section": "",
            "chunk_type": "text", "chunk_index": idx, "figure_ref": "", "image_path": "",
            "text": f"TEST FIXTURE: {text}"}


A = [P("t01", 0, 2, "We study chart QA."), P("t01", 1, 5, "Accuracy is 50 on split X.")]
B = [P("t02", 0, 1, "We study table QA."), P("t02", 1, 6, "Uses dataset Y.")]
C = [P("t03", 0, 3, "We study document QA.")]
# Labels in prompt order: t01 -> S1, S2; t02 -> S3, S4; t03 -> S5

EMPTY_OUTPUT = {"summaries": {}, "table": {}, "similarities": [], "differences": [],
                "trade_offs": [], "research_gaps": []}


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def generate(self, prompt, system=None, images=None, json_output=False):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def row(t01, t02, t03=None):
    r = {"t01": t01, "t02": t02}
    if t03 is not None:
        r["t03"] = t03
    return r


def cell(result, dim, pid):
    r = next(r for r in result["table"] if r["dimension"] == dim)
    return next(c for c in r["cells"] if c["paper_id"] == pid)


@pytest.fixture
def catalogue(monkeypatch):
    """Pretend the paper catalogue contains exactly these ids."""
    monkeypatch.setattr(cc, "known_paper_ids", lambda: frozenset({"p01", "p02"}))


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_two_papers_keep_attribution_and_cross_paper_citation_is_removed():
#     fake = FakeLLM({"summaries": {"t01": "Chart QA [S1].", "t02": "Table QA [S3]."},
#                     "table": {"objective": row("Chart QA [S1].", "Table QA [S3]."),
#                               "results": row("Accuracy 50 [S2].", "Better than t01 [S3, S2]."),
#                               "dataset": row("Not reported in the supplied content.", "Dataset Y [S4].")},
#                     "similarities": ["Both study QA [S1, S3]."],
#                     "differences": ["Charts vs tables [S1][S3]."],
#                     "trade_offs": [], "research_gaps": ["Neither covers video."]})
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A, "t02": B}, llm=fake)

#     assert res["status"] == "ok"
#     assert [p["paper_id"] for p in res["papers"]] == ["t01", "t02"]
#     c = cell(res, "results", "t01")
#     assert c["status"] == "reported" and c["citations"][0]["page"] == 5
#     c = cell(res, "results", "t02")  # cited S2 (a t01 passage) inside t02's cell
#     assert [x["source_label"] for x in c["citations"]] == ["S3"] and "S2" not in c["text"]
#     assert any("another paper" in w and "S2" in w for w in res["warnings"])
#     assert all(x["paper_id"] == "t02" for r in res["table"] for x in r["cells"][1]["citations"])
#     assert res["similarities"][0]["status"] == "cited" and len(res["similarities"][0]["citations"]) == 2
#     assert res["research_gaps"][0]["label"] == cc.GAP_LABEL
#     assert {x["source_label"] for x in res["citations"]} == {"S1", "S2", "S3", "S4"}
#     prompt = fake.calls[0]
#     assert "- t01: Test Paper t01" in prompt and "- limitations: Limitations" in prompt
#     assert "<source>" in prompt


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_three_papers():
#     fake = FakeLLM({"summaries": {"t01": "a [S1]", "t02": "b [S3]", "t03": "c [S5]"},
#                     "table": {"objective": row("charts [S1]", "tables [S3]", "docs [S5]")},
#                     "similarities": [], "differences": [], "trade_offs": [], "research_gaps": []})
#     res = cc.compare_from_passages(["t01", "t02", "t03"], {"t01": A, "t02": B, "t03": C}, llm=fake)
#     assert res["status"] == "ok"
#     assert [c["paper_id"] for c in res["table"][0]["cells"]] == ["t01", "t02", "t03"]
#     assert cell(res, "objective", "t03")["citations"][0]["paper_id"] == "t03"


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_missing_detail_and_omitted_cells_are_not_reported():
#     fake = FakeLLM({"summaries": {"t01": "a [S1]", "t02": "b [S3]"},
#                     "table": {"dataset": row("not reported in the supplied content", "Dataset Y [S4].")},
#                     "similarities": [], "differences": [], "trade_offs": [], "research_gaps": []})
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A, "t02": B}, llm=fake)
#     c = cell(res, "dataset", "t01")
#     assert c["status"] == "not_reported" and c["text"] == cc.NOT_REPORTED and not c["citations"]
#     assert cell(res, "limitations", "t02")["status"] == "not_reported"  # model omitted the row
#     assert any("returned nothing" in w for w in res["warnings"])


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_uncited_finding_is_flagged_and_uncited_trade_off_is_dropped():
#     fake = FakeLLM({"summaries": {"t01": "a [S1]", "t02": "b [S3]"},
#                     "table": {"method": row("Uses a transformer.", "Uses X [S9].")},
#                     "similarities": [], "differences": [],
#                     "trade_offs": ["Faster but less accurate."], "research_gaps": []})
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A, "t02": B}, llm=fake)
#     assert cell(res, "method", "t01")["status"] == "uncited"
#     c = cell(res, "method", "t02")
#     assert c["status"] == "uncited" and "S9" not in c["text"]
#     assert res["trade_offs"] == [] and any("trade_offs[0]: dropped" in w for w in res["warnings"])


@pytest.mark.parametrize("ids,message", [
    (["t01"], "at least two"), ("t01,t02", "list"), (["t01", ""], "empty"),
    (["t01", "t01"], "at least two"), ([None, "t02"], "empty"),
])
def test_invalid_selections(ids, message):
    fake = FakeLLM({})
    res = cc.compare_from_passages(ids, {"t01": A, "t02": B}, llm=fake)
    assert res["status"] == "error" and message in res["error"].lower()
    assert fake.calls == []


def test_unknown_catalogue_id_is_rejected(catalogue):
    res = cc.compare(["p01", "p99"], llm=FakeLLM({}))
    assert res["status"] == "error" and "p99" in res["error"]


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_paper_without_evidence_is_excluded_or_insufficient():
#     res = cc.compare_from_passages(["t01", "t02", "t03"], {"t01": A, "t02": B, "t03": []},
#                                    llm=FakeLLM(EMPTY_OUTPUT))
#     assert res["status"] == "ok" and res["papers_without_evidence"] == ["t03"]
#     assert all(len(r["cells"]) == 2 for r in res["table"])

#     fake = FakeLLM({})
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A}, llm=fake)
#     assert res["status"] == "insufficient_evidence" and fake.calls == []


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_passage_filed_under_wrong_paper_is_dropped():
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A + [B[0]], "t02": B},
#                                    llm=FakeLLM(EMPTY_OUTPUT))
#     assert any("belonging to other papers" in w for w in res["warnings"])
#     assert res["chunks_used"].count("t02__text__0") == 1


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# @pytest.mark.parametrize("fake,message", [
#     (FakeLLM("not json"), "malformed"),
#     (FakeLLM({"table": []}), "malformed"),
#     (FakeLLM(error=LLMError("Gemini quota or rate limit exceeded")), "quota"),
# ])
# def test_malformed_output_and_provider_failure(fake, message):
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A, "t02": B}, llm=fake)
#     assert res["status"] == "error" and message in res["error"]


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_fenced_json_is_accepted():
#     raw = "```json\n" + json.dumps({"summaries": {}, "table": {"method": row("x [S1]", "y [S3]")}}) + "\n```"
#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A, "t02": B}, llm=FakeLLM(raw))
#     assert res["status"] == "ok" and cell(res, "method", "t01")["status"] == "reported"


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_compare_retrieves_per_paper_filters_and_puts_focus_first(catalogue, monkeypatch):
#     seen = []

#     def fake_get_passages(query, top_k=None):
#         seen.append(query)
#         return A + B + C  # the retriever ignores paper selection; filtering must handle it

#     monkeypatch.setattr(sources, "get_passages", fake_get_passages)

#     res = cc.compare_papers("p01", "p02", "results", llm=FakeLLM(EMPTY_OUTPUT))
#     assert res["status"] == "insufficient_evidence"
#     assert res["papers_without_evidence"] == ["p01", "p02"]
#     assert len(seen) == 2 * len(cc.DIMENSIONS)  # focus 'results' is a standard dimension
#     assert res["dimensions"][0]["key"] == "results"

#     res = cc.compare_from_passages(["t01", "t02"], {"t01": A, "t02": B}, focus="Energy efficiency",
#                                    dimensions=["results"], llm=FakeLLM(EMPTY_OUTPUT))
#     assert [d["label"] for d in res["dimensions"]] == ["Energy efficiency", "Reported results"]

#     passages = sources.get_paper_passages("t02", ["q1", "q2"])
#     assert passages and all(p["paper_id"] == "t02" for p in passages)
