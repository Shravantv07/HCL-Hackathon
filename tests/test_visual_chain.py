"""
Tests for the visual (figure) Q&A chain.

The LLM is always replaced by FakeLLM here — no network calls. Images are small
synthetic TEST FIXTURES drawn with Pillow, not figures from real papers.
"""
import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.chains import sources
from src.chains import visual_chain as vc
from src.chains.llm import LLMError

OK = {"answer": "Three boxes connected left to right.", "figure_type": "architecture_diagram",
      "readability": "clear", "unclear_elements": [], "insufficient_evidence": False}


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def generate(self, prompt, system=None, images=None, json_output=False):
        self.calls.append({"prompt": prompt, "images": images, "json_output": json_output})
        if self.error:
            raise self.error
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


# --- synthetic test images ----------------------------------------------------

def _diagram():
    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    for x, label in [(40, "Encoder"), (260, "Fusion"), (480, "Decoder")]:
        d.rectangle([x, 140, x + 120, 220], outline="black", width=3)
        d.text((x + 25, 172), label, fill="black")
    d.line([160, 180, 260, 180], fill="black", width=3)
    d.line([380, 180, 480, 180], fill="black", width=3)
    d.text((10, 10), "TEST FIXTURE diagram", fill="gray")
    return img


def _chart():
    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    for i, h in enumerate([120, 240, 180]):
        d.rectangle([80 + i * 160, 360 - h, 180 + i * 160, 360], fill="steelblue")
        d.text((100 + i * 160, 365), f"M{i + 1}", fill="black")
    d.text((10, 10), "TEST FIXTURE chart", fill="gray")
    return img


def _table():
    img = Image.new("RGB", (600, 240), "white")
    d = ImageDraw.Draw(img)
    for r, cells in enumerate([["Model", "Score"], ["A", "1"], ["B", "2"]]):
        for c, t in enumerate(cells):
            d.rectangle([20 + c * 280, 20 + r * 60, 300 + c * 280, 80 + r * 60], outline="black")
            d.text((30 + c * 280, 40 + r * 60), t, fill="black")
    return img


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    root = tmp_path_factory.mktemp("figures")

    def save(img, name, fmt=None):
        path = root / name
        img.save(path, format=fmt)
        return str(path)

    paths = {
        "diagram": save(_diagram(), "t01_p3_fig1.png"),
        "chart": save(_chart(), "t02_p5_fig1.jpeg", "JPEG"),
        "table": save(_table(), "t03_p2_fig1.png"),
        "tiny": save(Image.new("RGB", (60, 40), "white"), "tiny.png"),
        "bmp": save(_diagram(), "diagram.bmp", "BMP"),
        "huge": save(Image.new("RGB", (5000, 1000), "white"), "huge.png"),
        "missing": str(root / "nope.png"),
        "root": root,
    }
    corrupt = root / "corrupt.png"
    corrupt.write_bytes(Path(paths["diagram"]).read_bytes()[:200])  # truncated PNG
    paths["corrupt"] = str(corrupt)
    text_file = root / "notes.png"
    text_file.write_text("this is not an image")
    paths["text"] = str(text_file)
    return paths


def fig(path, **overrides):
    base = {"paper_id": "t01", "page": 3, "fig_id": "t01_p3_fig1", "image_path": path,
            "caption": "Figure 1: TEST FIXTURE caption.", "nearby_text": "TEST FIXTURE nearby text.",
            "width": 640, "height": 360}
    base.update(overrides)
    return base


# --- answer_about_figure --------------------------------------------------------

# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_architecture_diagram_sends_actual_image_and_preserves_metadata(images):
#     fake = FakeLLM(OK)
#     res = vc.answer_about_figure("Explain this architecture.", fig(images["diagram"]), llm=fake)

#     assert res["status"] == "ok"
#     call = fake.calls[0]
#     assert call["json_output"] and len(call["images"]) == 1
#     sent = call["images"][0]
#     assert sent["mime_type"] == "image/png"
#     assert sent["data"] == Path(images["diagram"]).read_bytes()  # the original, unmodified image
#     assert res["source"] == {"paper": "t01", "paper_id": "t01", "page": 3, "fig_id": "t01_p3_fig1",
#                              "caption": "Figure 1: TEST FIXTURE caption."}
#     assert res["image_path"] == images["diagram"]
#     assert res["figure_type"] == "architecture_diagram"
#     c = res["citations"][0]
#     assert (c["paper_id"], c["page"], c["fig_ref"], c["source_label"]) == ("t01", 3, "t01_p3_fig1", "F1")
#     assert "<source>" in call["prompt"] and "caption: Figure 1" in call["prompt"]
#     assert "figure id: t01_p3_fig1" in call["prompt"]
#     assert not any("Missing figure metadata" in w for w in res["warnings"])


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_chart_jpeg_is_sent_as_jpeg(images):
#     fake = FakeLLM({**OK, "figure_type": "chart"})
#     res = vc.answer_about_figure("What does this chart show?",
#                                  fig(images["chart"], paper_id="t02", fig_id="t02_p5_fig1", page=5), llm=fake)
#     assert res["status"] == "ok"
#     assert fake.calls[0]["images"][0]["mime_type"] == "image/jpeg"
#     assert res["source"]["fig_id"] == "t02_p5_fig1" and res["figure_type"] == "chart"


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_table_with_unclear_values_is_reported(images):
#     fake = FakeLLM({"answer": "Two rows; second score unreadable.", "figure_type": "table",
#                     "readability": "partially_readable", "unclear_elements": ["row B score"],
#                     "insufficient_evidence": False})
#     res = vc.answer_about_figure("Compare the values.", fig(images["table"]), llm=fake)
#     assert res["readability"] == "partially_readable"
#     assert res["unclear_elements"] == ["row B score"]


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_unreadable_and_low_resolution_images_are_flagged(images):
#     fake = FakeLLM({**OK, "readability": "unreadable"})
#     res = vc.answer_about_figure("What is shown?", fig(images["tiny"]), llm=fake)
#     assert any("Low-resolution" in w for w in res["warnings"])
#     assert any("unreadable" in w for w in res["warnings"])
#     assert "low resolution" in fake.calls[0]["prompt"]


@pytest.mark.parametrize("key,message", [("missing", "not found"), (None, "No image")])
def test_missing_image_never_falls_back_to_caption(images, key, message):
    fake = FakeLLM(OK)
    res = vc.answer_about_figure("Explain.", fig(images[key] if key else ""), llm=fake)
    assert res["status"] == "error" and message in res["error"]
    assert fake.calls == []
    assert res["source"]["fig_id"] == "t01_p3_fig1"  # metadata still returned for display


@pytest.mark.parametrize("key", ["text", "corrupt"])
def test_invalid_or_corrupt_image_is_rejected(images, key):
    fake = FakeLLM(OK)
    res = vc.answer_about_figure("Explain.", fig(images[key]), llm=fake)
    assert res["status"] == "error" and "could not be read" in res["error"]
    assert fake.calls == []


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_unsupported_format_is_converted_and_huge_image_downscaled(images):
#     fake = FakeLLM(OK)
#     res = vc.answer_about_figure("Explain.", fig(images["bmp"]), llm=fake)
#     sent = fake.calls[0]["images"][0]
#     assert sent["mime_type"] == "image/png"
#     assert Image.open(io.BytesIO(sent["data"])).format == "PNG"
#     assert any("Converted BMP" in w for w in res["warnings"])

#     fake = FakeLLM(OK)
#     res = vc.answer_about_figure("Explain.", fig(images["huge"]), llm=fake)
#     assert max(Image.open(io.BytesIO(fake.calls[0]["images"][0]["data"])).size) == vc.MAX_SIDE_PX
#     assert any("Downscaled" in w for w in res["warnings"])


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# @pytest.mark.parametrize("fake,message", [
#     (FakeLLM(error=LLMError("Model 'x' is not available")), "not available"),
#     (FakeLLM("not json"), "malformed"),
#     (FakeLLM({"answer": ""}), "malformed"),
# ])
# def test_provider_failure_and_malformed_output(images, fake, message):
#     res = vc.answer_about_figure("Explain.", fig(images["diagram"]), llm=fake)
#     assert res["status"] == "error" and message in res["error"]


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_upload_without_metadata_still_works_and_warns(images):
#     fake = FakeLLM(OK)
#     res = vc.answer_about_figure("Explain.", {"image_bytes": Path(images["diagram"]).read_bytes()}, llm=fake)
#     assert res["status"] == "ok" and res["citations"] == []
#     assert res["source"] == {"paper": None, "paper_id": None, "page": None, "fig_id": None, "caption": None}
#     warning = next(w for w in res["warnings"] if "Missing figure metadata" in w)
#     assert all(x in warning for x in ("paper id", "figure id", "page", "caption"))
#     assert "page: unknown" in fake.calls[0]["prompt"]


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_insufficient_evidence_and_input_validation(images):
#     res = vc.answer_about_figure("What GPU?", fig(images["diagram"]),
#                                  llm=FakeLLM({**OK, "insufficient_evidence": True}))
#     assert res["status"] == "insufficient_evidence"
#     assert vc.answer_about_figure(" ", fig(images["diagram"]), llm=FakeLLM(OK))["status"] == "error"
#     assert vc.answer_about_figure("q", "not a dict", llm=FakeLLM(OK))["status"] == "error"


@pytest.mark.parametrize("stored", [
    r"C:\Users\someone-else\HCL-Hackathon\data\figures\p01_p3_fig1.png",   # index built on Windows
    "/home/someone-else/HCL-Hackathon/data/figures/p01_p3_fig1.png",       # index built on Linux/macOS
])
def test_stale_absolute_image_path_falls_back_to_local_figures_dir(tmp_path, monkeypatch, stored):
    # Member 1 stores absolute paths from the machine that ran extraction.
    (tmp_path / "p01_p3_fig1.png").write_bytes(b"x")
    monkeypatch.setattr(sources, "FIGURES_PATH", str(tmp_path))
    assert sources.normalize_figure({"image_path": stored})["image_path"] == str(tmp_path / "p01_p3_fig1.png")
    # If the file is not available locally either, the original path is kept (and reported missing later).
    missing = stored.replace("p01_p3_fig1", "absent")
    assert sources.normalize_figure({"image_path": missing})["image_path"] == missing


# --- answer_visual (integration layer) --------------------------------------------

# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_answer_visual_uses_figure_index(images, monkeypatch):
#     index_path = images["root"] / "figures_index.json"
#     index_path.write_text(json.dumps({"t01_p3_fig1": fig(images["diagram"])}))
#     monkeypatch.setattr(sources, "FIGURES_INDEX_PATH", str(index_path))

#     res = vc.answer_visual("Explain.", fig_id="t01_p3_fig1", llm=FakeLLM(OK))
#     assert res["status"] == "ok" and res["source"]["page"] == 3 and res["image_path"] == images["diagram"]

#     res = vc.answer_visual("Explain.", fig_id="missing_fig", llm=FakeLLM(OK))
#     assert res["status"] == "error" and "not in the figure index" in res["error"]

#     monkeypatch.setattr(sources, "FIGURES_INDEX_PATH", str(images["root"] / "absent.json"))
#     res = vc.answer_visual("Explain.", fig_id="t01_p3_fig1", llm=FakeLLM(OK))
#     assert res["status"] == "error" and "Run ingestion" in res["error"]


# [DISABLED: mock-based test — simulated Gemini response / synthetic data]
# def test_answer_visual_without_fig_id_uses_retrieval(images, monkeypatch):
#     # Member 2's retriever is still a stub; this must surface as a clear error.
#     res = vc.answer_visual("Explain the architecture.", llm=FakeLLM(OK))
#     assert res["status"] == "error" and "retriev" in res["error"].lower()

#     figure_chunk = {"id": "t01__figure__0", "document": "Figure 1: TEST FIXTURE caption. nearby",
#                     "metadata": {"paper_id": "t01", "title": "Test Paper A", "page": 3, "chunk_type": "figure",
#                                  "figure_ref": "t01_p3_fig1", "image_path": images["diagram"]}}
#     text_chunk = {"paper_id": "t01", "text": "body text", "chunk_type": "text", "chunk_index": 1}

#     monkeypatch.setattr(sources, "get_passages", lambda q, top_k=None: [text_chunk, figure_chunk])
#     res = vc.answer_visual("Explain the architecture.", llm=FakeLLM(OK))
#     assert res["status"] == "ok"
#     assert res["source"]["fig_id"] == "t01_p3_fig1" and res["source"]["paper"] == "Test Paper A"

#     monkeypatch.setattr(sources, "get_passages", lambda q, top_k=None: [text_chunk])
#     res = vc.answer_visual("Explain the architecture.", llm=FakeLLM(OK))
#     assert res["status"] == "insufficient_evidence"
