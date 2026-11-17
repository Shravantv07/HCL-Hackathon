"""Tests for input validation and prompt-injection defence."""
import pytest
from src.utils.security import validate_upload, wrap_source_content


def test_validate_upload_rejects_non_pdf(tmp_path):
    f = tmp_path / "malicious.exe"
    f.write_bytes(b"MZ")
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_upload(str(f))


def test_validate_upload_rejects_oversized_file(tmp_path):
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4" + b"0" * (51 * 1024 * 1024))
    with pytest.raises(ValueError, match="too large"):
        validate_upload(str(f))


def test_validate_upload_accepts_valid_pdf(tmp_path):
    f = tmp_path / "paper.pdf"
    f.write_bytes(b"%PDF-1.4 small content")
    validate_upload(str(f))  # should not raise


def test_wrap_source_content():
    wrapped = wrap_source_content("some paper text")
    assert wrapped.startswith("<source>")
    assert wrapped.endswith("</source>")
    assert "some paper text" in wrapped
