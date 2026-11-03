"""test_document_parser.py – Unit tests for app/services/document_parser.py.

Run with:
    cd "API Contract"
    python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import pytest

# Make sure the project root is on sys.path so app.* imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.document_parser import (
    extract_text,
    extract_text_from_txt,
    extract_text_from_pdf,
)


# ─────────────────────────────────────────────────────────────────────────────
# TXT parser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTxtParser:
    def _write_tmp(self, content: str, encoding="utf-8") -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        return path

    def test_returns_required_keys(self):
        path = self._write_tmp("Hello World. This is a test contract.")
        try:
            result = extract_text_from_txt(path)
        finally:
            os.remove(path)
        assert "text" in result
        assert "pages" in result
        assert "page_count" in result
        assert "word_count" in result

    def test_text_matches_file_content(self):
        content = "This is a simple contract with two sentences. Second sentence."
        path = self._write_tmp(content)
        try:
            result = extract_text_from_txt(path)
        finally:
            os.remove(path)
        assert result["text"] == content

    def test_page_count_is_one(self):
        path = self._write_tmp("Any text content.")
        try:
            result = extract_text_from_txt(path)
        finally:
            os.remove(path)
        assert result["page_count"] == 1

    def test_pages_list_has_one_entry(self):
        path = self._write_tmp("Any text content.")
        try:
            result = extract_text_from_txt(path)
        finally:
            os.remove(path)
        assert isinstance(result["pages"], list)
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page"] == 1
        assert result["pages"][0]["text"] == "Any text content."

    def test_word_count_is_correct(self):
        path = self._write_tmp("one two three four five")
        try:
            result = extract_text_from_txt(path)
        finally:
            os.remove(path)
        assert result["word_count"] == 5

    def test_empty_file_raises(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with pytest.raises(ValueError, match="empty"):
                extract_text_from_txt(path)
        finally:
            os.remove(path)

    def test_extract_text_dispatcher_routes_txt(self):
        path = self._write_tmp("Dispatcher test.")
        try:
            result = extract_text(path)
        finally:
            os.remove(path)
        assert result["page_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# PDF parser tests  (uses a minimal but real PDF created with PyPDF2)
# ─────────────────────────────────────────────────────────────────────────────

def _create_minimal_pdf(pages: list[str]) -> str:
    """Create a real (tiny) PDF with one text-layer page per entry in pages."""
    import PyPDF2
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        # PyPDF2 PdfWriter doesn't provide a simple "add text" method,
        # so we embed text using a minimal content stream.
        content_stream = (
            "BT /F1 12 Tf 72 720 Td (" + text.replace("(", r"\(").replace(")", r"\)") + ") Tj ET"
        ).encode()
        from PyPDF2.generic import (
            ArrayObject,
            ContentStream,
            DecodedStreamObject,
            NameObject,
        )
        stream_obj = DecodedStreamObject()
        stream_obj._data = content_stream
        page[NameObject("/Contents")] = writer._add_object(stream_obj)
        # Add a minimal font so the PDF is technically valid.
        from PyPDF2.generic import DictionaryObject, IndirectObject
        font_dict = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font_dict)}
                )
            }
        )
        page[NameObject("/Resources")] = resources

    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        writer.write(f)
    return path


class TestPdfParser:
    def test_returns_required_keys(self):
        """PDF parser returns text, pages, page_count, word_count."""
        # Use an existing PDF from the uploads directory for a real test.
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        pdfs = [
            f for f in os.listdir(upload_dir)
            if f.endswith(".pdf") and f != ".gitkeep"
        ] if os.path.isdir(upload_dir) else []

        if not pdfs:
            pytest.skip("No PDFs in uploads/ to test against – upload a contract first.")

        pdf_path = os.path.join(upload_dir, pdfs[0])
        result = extract_text_from_pdf(pdf_path)
        assert "text" in result
        assert "pages" in result
        assert "page_count" in result
        assert "word_count" in result

    def test_pages_list_length_matches_page_count(self):
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        pdfs = [
            f for f in os.listdir(upload_dir)
            if f.endswith(".pdf") and f != ".gitkeep"
        ] if os.path.isdir(upload_dir) else []
        if not pdfs:
            pytest.skip("No PDFs in uploads/.")
        pdf_path = os.path.join(upload_dir, pdfs[0])
        result = extract_text_from_pdf(pdf_path)
        assert len(result["pages"]) == result["page_count"]

    def test_pages_have_1based_numbers(self):
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        pdfs = [
            f for f in os.listdir(upload_dir)
            if f.endswith(".pdf") and f != ".gitkeep"
        ] if os.path.isdir(upload_dir) else []
        if not pdfs:
            pytest.skip("No PDFs in uploads/.")
        pdf_path = os.path.join(upload_dir, pdfs[0])
        result = extract_text_from_pdf(pdf_path)
        page_numbers = [p["page"] for p in result["pages"]]
        assert page_numbers[0] == 1
        assert page_numbers == list(range(1, len(page_numbers) + 1))

    def test_empty_pdf_raises(self):
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            with pytest.raises(ValueError):
                extract_text_from_pdf(path)
        finally:
            os.remove(path)

    def test_extract_text_dispatcher_routes_pdf(self):
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        pdfs = [
            f for f in os.listdir(upload_dir)
            if f.endswith(".pdf") and f != ".gitkeep"
        ] if os.path.isdir(upload_dir) else []
        if not pdfs:
            pytest.skip("No PDFs in uploads/.")
        pdf_path = os.path.join(upload_dir, pdfs[0])
        result = extract_text(pdf_path)
        assert "pages" in result

