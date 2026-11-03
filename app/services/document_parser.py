# Read the content of a file and return structured text + page-level data.
import os
import logging
import PyPDF2

logger = logging.getLogger(__name__)


def extract_text_from_txt(file_path: str) -> dict:
    """Extract text from a plain-text file.

    Returns a dict with keys: text, pages, page_count, word_count.
    The entire file is treated as a single "page 1" to keep the interface
    consistent with the PDF parser.
    """
    if os.path.getsize(file_path) == 0:
        raise ValueError("File is empty")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        stripped = text.strip()
        return {
            "text": stripped,
            "pages": [{"page": 1, "text": stripped}],
            "page_count": 1,
            "word_count": len(stripped.split()),
        }
    except Exception as e:
        raise ValueError(f"Failed to read text file: {str(e)}")


def extract_text_from_pdf(file_path: str) -> dict:
    """Extract text from a PDF file, preserving page-level information.

    Returns a dict with keys: text, pages, page_count, word_count.

    - ``text``       – complete contract text (all pages joined with newlines).
    - ``pages``      – list of {"page": <1-based int>, "text": <str>} dicts,
                       one entry per PDF page even when a page has no text.
    - ``page_count`` – total number of PDF pages.
    - ``word_count`` – word count of the combined text.

    Raises ValueError for empty files, corrupt PDFs, or PDFs that appear to
    be scanned (no extractable text on any page).
    """
    if os.path.getsize(file_path) == 0:
        raise ValueError("PDF file is empty")
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            pages = []
            all_text_parts = []

            for idx, page_obj in enumerate(pdf_reader.pages):
                page_number = idx + 1  # 1-based
                try:
                    raw = page_obj.extract_text() or ""
                except Exception as page_err:
                    logger.warning(
                        "Could not extract text from page %d of %s: %s",
                        page_number,
                        file_path,
                        page_err,
                    )
                    raw = ""

                page_text = raw.strip()
                pages.append({"page": page_number, "text": page_text})
                if page_text:
                    all_text_parts.append(page_text)

        full_text = "\n\n".join(all_text_parts).strip()

        if not full_text and num_pages > 0:
            raise ValueError(
                "No extractable text found in this PDF. The file may be a scanned "
                "document that requires OCR. OCR support can be added in a future version."
            )

        return {
            "text": full_text,
            "pages": pages,
            "page_count": num_pages,
            "word_count": len(full_text.split()),
        }

    except PyPDF2.errors.PyPDFError as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    except ValueError:
        # Re-raise our own descriptive errors unchanged.
        raise
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {str(e)}")


def extract_text(file_path: str) -> dict:
    """Route to the correct extractor based on file extension.

    Always returns a dict with keys: text, pages, page_count, word_count.
    The ``pages`` list contains one entry per source page with 1-based page
    numbers and the extracted text for that page.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
