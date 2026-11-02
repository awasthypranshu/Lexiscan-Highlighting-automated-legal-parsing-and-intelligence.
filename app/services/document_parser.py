#Read the content of file that we created and return that
import os
import PyPDF2

def extract_text_from_txt(file_path: str) -> dict:
    if os.path.getsize(file_path) == 0:
        raise ValueError("File is empty")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {"text": text.strip(), "page_count": str(len(text.splitlines())), "word_count": len(text.split())}
    except Exception as e:
        raise ValueError(f"Failed to read text file: {str(e)}")

def extract_text_from_pdf(file_path: str) -> dict:
    if os.path.getsize(file_path) == 0:
        raise ValueError("PDF file is empty")
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return {"text": text.strip(), "page_count": str(len(pdf_reader.pages)), "word_count": len(text.split())}
    except PyPDF2.errors.PyPDFError as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {str(e)}")

def extract_text(file_path: str) -> dict:
    #extension check
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type")