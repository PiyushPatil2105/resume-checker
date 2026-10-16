import io
import re

def extract_text_from_pdf(pdf_file_or_bytes) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz) if available,
    falling back to PyPDF2. Handles both file-like objects and byte streams.
    """
    text = ""
    # Try PyMuPDF (fitz) first - superior at handling multi-column resumes and formatting
    try:
        import fitz  # PyMuPDF
        if hasattr(pdf_file_or_bytes, "read"):
            data = pdf_file_or_bytes.read()
            # Reset pointer if possible
            if hasattr(pdf_file_or_bytes, "seek"):
                pdf_file_or_bytes.seek(0)
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open(stream=pdf_file_or_bytes, filetype="pdf")
            
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        if text.strip():
            return clean_extracted_text(text)
    except Exception as e:
        print(f"PyMuPDF extraction note: {e}, falling back to PyPDF2")

    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader
        if hasattr(pdf_file_or_bytes, "seek"):
            pdf_file_or_bytes.seek(0)
        reader = PdfReader(pdf_file_or_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return clean_extracted_text(text)
    except Exception as e:
        print(f"Error in PyPDF2 extraction: {e}")
        return ""

def clean_extracted_text(raw_text: str) -> str:
    """
    Cleans raw extracted PDF text by removing excessive whitespace,
    non-printable symbols, and unifying line endings.
    """
    if not raw_text:
        return ""
    # Normalize unicode spaces & special chars
    cleaned = raw_text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    # Replace multiple consecutive spaces or tabs with a single space
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    # Replace 3 or more consecutive newlines with 2
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
