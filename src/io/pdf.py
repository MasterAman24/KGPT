import io
from typing import List
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from .ocr import ocr_image_to_text

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts: List[str] = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    except Exception:
        pass

    if text_parts:
        return "\n\n".join(text_parts).strip()

    images = convert_from_bytes(file_bytes, fmt="png")
    ocr_texts = [ocr_image_to_text(img) for img in images]
    return "\n\n".join(ocr_texts).strip()
