import fitz  # PyMuPDF
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def parse_and_chunk_pdf(pdf_path: str) -> list[str]:
    """
    Đọc file PDF và chia nhỏ thành các đoạn text (chunks) với kỹ thuật overlap.
    """
    print(f"Đang đọc PDF từ: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Lỗi đọc PDF: {e}")
        return []

    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += f"\n[Page {page_num + 1}] " + text

    # Làm sạch text cơ bản
    full_text = full_text.replace("\n", " ").replace("  ", " ").strip()
    
    if not full_text:
        return []

    # Chunking with overlap
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunks.append(full_text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"Đã chia PDF thành {len(chunks)} chunks.")
    return chunks