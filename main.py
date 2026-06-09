import os
from src.config import PDF_FILE_PATH
from src.ingestion.pdf_parser import parse_and_chunk_pdf
from src.ingestion.embedder import embed_texts
from src.ingestion.qdrant_indexer import setup_collections, upsert_traditional, upsert_hybrid
from src.evaluation.comparator import compare_results

def build_system():
    """Pipeline nạp dữ liệu"""
    # Kiểm tra xem PDF đã có chưa
    if not os.path.exists(PDF_FILE_PATH):
        print(f"❌ Không tìm thấy file PDF tại: {PDF_FILE_PATH}")
        print("Vui lòng đặt file PDF vào thư mục 'data/raw/' và cập nhật lại config.py")
        return False

    # 1. Parse & Chunk PDF
    chunks = parse_and_chunk_pdf(PDF_FILE_PATH)
    if not chunks:
        return False

    # 2. Embedding
    dense_vectors, sparse_vectors = embed_texts(chunks)

    # 3. Indexing vào Qdrant
    setup_collections()
    upsert_traditional(chunks, dense_vectors)
    upsert_hybrid(chunks, dense_vectors, sparse_vectors)
    
    print("\n✅ BUILD SYSTEM HOÀN TẤT!\n")
    return True

def run_evaluation():
    """Chạy test so sánh"""
    # Các câu hỏi test: Bao gồm cả câu hỏi ngữ nghĩa và chứa từ khóa chính xác (mã số, tên riêng)
    test_queries = [
        "OKRs", # Ngữ nghĩa (Cả 2 đều tìm được)
        "HAMPAPUR RANGADORE BINOD",
        "Công ty FPT Software có mã chứng khoán là gì?", # Từ khóa chuyên ngành (Traditional dễ miss)
        "Ai là người phụ trách mảng Bán lẻ của FPT?", # Tên riêng
        "ĐHĐCĐ 2023", # Tên riêng
        "Đại hội đồng cổ đông 2023", # Tên riêng
        "ESG là gì? ",
        "ESG"
    ]

    for query in test_queries:
        compare_results(query)

if __name__ == "__main__":
    if build_system():
        run_evaluation()
    # run_evaluation()