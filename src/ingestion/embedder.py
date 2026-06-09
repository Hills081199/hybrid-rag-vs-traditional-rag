import os
from fastembed import TextEmbedding, SparseTextEmbedding
from src.config import DENSE_MODEL_NAME, SPARSE_MODEL_NAME

# Khởi tạo model 1 lần (Singleton pattern cơ bản)
_dense_model = None
_sparse_model = None

# Tạo đường dẫn cache ngay trong thư mục dự án để tránh lỗi Permission của Windows
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_dense_model():
    global _dense_model
    if _dense_model is None:
        print(f"Đang load Dense model: {DENSE_MODEL_NAME}...")
        _dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME, cache_dir=CACHE_DIR)
    return _dense_model

def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        print(f"Đang load Sparse model: {SPARSE_MODEL_NAME}...")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, cache_dir=CACHE_DIR)
    return _sparse_model

def embed_texts(chunks: list[str]):
    """Trả về danh sách Dense vectors và Sparse vectors"""
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    
    print("Đang generate Dense vectors...")
    dense_vectors = list(dense_model.embed(chunks))
    
    print("Đang generate Sparse vectors (BM25)...")
    sparse_vectors = list(sparse_model.embed(chunks))
    
    return dense_vectors, sparse_vectors

def embed_query(query: str):
    """Embedding cho 1 câu hỏi"""
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    
    query_dense = list(dense_model.embed([query]))[0].tolist()
    query_sparse = list(sparse_model.embed([query]))[0].as_object()
    
    return query_dense, query_sparse