from qdrant_client import QdrantClient
from src.config import QDRANT_HOST, QDRANT_PORT, TRADITIONAL_COLLECTION
from src.ingestion.embedder import embed_query

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def traditional_search(query: str, limit: int = 10):
    """Tìm kiếm chỉ dùng Dense Vector"""
    query_dense, _ = embed_query(query) # Bỏ qua sparse
    
    results = client.search(
        collection_name=TRADITIONAL_COLLECTION,
        query_vector=query_dense,
        limit=limit,
        with_payload=True
    )
    return results