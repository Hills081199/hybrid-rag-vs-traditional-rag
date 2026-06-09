from qdrant_client import QdrantClient, models
from src.config import QDRANT_HOST, QDRANT_PORT, HYBRID_COLLECTION
from src.ingestion.embedder import embed_query

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def hybrid_search(query: str, limit: int = 10):
    """Tìm kiếm Hybrid (Dense + Sparse) fusion bằng RRF"""
    query_dense, query_sparse = embed_query(query)
    
    results = client.query_points(
        collection_name=HYBRID_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=5, 
            ),
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=5, 
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF), 
        limit=limit,
        with_payload=True,
    )
    return results.points