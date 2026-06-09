from qdrant_client import QdrantClient, models
from src.config import QDRANT_HOST, QDRANT_PORT, TRADITIONAL_COLLECTION, HYBRID_COLLECTION, DENSE_VECTOR_SIZE

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def setup_collections():
    """Tạo (hoặc tạo lại) 2 collection: 1 cho Traditional, 1 cho Hybrid"""
    
    # 1. Traditional Collection (Chỉ Dense)
    # Dùng recreate_collection để tự động xóa collection cũ nếu kích thước vector thay đổi
    client.recreate_collection(
        collection_name=TRADITIONAL_COLLECTION,
        vectors_config=models.VectorParams(
            size=DENSE_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        )
    )
    print(f"Đã tạo collection: {TRADITIONAL_COLLECTION}")
    
    # 2. Hybrid Collection (Dense + Sparse)
    client.recreate_collection(
        collection_name=HYBRID_COLLECTION,
        vectors_config={
            # Khai báo rõ tên của Dense Vector là "dense"
            "dense": models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            # Khai báo rõ tên của Sparse Vector là "sparse"
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )
    print(f"Đã tạo collection: {HYBRID_COLLECTION}")

def upsert_traditional(chunks: list[str], dense_vectors):
    points = []
    for idx, (text, vector) in enumerate(zip(chunks, dense_vectors)):
        points.append(
            models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={"text": text}
            )
        )
    client.upsert(collection_name=TRADITIONAL_COLLECTION, points=points)
    print(f"Đã upsert {len(points)} points vào {TRADITIONAL_COLLECTION}")

def upsert_hybrid(chunks: list[str], dense_vectors, sparse_vectors):
    points = []
    for idx, (text, d_vec, s_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "dense": d_vec.tolist(),
                    "sparse": s_vec.as_object(),
                },
                payload={"text": text}
            )
        )
    client.upsert(collection_name=HYBRID_COLLECTION, points=points)
    print(f"Đã upsert {len(points)} points vào {HYBRID_COLLECTION}")