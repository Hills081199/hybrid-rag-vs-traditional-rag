import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant Config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection Names
TRADITIONAL_COLLECTION = "traditional_rag"
HYBRID_COLLECTION = "hybrid_rag"

# FastEmbed Models
DENSE_MODEL_NAME = "BAAI/bge-small-zh-v1.5" # Hỗ trợ tốt tiếng Việt/Tiếng Trung
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_SIZE = 512 # Kích thước output của bge-small

# LLM Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-default-key-here")
LLM_MODEL = "gpt-4o-mini"

# Data Config
PDF_FILE_PATH = "data/raw/sample_document.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50