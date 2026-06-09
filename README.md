# Comparing Traditional RAG and Hybrid RAG

This project implements and compares two Retrieval-Augmented Generation (RAG) methods: **Traditional RAG** (using only Semantic Search) and **Hybrid RAG** (combining Semantic Search and Keyword Search).

## 1. Theory: Traditional RAG vs Hybrid RAG

### Traditional RAG (Semantic Search)
Traditional RAG relies primarily on semantic search using **Dense Vectors**.
- **How it works:** Text data is passed through an Embedding Model to generate dense vectors containing semantic information. When a user asks a question, the query is also converted into a similar vector. The system calculates the distance (e.g., Cosine Similarity) between the query vector and document vectors to find the most semantically relevant documents.
- **Pros:** Understands context and synonymy. Even if the query does not contain exact keywords found in the document, the system can still find information based on meaning.
- **Cons:** Often ineffective when searching for exact keywords, ID numbers, proper nouns, or specialized acronyms. It might return "seemingly relevant" texts that lack the precise information the user needs.

### Hybrid RAG
Hybrid RAG combines the strengths of both semantic search (Dense Vectors) and keyword search (Sparse Vectors - e.g., BM25).
- **How it works:** 
  - **Ingestion process:** The system embeds text into both Dense Vectors (to capture semantics) and Sparse Vectors (to count keyword frequency, focusing on specific vocabulary).
  - **Retrieval process:** When a query is made, the system performs both semantic search and keyword search simultaneously on the vector database.
  - **Fusion:** Results from both methods are then combined using algorithms like **Reciprocal Rank Fusion (RRF)**. RRF helps re-rank and produce a final list of results that balances both factors.
- **Pros:** Completely overcomes the limitations of Traditional RAG. It understands complex semantics while accurately retrieving ID numbers, proper nouns, and acronyms (e.g., "AGM 2023", stock tickers, etc.).

## 2. Workflow

### Ingestion Phase
Both methods begin by reading a PDF file and breaking it down into chunks.

- **Traditional RAG:** Generates a Dense Vector for each chunk and saves it into Qdrant (`traditional_rag` collection).
  
  ![Ingestion Traditional RAG](./ingestion_phase_traditional_rag.PNG)

- **Hybrid RAG:** Generates BOTH Dense Vectors and Sparse Vectors (BM25) for each chunk, then saves both into Qdrant (`hybrid_rag` collection).
  
  ![Ingestion Hybrid RAG](./ingestion_phase_hybrid_rag.PNG)

### Retrieval Phase

- **Traditional RAG:** Embeds the query into a Dense Vector and searches for the most similar vectors via Cosine/Dot Product distance.
  
  ![Retrieval Traditional RAG](./traditional_retrieval_flow.PNG)

- **Hybrid RAG:** Embeds the query into BOTH Dense and Sparse Vectors. Uses Qdrant's `FusionQuery` feature combined with the `RRF` algorithm to retrieve and rank the combined results from both search branches.
  
  ![Retrieval Hybrid RAG](./Hybrid_retrieval_flow.PNG)

## 3. Project Structure

The project is clearly organized according to the stages of a RAG pipeline:

![Folder Structure](./FolderStructure.PNG)

- `src/config.py`: General configuration file for the project (paths, model names, DB config).
- `src/ingestion/`: Data preprocessing logic.
  - `pdf_parser.py`: Reads the PDF file and splits it into small chunks.
  - `embedder.py`: Loads `fastembed` models (Dense: `bge-small-zh-v1.5`, Sparse: `bm25`) to perform text embedding.
  - `qdrant_indexer.py`: Initializes collections and upserts vectors into the Qdrant database.
- `src/retrieval/`: Contains search logic for both `traditional_rag.py` and `hybrid_rag.py`.
- `src/evaluation/`: Logic to compare the returned results of the 2 methods with various test queries.
- `data/raw/`: Directory storing input PDF documents.

## 4. How to Run the Project

### System Requirements
- Python 3.9+
- Docker (used to run Qdrant Vector Database)

### Step 1: Install Dependencies
Open a terminal at the root of the project and run:
```bash
pip install -r requirements.txt
```

### Step 2: Start Qdrant
The project uses Qdrant as the vector database to store both Dense and Sparse vectors. Start Qdrant via Docker using the following command:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### Step 3: Prepare Data and Environment Variables
1. Create a PDF file you want to test, rename it to `sample_document.pdf`, and place it in the `data/raw/` directory. (If you want to use a different name, update the `PDF_FILE_PATH` variable in `src/config.py`).
2. Create a file named `.env` in the root directory of the project and enter your OpenAI API Key:
```env
OPENAI_API_KEY=your-openai-api-key-here
```
*(This API Key is used by the project for LLMs if there is an automated answer generation/synthesis step).*

### Step 4: Run the Project and View the Comparison
Execute the following command:
```bash
python main.py
```

**Execution Flow:**
1. Checks if the PDF document exists, then parses and chunks it.
2. Automatically downloads the Embedding models (`BAAI/bge-small-zh-v1.5` for semantics and `Qdrant/bm25` for keywords) into an internal `models_cache/` directory.
3. Embeds the data and indexes it into 2 Qdrant collections.
4. Runs a built-in set of test queries. You will clearly see that for queries containing **proper nouns or specialized codes**, Hybrid RAG will retrieve information much better than Traditional RAG.