from src.retrieval.traditional_rag import traditional_search
from src.retrieval.hybrid_rag import hybrid_search
from src.generation.llm_generator import generate_answer

def compare_results(query: str):
    print(f"\n{'='*80}")
    print(f"❓ CÂU HỎI: {query}")
    print(f"{'='*80}")
    
    # 1. Traditional Path
    trad_context = traditional_search(query)
    trad_answer = generate_answer(query, trad_context, "Traditional RAG")
    
    # 2. Hybrid Path
    hybrid_context = hybrid_search(query)
    hybrid_answer = generate_answer(query, hybrid_context, "Hybrid RAG")
    
    # 3. Display Results
    print("\n🔵 [TRADITIONAL RAG - Dense Only]")
    print(f"   Top 1 Context: {trad_context[0].payload['text'][:150]}...")
    print(f"   🗣️ Đáp án: {trad_answer}")
    
    print("\n🟢 [HYBRID RAG - Dense + BM25 + RRF]")
    print(f"   Top 1 Context: {hybrid_context[0].payload['text'][:150]}...")
    print(f"   🗣️ Đáp án: {hybrid_answer}")
    print(f"{'='*80}\n")