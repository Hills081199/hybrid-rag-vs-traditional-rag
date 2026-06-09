from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query: str, context_points, method_name: str = "RAG") -> str:
    # Trích text từ payload
    context = "\n\n".join([point.payload['text'] for point in context_points])
    
    prompt = f"""
    Bạn là một trợ lý AI chuyên nghiệp. Dựa CHẾT vào thông tin Context dưới đây, hãy trả lời câu hỏi của người dùng.
    Nếu thông tin không có trong Context, hãy nói "Tôi không tìm thấy thông tin trong tài liệu".
    
    Context:
    {context}
    
    Câu hỏi của người dùng: {query}
    Trả lời:
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý phân tích tài liệu chính xác."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi gọi LLM: {e}"