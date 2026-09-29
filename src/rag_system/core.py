import os 
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", 'https://openrouter.ai/api/v1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

def extract_text_from_file(file_path:str)->str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt" or ext == ".md":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == '.pdf':
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return '\n'.join(text_parts)

    else:
        raise ValueError(f'Формат {ext} не поддерживается')

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks

def get_text_embedding(text: str) -> list[float]:
    try:
        response = client.embeddings.create(
            model = 'nvidia/nemotron-3-embed-1b:free',
            input = text
        )
        return response.data[0].embedding

    except Exception as e:
        raise RuntimeError(f'Произошла ошибка во время генерации эмбеддинга: {e}')

RAG_SYSTEM_PROMPT = """
You are a AI assistant and an expert in analyzing internal documents.
Your task is to answer user questions EXCLUSIVELY based on the provided context.

STRICT RULES:
1. Answer in a professional and clear tone.
2. Use ONLY facts from the <context> section. It is strictly forbidden to use your internal knowledge, guess facts, or hallucinate.
3. If there is no answer to the question in the <context>, notify the user that there is no answer in provided documents.
4. If the <context> is empty, do not try to guess the answer.
5. Structure your answer (use bullet points or paragraphs) if it makes the information clearer.
"""

def generate_rag_answer(user_question:str, retrieved_contexts: list[str]) -> str:
    if not retrieved_contexts:
        return "There is no answer in provided documents"
    
    formatted_contexts = []
    for i, context in enumerate(retrieved_contexts, 1):
        formatted_contexts.append(f"[Document {i}]\n {context.strip()}")
    
    context_block = "\n\n---\n\n".join(formatted_contexts)

    user_prompt = f"""
    <context>
    {context_block}
    </context>

    <question>
    {user_question}
    </question>
    """

    messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
    
    try:
        response = client.chat.completions.create(
            model = 'qwen/qwen3.8-27b:free',
            messages=messages,
            temperature=0.0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f'Произошла ошибка во время генерации ответа: {e}')


