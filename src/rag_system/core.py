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
    ext = os.path.splittext(file_path)[1].lower()

    if ext == ".txt":
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

def chunk_text(text: str, chunk_size = 500: int, chunk_overlap = 100: int) -> list[str]:
    
