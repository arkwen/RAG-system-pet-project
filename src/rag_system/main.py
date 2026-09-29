import os
import shutil
from .core import extract_text_from_file, chunk_text, get_text_embedding, generate_rag_answer
from .database import VectorDBManager
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
import asyncio
import aiofiles

app = FastAPI(title = 'RAG API')
db_manager = VectorDBManager()

UPLOAD_DIR = 'data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AskRequest(BaseModel):
    question:str

@app.post('/upload')
async def upload_document(file: UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Недопустимый формат файла. Разрешены только файлы .txt, .md и .pdf")
    file_name = Path(file.filename).name
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    try:
        file_content = await file.read()
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении файла: {e}")

    try:
        raw_text = await asyncio.to_thread(extract_text_from_file, file_path)
        chunks = await asyncio.to_thread(chunk_text, raw_text)

        if not chunks:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Файл не содержит текста")
        
        embedding_tasks = [asyncio.to_thread(get_embedding, chunk) for chunk in chunks]
        embeddings = await asyncio.gather(*embedding_tasks)

        await asyncio.to_thread(
            db_manager.add_documents,
            chunks=chunks,
            embeddings=embeddings,
            doc_name=file_name
        )

        return {
            "status": "success",
            "message": f"Документ '{safe_filename}' успешно обработан.",
            "chunks_count": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

