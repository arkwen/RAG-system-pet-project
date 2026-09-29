import os 
import chromadb

class VectorDBManager:
    def __init__(self, data_directory = 'chroma_data'):
        os.makedirs(data_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=data_directory)
        self.collection = self.client.get_or_create_collection(
            name = 'documents_rag_collection', 
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: list[str], embeddings: list[list[float]], doc_name:str):
        if not chunks or embeddings:
            return

        ids = [f'{doc_name}_chunk_{i}' for i in range(len(chunks))]
        metadatas = [{'source_file_name': doc_name} for _ in chunks]

        try:
            self.collection.delete(where={"source": doc_name})
        except Exception:
            pass

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
    
    def search_similar_context(self, query_embedding: list[float], k: int = 5) -> list[str]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []


