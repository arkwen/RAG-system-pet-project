import os 
import chromadb
from chromadg.congif import Settings

class VectorDBManager:
    def __init__(self, data_directory = 'chroma_data'):
        os.makedirs(data_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=data_directory)
        self.collection = self.client.get_or_create_collection(
            name = 'documents_rag_collection', 
            metadata={"hnsw:space": "cosine"}
        )

