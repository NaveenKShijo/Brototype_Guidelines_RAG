# connects to chroma vector database

from src.config import CHROMA_COLLECTION_NAME
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

def get_vector_store():
    client = chromadb.Client()
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(
        chroma_collection = collection
    )
    storage_context = StorageContext.from_defaults(
        vector_store = vector_store
    )
    return storage_context