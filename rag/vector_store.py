"""
Vector Store module
-------------------
Store all the chunks vector into a vector database
2 functions build the vector and them load the vector DB
"""
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
PERSIST_DIR = "vector_db"
COLLECTION_NAME = "handbook"

def build_vector_store(chunks: list[Document], embedding_model: Embeddings,) -> Chroma:
    if not Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = PERSIST_DIR,
        collection_name = COLLECTION_NAME
    )
    print("Saved the vector store")
    return vector_store

def load_vector_store(embedding_model: Embeddings) -> Chroma:
    if not Path(PERSIST_DIR).exists():
        raise FileNotFoundError("Vector not found")
    return Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embedding_model,
        persist_directory = PERSIST_DIR
    )
if __name__ == "__main__":
    from rag.chunker import split_documents
    from rag.embedding import get_embeddings
    from rag.loader import load_documents
    documents = load_documents("Data/Handbook")
    chunks = split_documents(documents)
    embedding_model = get_embeddings()
    build_vector_store(chunks, embedding_model)

    vector_store = load_vector_store(embedding_model)
    print(f"Chunks embedded in DB : {len(vector_store.get()['ids'])}")