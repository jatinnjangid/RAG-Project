"""
Ingest the doc, creates the vector database
"""

from rag.chunker import split_documents
from rag.embedding import get_embeddings
from rag.loader import load_documents
from rag.vector_store import build_vector_store


def main():
    documents = load_documents("Data/Handbook")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
    embedding_model = get_embeddings()
    build_vector_store(chunks, embedding_model)

if __name__ == "__main__":
    main()
    