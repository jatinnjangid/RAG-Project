"""
Split the document into small chunks using the Recursive Character splitter
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} documents got converted into {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    from rag.loader import load_documents

    documents = load_documents("Data/Handbook")
    chunks = split_documents(documents)
    print(chunks[1].metadata)
    print(chunks[1].page_content)