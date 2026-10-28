"""
Loader Module : Read all the file in the data/handbook folder
and then convert it into documents. 
Document = text Content + metadata
"""
from pathlib import Path

from langchain_core.documents import Document

def load_documents(folder: str) -> list[Document]:
    folder_path = Path(folder)
    documents = []

    for file_path in (folder_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        doc = Document(page_content=text, metadata={"source": str(file_path.name)}) 
        documents.append(doc)
        print(f"loaded: {file_path.name} with ({len(text)} characters)")
    return documents

if __name__ == "__main__":
    documents = load_documents("data/handbook")
    print(f"\nTotal documents: {len(documents)}")

    first = documents[0]
    
    print(f"\nThis is the metadata {first.metadata}")
    print(f"\nThis is the page content {first.page_content[:300]}")
    
