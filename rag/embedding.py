"""
Embeddings module
-----------------
Load the embedding model to convert text chunks
into vector embeddings containing semantic meaning.
"""
from langchain_huggingface import HuggingFaceEmbeddings
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

if __name__ == "__main__":
    import numpy as np
    def cosine_similarity(a : list[float], b : list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    model = get_embeddings()
    vectore = model.embed_query("how long is the cooldown period?")
    print(f"first 5 values : {vectore[:5]}")
    print(f"vector length : {len(vectore)}")
    question = "how long is the cooldown period?"

    sentences = [
        "In between each cycle, we spent to weeks coling down",
        "Employees wil recieve early allowance",
        "This recepie need two cups of flour"    ]