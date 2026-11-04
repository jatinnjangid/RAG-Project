"""
Reciprocal Rank Fusion
----------------------
Merge several rank lists into one.
"""

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    result_list: list[list[Document]],
    k: int = 60
) -> list[Document]:

    chunks = {}
    scores = {}

    for results in result_list:
        for rank, chunk in enumerate(results, start=1):
            key = chunk.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            chunks[key] = chunk

    ranked_chunks = sorted(scores, key = scores.get, reverse=True)
    return [chunks[key] for key in ranked_chunks]

if __name__ == "__main__":
    A = Document(page_content="A")
    B = Document(page_content="B")
    C = Document(page_content="C")
    d = Document(page_content="D")
