"""
keyword retriever 
------------------
find the chunks based on the key words 

"""
import re
from langchain_core.callbacks import CallbackManagerForRetrieverRun, get_callback_manager
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """Tokenize the text into words."""
    return re.findall(r"\w+", text.lower())

class KeywordRetriever(BaseRetriever):
    chunks: list[Document]
    k: int = 4
    bm25: BM25Okapi | None = None

    def model_post_init(self, __context) -> None:
        self.bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in self.chunks])


    def _get_relevant_documents(
    self,
    query: str,
    *,
    run_manager: CallbackManagerForRetrieverRun) -> list[Document]:

        if self.bm25 is None:
            raise ValueError("BM25 index is not initialized.")

        scores = self.bm25.get_scores(tokenize(query))

        best = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[: self.k]

        return [self.chunks[i] for i in best if scores[i] > 0]