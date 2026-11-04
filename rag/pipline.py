"""
RAG Pipeline
--------------
Question -> retrieve chunks -> build the prompt -> LLM -> ans + src
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from rag.prompt import RAG_PROMPT, format_context

class RAGPipeline:

    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.chain = RAG_PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        #step 1: Retriever
        chunks = self.retriever.invoke(question)

        # step 2: Augment + Generate

        answer = self.chain.invoke({
            "context": format_context(chunks),
            "question": question,
        })
        sources = [
            {"source": chunk.metadata["source"], "content": chunk.page_content}
            for chunk in chunks
        ]
        return {"answer": answer, "sources": sources}