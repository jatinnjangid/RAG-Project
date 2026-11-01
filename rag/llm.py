"""
LLM Module
- - - - - - 
used to answer the question using the retrieved context and prompt
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
def get_llm(model: str = "openai/gpt-oss-120b", temperature: float = 0.1, max_tokens: int = 500) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not found")
    return ChatGroq(model = model, temperature=temperature, max_tokens = max_tokens )

if __name__ == "__main__":
    llm = get_llm()
    question = "What is RAG ans in one line."
    response = llm.invoke(question)

    print(response.content)
    