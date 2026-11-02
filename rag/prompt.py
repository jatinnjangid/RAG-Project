"""
prompt module
- - - - - -
this is a prompt template where the instructions for the chatbots to follow is given.

"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """ You are a helpful assistant that answers employee queries using the company handbook.
RULES:
- Answer ONLY using the context provided. Do not use outside knowledge.
- If the answer is not in the context, reply exactly: "I couldn't find that in the handbook."
- Keep answers short, clear, and in full sentences.
- Answer in the same language as the question.
"""

USER_PROMPT = """Context: {context}
Question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT)
])

def format_context(chunks: list) -> str:
    return "\n".join(
        f"[Source: {chunk.metadata['source']}]\n{chunk.page_content}"
        for chunk in chunks
    )