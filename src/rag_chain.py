"""
Basic RAG chain over the vector store: retrieve -> format -> prompt -> model
-> parse. This is the Day-1 milestone chain (single vector retriever).
src/retrieval.py (Day 2) swaps in the hybrid BM25+vector retriever without
changing this chain's shape.

Run directly for a quick CLI smoke test: python -m src.rag_chain
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import get_chat_model

RAG_SYSTEM_PROMPT = (
    "You are a contract intelligence assistant. Answer only using the "
    "provided context. If the answer is not in the context, say so clearly "
    "-- never guess or make up an answer. Cite the source file for every "
    "claim, in the form [filename].\n\n"
    "Context:\n{context}"
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything relevant to that in the ingested documents. "
    "Try rephrasing, or check that the right contract has been ingested."
)


def format_docs(docs) -> str:
    """Turn retrieved Documents into a citable context block. Returns an
    explicit empty-context sentinel rather than an empty string, so the
    empty-retrieval case is handled deterministically instead of leaning
    solely on the model to notice an empty context and behave itself."""
    if not docs:
        return "NO_RELEVANT_DOCUMENTS_FOUND"
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs
    )


def _short_circuit_if_empty(inputs: dict):
    """Runs after context is formatted but before the prompt/model. If
    retrieval found nothing, skip the LLM call entirely and return the
    canned message -- guarantees no hallucinated answer on empty retrieval,
    rather than depending only on the system prompt to catch it."""
    if inputs["context"] == "NO_RELEVANT_DOCUMENTS_FOUND":
        return NO_CONTEXT_MESSAGE
    return None


def build_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
    )
    model = get_chat_model()

    core_chain = prompt | model | StrOutputParser()

    def route(inputs: dict) -> str:
        short_circuit = _short_circuit_if_empty(inputs)
        if short_circuit is not None:
            return short_circuit
        return core_chain.invoke(inputs)

    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | RunnableLambda(route)
    )


if __name__ == "__main__":
    from src.ingest import ingest

    vectorstore, _ = ingest()
    if vectorstore is None:
        # Nothing new was ingested this run (manifest says everything's
        # current) -- reopen the persisted store instead.
        from langchain_chroma import Chroma

        from src.config import CHROMA_DIR, get_embeddings

        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings()
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = build_rag_chain(retriever)

    print("\nClauseIQ -- basic RAG chat (Day 1). Ctrl+C to exit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        answer = rag_chain.invoke(question)
        print(f"\nClauseIQ: {answer}\n")
