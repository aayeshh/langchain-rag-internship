"""
ClauseIQ CLI -- the main entrypoint for the project.

Run: python -m src.cli

On startup: runs ingestion (skips unchanged files), opens the persisted
vector store, builds the hybrid retriever, and starts an interactive chat
loop backed by the agent with thread-scoped memory (one thread_id per CLI
run, so history persists across turns in the same session but not across
separate runs -- InMemorySaver is in-process only by design here; swap in
a persistent checkpointer if cross-session memory is ever needed).
"""

import uuid

from src.agent import build_agent
from src.config import CHROMA_DIR, get_embeddings
from src.ingest import ingest
from src.retrieval import build_hybrid_retriever


def get_or_build_vectorstore():
    vectorstore, _ = ingest()
    if vectorstore is not None:
        return vectorstore

    # Nothing new was ingested (manifest says everything's current) --
    # reopen the existing persisted store.
    from langchain_chroma import Chroma

    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())


def main():
    print("ClauseIQ -- starting up...\n")
    vectorstore = get_or_build_vectorstore()
    retriever = build_hybrid_retriever(vectorstore)
    agent = build_agent(retriever)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nClauseIQ is ready. Ask about your contracts, or ask for a")
    print("deadline check or a follow-up email draft. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not question:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config
        )
        answer = result["messages"][-1].content
        print(f"\nClauseIQ: {answer}\n")


if __name__ == "__main__":
    main()
