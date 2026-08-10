"""
Hybrid retrieval: BM25 (keyword) + vector (dense) combined via
EnsembleRetriever. Dense search handles paraphrased questions; BM25 catches
exact terms -- contract IDs, dollar amounts, party names -- that embeddings
can blur together.

BM25Retriever is built in-memory from the same chunk set used to build the
vector store, so it needs the full chunk list at build time (unlike the
vector store, it isn't persisted -- rebuilding it from already-persisted
chunks is cheap, no embedding calls involved).
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import HYBRID_WEIGHTS, RETRIEVER_K


def _load_all_chunks(vectorstore) -> list[Document]:
    """Pull every chunk currently persisted in Chroma back out as Documents.
    Used to (re)build the BM25 index so it always mirrors the full corpus
    -- including chunks embedded in earlier runs, not just this session's
    newly-added ones. This is a local read, not an API call."""
    raw = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def build_hybrid_retriever(vectorstore):
    """vectorstore: a persisted Chroma store already containing all chunks
    (old and new) from src.ingest. BM25 has no persistence of its own, so
    its index is rebuilt from the vector store's contents each time the
    app starts -- a local rebuild, not a re-embedding, so it costs no
    API calls."""
    try:
        # LangChain 1.0+ moved EnsembleRetriever into langchain-classic.
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        # Older langchain (<1.0) still has it under langchain.retrievers.
        from langchain.retrievers import EnsembleRetriever

    all_chunks = _load_all_chunks(vectorstore)
    if not all_chunks:
        raise ValueError(
            "No chunks found in the vector store. Run ingestion first: "
            "python -m src.ingest"
        )

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = RETRIEVER_K

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=HYBRID_WEIGHTS,
    )
