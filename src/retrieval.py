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


def build_hybrid_retriever(vectorstore, use_reranker: bool = False):
    """vectorstore: a persisted Chroma store already containing all chunks
    (old and new) from src.ingest. BM25 has no persistence of its own, so
    its index is rebuilt from the vector store's contents each time the
    app starts -- a local rebuild, not a re-embedding, so it costs no
    API calls.

    use_reranker: when True, wraps the ensemble retriever in a
    ContextualCompressionRetriever using an EmbeddingsFilter -- this is the
    "one deliberate pipeline change" Phase 2's before/after LangSmith
    experiment compares. EmbeddingsFilter re-scores each already-retrieved
    chunk by embedding similarity to the query and drops weak matches,
    rather than an LLM-based reranker -- deliberately chosen so reranking
    costs embedding calls (Gemini, generous free tier) instead of extra
    chat-model calls (Groq, the tighter free tier)."""
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

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=HYBRID_WEIGHTS,
    )

    if not use_reranker:
        return ensemble

    try:
        from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
    except ImportError:
        from langchain.retrievers.document_compressors import EmbeddingsFilter

    from langchain_core.runnables import RunnableLambda

    from src.config import get_embeddings

    # 0.75 turned out to be far too aggressive in practice -- cosine
    # similarity between a natural-language question and its answering
    # passage is often well below 0.7 even when highly relevant (they're
    # different surface forms, not paraphrases of each other), so 0.75
    # was silently filtering out EVERY candidate chunk on nearly every
    # query, leaving the agent with empty context and no way to answer
    # questions the baseline retriever handled correctly. 0.5 is a
    # gentler cut that should still drop genuinely irrelevant near-misses
    # without wiping out real matches.
    compressor = EmbeddingsFilter(embeddings=get_embeddings(), similarity_threshold=0.5)

    def rerank_with_fallback(query: str):
        """Re-scores the hybrid retriever's own results and drops weak
        matches -- but never returns fewer than zero usable chunks when
        the hybrid retriever found something. If the threshold (whatever
        it's set to) turns out to filter out every candidate for a given
        query, that's a mis-tuned filter, not a real 'nothing relevant
        exists' case -- so fall back to the unfiltered hybrid results
        rather than leaving the agent worse off than no reranking at all.
        This is the fix for the exact failure observed in testing: 0.75
        emptied the context on questions the baseline answered fine."""
        original_docs = ensemble.invoke(query)
        if not original_docs:
            return original_docs
        filtered_docs = compressor.compress_documents(original_docs, query)
        return list(filtered_docs) if filtered_docs else original_docs

    return RunnableLambda(rerank_with_fallback)
