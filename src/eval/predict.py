"""
predict functions: the thing `evaluate()` actually calls per dataset
example. Each takes a dataset row's `inputs` dict and returns an `outputs`
dict shaped to match what the corresponding evaluators expect.

Two predict functions because the two datasets test two different
pipelines (see src/eval/dataset.py's module docstring).
"""

import uuid


def build_qa_predict(agent, retriever):
    """Returns a predict(inputs) function closed over a specific agent +
    retriever pair, so run_eval.py can build one for the baseline retriever
    and another for the reranked retriever without global state."""

    def predict(inputs: dict) -> dict:
        question = inputs["question"]

        # Retrieved separately (not just read off the agent's tool call)
        # so we always have context to hand the faithfulness evaluator,
        # even on a turn where the agent's tool-calling behaves
        # unexpectedly -- context capture shouldn't depend on the agent
        # having called the tool "correctly."
        from src.rag_chain import format_docs

        retrieved = retriever.invoke(question)
        context = format_docs(retrieved)

        # Fresh thread per example so eval runs don't leak conversation
        # memory between unrelated dataset rows.
        thread_id = f"eval-{uuid.uuid4()}"
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].content

        return {"answer": answer, "context": context}

    return predict


def predict_extraction(inputs: dict) -> dict:
    """Calls src.extraction.extract_summary directly against the real
    document text on disk -- deliberately bypasses retrieval/the agent
    entirely, since extraction accuracy is a property of the extraction
    pipeline, not of retrieval."""
    from src.config import CONTRACTS_DIR
    from src.extraction import ContractSummary, _combine_pages, extract_summary
    from src.ingest import LOADERS_BY_SUFFIX

    filename = inputs["filename"]
    file_path = CONTRACTS_DIR / filename
    loader_cls = LOADERS_BY_SUFFIX[file_path.suffix.lower()]
    docs = loader_cls(str(file_path)).load()
    full_text = _combine_pages(docs)

    summary: ContractSummary = extract_summary(filename, full_text)
    return summary.model_dump()
