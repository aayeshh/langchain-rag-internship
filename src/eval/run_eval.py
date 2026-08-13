"""
Runs the Phase 2 LangSmith experiments:

1. QA baseline (reranker off) against clauseiq-golden-set
2. QA with-reranker (the "one deliberate pipeline change") against the
   *same* dataset, same evaluators -- the before/after comparison
3. Extraction accuracy against clauseiq-extraction-set (run once --
   reranking doesn't affect extraction, which never touches the
   retriever, so there's no "before/after" to compare here)

Run all three: python -m src.eval.run_eval
Run just one (e.g. after re-hitting a quota mid-run, or for future
regression testing after a pipeline change): 
    python -m src.eval.run_eval --only baseline
    python -m src.eval.run_eval --only reranker
    python -m src.eval.run_eval --only extraction
"""

# Explicit, not relied-upon-via-import-order -- see the same comment in
# src/eval/dataset.py for why this matters here specifically.
from dotenv import load_dotenv

load_dotenv()

import argparse

from src.eval.dataset import EXTRACTION_DATASET_NAME, QA_DATASET_NAME, build_extraction_dataset, build_qa_dataset
from src.eval.evaluators import extraction_field_evaluator, faithfulness_evaluator, retrieval_hit_evaluator
from src.eval.predict import build_qa_predict, predict_extraction


def _get_vectorstore():
    from langchain_chroma import Chroma

    from src.config import CHROMA_DIR, get_embeddings
    from src.ingest import ingest

    vectorstore, _ = ingest()
    if vectorstore is not None:
        return vectorstore
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())


def run_qa_experiment(vectorstore, use_reranker: bool, experiment_prefix: str):
    from langsmith.evaluation import evaluate

    from src.agent import build_agent
    from src.retrieval import build_hybrid_retriever

    retriever = build_hybrid_retriever(vectorstore, use_reranker=use_reranker)
    agent = build_agent(retriever)
    predict = build_qa_predict(agent, retriever)

    return evaluate(
        predict,
        data=QA_DATASET_NAME,
        evaluators=[retrieval_hit_evaluator, faithfulness_evaluator],
        experiment_prefix=experiment_prefix,
        metadata={"reranker": "on" if use_reranker else "off"},
        # Reranker mode fires ~9x more embedding calls per question
        # (re-scores every candidate chunk, not just the query) -- capping
        # concurrency to 2 spreads those out instead of bursting past
        # Gemini's free-tier per-minute embedding quota the way an
        # unthrottled run did.
        max_concurrency=2 if use_reranker else None,
    )


def run_extraction_experiment(experiment_prefix: str = "clauseiq-extraction"):
    from langsmith.evaluation import evaluate

    return evaluate(
        predict_extraction,
        data=EXTRACTION_DATASET_NAME,
        evaluators=[extraction_field_evaluator],
        experiment_prefix=experiment_prefix,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["baseline", "reranker", "extraction"],
        default=None,
        help="Run just one experiment instead of all three. Datasets are "
        "created once and reused (idempotent) -- safe to re-run any single "
        "experiment without re-running the others.",
    )
    args = parser.parse_args()

    print("Ensuring datasets exist...")
    build_qa_dataset()
    build_extraction_dataset()

    print("\nBuilding vector store / retriever base...")
    vectorstore = _get_vectorstore()

    if args.only in (None, "baseline"):
        print("\n=== Experiment 1: QA baseline (reranker off) ===")
        run_qa_experiment(vectorstore, use_reranker=False, experiment_prefix="clauseiq-baseline")

    if args.only in (None, "reranker"):
        print("\n=== Experiment 2: QA with reranker (the deliberate change) ===")
        run_qa_experiment(vectorstore, use_reranker=True, experiment_prefix="clauseiq-with-reranker")

    if args.only in (None, "extraction"):
        print("\n=== Experiment 3: Structured extraction accuracy ===")
        run_extraction_experiment()

    print(
        "\nDone. Open the LangSmith UI, find the 'clauseiq-golden-set' "
        "dataset, and compare the 'clauseiq-baseline' and "
        "'clauseiq-with-reranker' experiments side by side."
    )


if __name__ == "__main__":
    main()
