"""
Evaluators for the two LangSmith experiments (QA and extraction).

Each evaluator takes (run, example) and returns a dict with at least
"key" and "score". run.outputs is whatever src/eval/predict.py returned
for that example; example.outputs is the ground truth from
src/eval/dataset.py.
"""

from pydantic import BaseModel, Field


# --- Retrieval hit-rate (QA dataset) ---------------------------------------

def retrieval_hit_evaluator(run, example) -> dict:
    """Did the source file that actually contains the answer get retrieved
    at all? Checks run.outputs['context'] (the raw retrieved chunks,
    tagged with their source filenames by format_docs) rather than the
    final answer text -- a wrong final answer could still cite the right
    source, and a right final answer could theoretically be a lucky guess
    without the source actually having been retrieved, so context is the
    correct thing to check for this specific evaluator."""
    expected_source = example.outputs.get("expected_source")
    if expected_source is None:
        # "Should say not specified" cases have no source to hit --
        # not applicable, not a failure. LangSmith aggregates None scores
        # separately from 0s, so this doesn't silently deflate the
        # aggregate hit-rate on cases where there was nothing to retrieve.
        return {"key": "retrieval_hit", "score": None}

    context = run.outputs.get("context", "")
    hit = expected_source in context
    return {"key": "retrieval_hit", "score": int(hit)}


# --- Faithfulness (LLM-as-judge, QA dataset) --------------------------------

class FaithfulnessGrade(BaseModel):
    faithful: bool = Field(
        description="True only if every claim in the answer is directly supported by the context"
    )
    reasoning: str = Field(description="One or two sentences explaining the grade")


def faithfulness_evaluator(run, example) -> dict:
    """LLM-as-judge: does the answer only make claims supported by the
    retrieved context? Runs on Gemini (get_judge_model), not the Groq
    model the agent itself uses -- see src/config.py's JUDGE_MODEL_NAME
    comment for why (keeps eval traffic off Groq's tighter free tier)."""
    from src.config import get_judge_model

    context = run.outputs.get("context", "")
    answer = run.outputs.get("answer", "")

    if not context.strip():
        # Nothing was retrieved -- faithfulness is vacuously fine as long
        # as the answer correctly says so, rather than asking the judge to
        # grade faithfulness against an empty context (which is a
        # different, ill-posed question).
        said_not_found = any(
            phrase in answer.lower()
            for phrase in ["not specified", "couldn't find", "no relevant", "don't have"]
        )
        return {
            "key": "faithfulness",
            "score": int(said_not_found),
            "comment": "No context retrieved; scored on whether the answer correctly declined to guess.",
        }

    judge = get_judge_model().with_structured_output(FaithfulnessGrade)
    result = judge.invoke(
        f"Context retrieved by the system:\n{context}\n\n"
        f"Answer given:\n{answer}\n\n"
        "Does the answer ONLY make claims that are directly supported by "
        "the context above? Answer strictly based on this -- don't reward "
        "a correct-sounding answer that isn't actually grounded in the "
        "context."
    )
    return {
        "key": "faithfulness",
        "score": int(result.faithful),
        "comment": result.reasoning,
    }


# --- Structured-extraction field accuracy (extraction dataset) -------------

def extraction_field_evaluator(run, example) -> dict:
    """Compares each ground-truth field in example.outputs against the
    matching field in run.outputs (from predict_extraction -- the actual
    src.extraction.extract_summary() result for that document), field by
    field. Returns a per-field breakdown plus an overall score, rather
    than one opaque pass/fail, so a partial miss (e.g. right parties,
    wrong date) is visible in the results instead of just failing the
    whole row."""
    expected = example.outputs
    actual = run.outputs

    field_results = {}
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if isinstance(expected_value, list):
            # Party name matching: order-insensitive, case-insensitive.
            match = sorted(str(v).lower() for v in (actual_value or [])) == sorted(
                str(v).lower() for v in expected_value
            )
        elif expected_value is None:
            # A field that should genuinely be null (e.g. the dental
            # policy's notice period) only passes if the model also left
            # it null -- NOT if it happens to guess something that looks
            # plausible. This is the case that would catch a faithfulness
            # regression in the extraction schema itself.
            match = actual_value is None
        else:
            match = str(actual_value).strip().lower() == str(expected_value).strip().lower()
        field_results[field] = match

    overall_score = sum(field_results.values()) / len(field_results) if field_results else 0

    return {
        "key": "extraction_accuracy",
        "score": overall_score,
        "comment": f"Field matches: {field_results}",
    }
