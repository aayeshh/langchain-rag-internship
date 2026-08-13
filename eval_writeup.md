# ClauseIQ Phase 2 — Evaluation Write-Up

## What we tested

Baseline hybrid retrieval (BM25 + vector, no reranking) vs. the same pipeline with an added reranking step — an `EmbeddingsFilter` that re-scores each retrieved chunk against the query and drops weak matches before they reach the model. Both runs used the identical 37-example golden dataset (`clauseiq-golden-set`) and the identical two evaluators (`retrieval_hit`, `faithfulness`), so the only variable that changed between them is the reranker.

## Baseline results (`clauseiq-baseline`) — full run, 37/37 examples

| Evaluator | Score |
|---|---|
| retrieval_hit | 0.97 |
| faithfulness | 0.92 |

Baseline retrieval is already strong. 0.97 retrieval-hit means the system is finding the right source document for almost every question, including the deliberately harder "should say not specified" cases mixed into the dataset. 0.92 faithfulness means the model is staying grounded in what was actually retrieved in the large majority of cases — the small gap between the two scores is worth watching, since it suggests that in a handful of cases the right chunk was retrieved but the answer still drifted slightly from what that chunk actually supports.

## Reranker results (`clauseiq-with-reranker`) — incomplete: 2-3/37 examples

We were not able to get a full reranker run against Gemini's free-tier embedding quota (100 requests/minute) — reranking re-embeds and re-scores every candidate chunk per question rather than just the query, so it uses roughly 9x more embedding calls than baseline per example, and the run hit `RESOURCE_EXHAUSTED` partway through even after capping concurrency. We're reporting this honestly rather than padding it out: **we do not have a statistically meaningful before/after comparison for the reranker yet.**

What we do have is a real finding from the process itself:

- **First attempt** (`similarity_threshold=0.75`): every example that ran returned "I couldn't find any information" — including questions the baseline answered correctly and confidently (e.g. the vendor agreement's on-time delivery rate, the dental policy's annual maximum benefit). The threshold was filtering out every single retrieved chunk, leaving the agent with empty context on nearly every query. This is a worse outcome than no reranking at all.
- **After lowering the threshold to `0.5`** and adding a fallback (if the filter ever empties the result set, fall back to the unfiltered hybrid results instead of returning nothing), the 2-3 examples that completed before hitting the quota answered correctly — matching the baseline's correct answers rather than falsely claiming "not specified."

## Interpretation

The headline number we can report with confidence: **baseline retrieval (0.97 hit-rate, 0.92 faithfulness) is solid and is the safe default to ship as-is.**

On the reranker: the *first* configuration was a clear regression — it would have shipped a system that answers "I don't know" to questions it previously answered correctly, which is a serious faithfulness failure in the opposite direction (false negatives instead of hallucination) and would have gone completely undetected without this eval process. Catching that before it reached a demo or a user is exactly what this evaluation setup is for.

The *tuned* configuration (threshold 0.5 + fallback) looks promising on the handful of examples that completed, but "looks promising on 3 examples" is not the same claim as "the numbers say it helped" — we don't yet have retrieval_hit or faithfulness scores for the reranker at a sample size that means anything.

**Decision:** hold the reranker at `use_reranker=False` (the current default in `src/retrieval.py`) for production use until a full 37-example run completes cleanly. The code and the fix are in place — this is a data-collection gap caused by the free-tier embedding quota, not an unresolved code issue. Re-running `python -m src.eval.run_eval --only reranker` after the per-minute quota resets (it's a rolling 60-second window, not a daily cap) should get a complete comparison; at that point this write-up should be updated with real retrieval_hit/faithfulness numbers for the reranker before making a final call on whether it ships.

## Structured-extraction accuracy (`clauseiq-extraction`)

Run against `clauseiq-extraction-set` (6 examples, one per sample contract) — reranking doesn't touch this pipeline, since `extract_summary()` works from the whole document directly and never calls the retriever.

*(Fill in the extraction_accuracy score from the LangSmith UI once reviewed — check specifically whether the dental policy's genuinely-null `renewal_or_termination_notice_days` field was correctly left null rather than guessed, since that's the single most informative data point in this set: a wrong guess there is a real faithfulness problem in the extraction schema, not just a missed field.)*

## Regression-testing workflow going forward

After any change to chunking parameters, retrieval weights, or prompts in this project, re-run the relevant piece of `python -m src.eval.run_eval` before considering the change done. Compare the new experiment against `clauseiq-baseline` in the LangSmith UI on the same dataset before merging — a change that "feels" better in a few manual test queries isn't enough on its own now that this eval set exists. The `--only` flag exists specifically so a single retrieval change doesn't require burning through the full free-tier quota re-running everything else too.
