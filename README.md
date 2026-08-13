# ClauseIQ — AI Contract & Policy Intelligence Assistant

ClauseIQ turns a folder of contracts (leases, vendor agreements, NDAs, insurance policies, supplier terms) into something you can ask questions to — with cited answers, automatic structured summaries, and proactive deadline/risk flagging. Phase 2 adds LangSmith tracing and evaluation, so changes to the pipeline can be measured, not just eyeballed.

Built with LangChain (no LangGraph state-graph code, no LangSmith Deployment, no MCP).

## What it does

- **Ingests** PDF and DOCX contracts, splits them into chunks, embeds them, and persists them to a local vector store — skipping re-embedding for files that haven't changed.
- **Extracts a structured summary** for every document on ingest: parties, effective date, renewal/termination notice period, payment terms, and a flagged key risk — via typed structured output, not free-form text.
- **Answers questions** over the documents using hybrid search (keyword + semantic), with every answer citing its source file, and a hard guarantee against hallucinated answers when nothing relevant is found.
- **Takes action**, not just retrieves: a tool that calculates days remaining until a renewal/termination deadline and flags urgency, and a tool that drafts a short follow-up email about a renewal or flagged risk.
- **Remembers conversation context** within a session (thread-scoped memory via a LangGraph checkpointer, wired in through `create_agent` — the only LangGraph runtime piece this project touches, per the project constraints).
- **Caches LLM responses** during dev/testing so repeated identical queries don't re-hit the API.
- **Traces every query end-to-end** in LangSmith (retrieval → tool calls → generation), and can be evaluated against a 37-question golden dataset with automated faithfulness, retrieval-hit, and structured-extraction-accuracy scoring — so a pipeline change (like adding a reranking step) can be proven to help or hurt with real numbers, not just a vibe check.

## Tech stack

- **LangChain** (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-chroma`)
- **Chat model:** Groq (`openai/gpt-oss-120b`) — chosen for reliable multi-tool-call tool-calling on a free tier (smaller `gpt-oss-20b` occasionally produced malformed tool calls under a multi-tool turn)
- **Embeddings:** Google Gemini (`gemini-embedding-001`)
- **Judge model (eval only):** Google Gemini (`gemini-3.5-flash-lite`) — kept on a separate provider from the main Groq agent so a 30-50+ call eval run doesn't stack onto Groq's tighter free-tier limits
- **Vector store:** Chroma (persisted locally)
- **Retrieval:** `BM25Retriever` + vector search combined via `EnsembleRetriever`, with an optional `EmbeddingsFilter` reranking step
- **Agent:** `create_agent` with a checkpointer for thread-scoped memory
- **Evaluation/observability:** LangSmith (`langsmith`) — tracing, datasets, evaluators, experiment comparison

Swapping any model provider only requires changing the relevant `*_MODEL_NAME` constant and its client function in `src/config.py` — nothing else in the project touches provider specifics directly.

**Note on model IDs:** both Groq and Gemini retire model versions periodically. This project has already hit two retirements during development (Gemini's `text-embedding-004` on Jan 14, 2026, and `gemini-2.0-flash` on June 1, 2026). If you get a 404 "model not found" error, check the provider's current model list before assuming the code is broken.

## Project structure

```
clauseIQ/
├── .env.example              # copy to .env and fill in your keys
├── requirements.txt
├── README.md
├── demo_script.md             # 5-minute client-facing demo script
├── sellability.md              # who buys this, problem solved, pricing angle
├── eval_writeup.md              # Phase 2: baseline vs. reranker results and interpretation
├── data/
│   ├── contracts/                # sample contracts used for testing (6 files, 4 verticals)
│   └── summaries/                  # generated on ingest — one JSON summary per contract (gitignored)
├── chroma_db/                        # generated on ingest — persisted vector store (gitignored)
├── .langchain_cache.db                 # generated — SQLite LLM response cache (gitignored)
└── src/
    ├── config.py                        # paths, chunking params, model setup
    ├── ingest.py                          # load → extract → split → embed → persist (with caching)
    ├── extraction.py                       # structured contract summary extraction
    ├── retrieval.py                          # hybrid BM25 + vector retriever, optional reranker
    ├── rag_chain.py                            # standalone LCEL RAG chain (Day 1 milestone / reference)
    ├── tools.py                                  # deadline-urgency and email-draft tools
    ├── agent.py                                    # create_agent + checkpointer + LLM cache setup
    ├── cli.py                                        # main entrypoint — the chat loop
    └── eval/                                           # Phase 2 — LangSmith evaluation
        ├── dataset.py                                     # 37 QA + 6 extraction ground-truth examples
        ├── evaluators.py                                    # faithfulness, retrieval-hit, extraction-accuracy
        ├── predict.py                                         # wraps the agent/extraction for evaluate()
        ├── run_eval.py                                          # experiment runner (--only flag for re-runs)
        └── online_eval.py                                         # bonus: live-traffic monitoring
```

## Setup

1. **Clone and enter the project, create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get free API keys and set up `.env`:**
   - Groq (chat model): [console.groq.com/keys](https://console.groq.com/keys)
   - Gemini (embeddings + judge model): [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - LangSmith (tracing + eval): [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys
   ```bash
   cp .env.example .env
   # then edit .env and paste in your real keys
   ```

4. **Run ingestion** (loads the sample contracts, extracts summaries, embeds, and persists — only needs to fully run once; later runs skip unchanged files):
   ```bash
   python -m src.ingest
   ```

5. **Start chatting:**
   ```bash
   python -m src.cli
   ```
   With `LANGSMITH_TRACING=true` set in `.env`, every query is automatically traced — check the `clauseiq-dev` project at [smith.langchain.com](https://smith.langchain.com) to see the full nested trace (retrieval → tool call → generation) for any query.

6. **Run the evaluation suite** (creates the golden datasets if they don't exist yet, then runs baseline, reranker, and extraction experiments):
   ```bash
   python -m src.eval.run_eval
   ```
   To re-run just one experiment (e.g. after a pipeline change, without re-running the others):
   ```bash
   python -m src.eval.run_eval --only baseline
   python -m src.eval.run_eval --only reranker
   python -m src.eval.run_eval --only extraction
   ```
   Results appear in the LangSmith UI under the `clauseiq-golden-set` and `clauseiq-extraction-set` datasets — open two experiments together for the side-by-side compare view.

## Sample data

`data/contracts/` contains 6 sample documents spanning real estate (residential lease), hospitality (vendor/linen service agreement), general business (mutual NDA), healthcare (dental insurance policy summary), e-commerce (supplier terms), and one deliberately garbled/OCR-style PDF used to test robustness against badly-formatted documents. All are synthetic/template-based — no real client data. The 37-example golden Q&A set and 6-example extraction ground-truth set in `src/eval/dataset.py` are built directly from these documents' actual content.

## Notes on design choices

- **Chunking** (1000/150 chars) keeps most contract clauses intact in a single chunk while still giving the retriever resolution to pull a specific clause rather than a whole page.
- **Hybrid retrieval weighting** (0.4 BM25 / 0.6 vector) favors semantic matching for natural-language questions while still catching exact terms — dollar amounts, clause numbers, party names — that embeddings alone can blur.
- **Structured extraction fields are `Optional`** by design — a document that doesn't state a termination notice period (e.g. the dental insurance sample) gets `null`, not a guessed value.
- **Empty retrieval is handled deterministically**, not just via prompt instruction: if the retriever returns nothing relevant, the system returns a canned "not found" response without calling the LLM at all.
- **The reranker has a fallback safety net**: if the `EmbeddingsFilter` compression step ever filters out every candidate chunk (a mis-tuned threshold, a different embedding model), `src/retrieval.py` falls back to the unfiltered hybrid results rather than leaving the agent with empty context. This was tuned in practice — an initial `similarity_threshold=0.75` was silently emptying context on nearly every query; `0.5` plus the fallback fixed it. See `eval_writeup.md` for what this looked like in the actual eval numbers.
- **Eval datasets are idempotent to build**: `src/eval/dataset.py` checks whether a dataset already has examples before adding more, so re-running `python -m src.eval.run_eval` any number of times never duplicates rows.
