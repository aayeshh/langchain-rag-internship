# ClauseIQ — AI Contract & Policy Intelligence Assistant

ClauseIQ turns a folder of contracts (leases, vendor agreements, NDAs, insurance policies, supplier terms) into something you can ask questions to — with cited answers, automatic structured summaries, and proactive deadline/risk flagging. Built with LangChain (no LangGraph state-graph code, no LangSmith, no MCP).

## What it does

- **Ingests** PDF and DOCX contracts, splits them into chunks, embeds them, and persists them to a local vector store — skipping re-embedding for files that haven't changed.
- **Extracts a structured summary** for every document on ingest: parties, effective date, renewal/termination notice period, payment terms, and a flagged key risk — via typed structured output, not free-form text.
- **Answers questions** over the documents using hybrid search (keyword + semantic), with every answer citing its source file, and a hard guarantee against hallucinated answers when nothing relevant is found.
- **Takes action**, not just retrieves: a tool that calculates days remaining until a renewal/termination deadline and flags urgency, and a tool that drafts a short follow-up email about a renewal or flagged risk.
- **Remembers conversation context** within a session (thread-scoped memory via a LangGraph checkpointer, wired in through `create_agent` — the only LangGraph runtime piece this project touches, per the project constraints).
- **Caches LLM responses** during dev/testing so repeated identical queries don't re-hit the API.

## Tech stack

- **LangChain** (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-chroma`)
- **Chat model:** Groq (`openai/gpt-oss-20b`) — chosen for reliable tool-calling on a free tier
- **Embeddings:** Google Gemini (`text-embedding-004`)
- **Vector store:** Chroma (persisted locally)
- **Retrieval:** `BM25Retriever` + vector search combined via `EnsembleRetriever`
- **Agent:** `create_agent` with a checkpointer for thread-scoped memory

Swapping either model provider only requires changing `CHAT_MODEL_NAME` / `EMBEDDING_MODEL_NAME` and the corresponding client in `src/config.py` — nothing else in the project touches provider specifics directly.

## Project structure

```
clauseIQ/
├── .env.example          # copy to .env and fill in your keys
├── requirements.txt
├── README.md
├── demo_script.md         # 5-minute client-facing demo script
├── sellability.md          # who buys this, problem solved, pricing angle
├── data/
│   ├── contracts/           # sample contracts used for testing (6 files, 4 verticals)
│   └── summaries/             # generated on ingest — one JSON summary per contract
├── chroma_db/                  # generated on ingest — persisted vector store
└── src/
    ├── config.py                # paths, chunking params, model setup
    ├── ingest.py                  # load → extract → split → embed → persist (with caching)
    ├── extraction.py               # structured contract summary extraction
    ├── retrieval.py                  # hybrid BM25 + vector retriever
    ├── rag_chain.py                    # standalone LCEL RAG chain (Day 1 milestone / reference)
    ├── tools.py                          # deadline-urgency and email-draft tools
    ├── agent.py                            # create_agent + checkpointer + LLM cache setup
    └── cli.py                                # main entrypoint — the chat loop
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
   - Gemini (embeddings): [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
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

## Sample data

`data/contracts/` contains 6 sample documents spanning real estate (residential lease), hospitality (vendor/linen service agreement), general business (mutual NDA), healthcare (dental insurance policy summary), e-commerce (supplier terms), and one deliberately garbled/OCR-style PDF used to test robustness against badly-formatted documents. All are synthetic/template-based — no real client data.

## Notes on design choices

- **Chunking** (1000/150 chars) keeps most contract clauses intact in a single chunk while still giving the retriever resolution to pull a specific clause rather than a whole page.
- **Hybrid retrieval weighting** (0.4 BM25 / 0.6 vector) favors semantic matching for natural-language questions while still catching exact terms — dollar amounts, clause numbers, party names — that embeddings alone can blur.
- **Structured extraction fields are `Optional`** by design — a document that doesn't state a termination notice period (e.g. the dental insurance sample) gets `null`, not a guessed value.
- **Empty retrieval is handled deterministically**, not just via prompt instruction: if the retriever returns nothing relevant, the system returns a canned "not found" response without calling the LLM at all.
