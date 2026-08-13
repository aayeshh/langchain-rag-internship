"""
Central config: env loading, model init, and shared paths.

Every entrypoint script should import from here rather than re-loading
.env or re-instantiating models — keeps chunking params, model choice,
and paths consistent across ingestion, retrieval, and the agent.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = BASE_DIR / "data" / "contracts"
CHROMA_DIR = BASE_DIR / "chroma_db"
CACHE_DB_PATH = BASE_DIR / ".langchain_cache.db"

# --- Chunking ------------------------------------------------------------
# 1000/150 is the guide's suggested default. Contracts have long, dense
# paragraphs (a whole clause is often 300-800 chars), so this keeps most
# clauses intact in a single chunk while still giving the vector store
# enough resolution to retrieve a specific clause rather than a whole page.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval -----------------------------------------------------------
# k=4 per retriever before merging is the guide's default; contract Q&A
# tends to need 1-3 supporting chunks per answer, so 4 leaves headroom
# without flooding the prompt.
RETRIEVER_K = 4
# Weight vector search slightly higher than BM25: most contract questions
# are phrased in natural language ("when can we get out of this?") rather
# than exact-term lookups, but BM25 still matters for dollar amounts,
# party names, and clause numbers that embeddings can blur.
HYBRID_WEIGHTS = [0.4, 0.6]  # [bm25, vector]

# --- Models ----------------------------------------------------------------
# Using Groq (chat) + Gemini (embeddings) since both have usable free tiers,
# instead of Anthropic/OpenAI which require paid API keys. Swapping providers
# only means changing these two lines and get_chat_model/get_embeddings below
# -- nothing else in the project touches model provider specifics directly.
#
# Model choice matters here specifically because this project relies on
# reliable tool-calling (search_documents, check_deadline_urgency,
# draft_renewal_email). Groq's llama-3.3-70b-versatile is fast but its tool
# calls occasionally come back malformed ("tool_use_failed" from the API).
# openai/gpt-oss-20b (OpenAI's open-weight model, hosted on Groq) is built
# for reliable function calling and is also on Groq's free tier -- use that
# instead. If you hit tool_use_failed errors again, groq:openai/gpt-oss-120b
# is the larger, even more reliable (but slower) alternative.
CHAT_MODEL_NAME = "groq:openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

# Phase 2 (LangSmith eval) adds a second model role: the LLM-as-judge used by
# the faithfulness evaluator. An eval run calls this once per dataset example
# (30-50+ calls per experiment, and you'll run at least two experiments) --
# stacking that onto the same Groq key used by the main agent risks hitting
# its free-tier token/request limit fast. Gemini already has a paid-free-tier
# presence in this project via embeddings, so the judge uses Gemini's chat
# model too, keeping judge-eval traffic off Groq entirely.
JUDGE_MODEL_NAME = "google_genai:gemini-3.5-flash-lite"


def get_chat_model(temperature: float = 0):
    """Single place to construct the chat model, so every module (ingest,
    extraction, RAG chain, agent) uses the exact same model config."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(CHAT_MODEL_NAME, temperature=temperature)


def get_judge_model(temperature: float = 0):
    """The LLM-as-judge model for evaluators (src/eval/evaluators.py).
    Deliberately a different provider (Gemini) than get_chat_model()'s Groq
    -- see JUDGE_MODEL_NAME comment above."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(JUDGE_MODEL_NAME, temperature=temperature)


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
