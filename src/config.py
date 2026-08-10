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
CHAT_MODEL_NAME = "groq:openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "gemini-embedding-2"


def get_chat_model(temperature: float = 0):
    """Single place to construct the chat model, so every module (ingest,
    extraction, RAG chain, agent) uses the exact same model config."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(CHAT_MODEL_NAME, temperature=temperature)


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
