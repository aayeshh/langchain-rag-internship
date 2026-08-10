"""
The ClauseIQ agent: create_agent wired with a document-search tool (backed
by hybrid retrieval), the two action tools, and thread-scoped conversation
memory via a checkpointer. LLM response caching is enabled at import time
so repeated identical calls during dev/testing don't re-hit the API.

thread_id = one continuous conversation. src/cli.py assigns one thread_id
per CLI session; a future web front end would need one per user session.
"""

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.config import CACHE_DB_PATH, get_chat_model

# Enabled once, at import time, so every chat model call anywhere in the
# app (agent, RAG chain, extraction) benefits -- required by the brief's
# "set_llm_cache in place for dev/testing" checklist item.
set_llm_cache(SQLiteCache(database_path=str(CACHE_DB_PATH)))

SYSTEM_PROMPT = (
    "You are ClauseIQ, a contract operations assistant. You help users "
    "understand contracts and policies in their document set, and take "
    "action on renewal deadlines and flagged risks.\n\n"
    "Rules:\n"
    "- Use search_documents for any question about what a contract says. "
    "Answer only from what it returns, and cite the source file for every "
    "claim in the form [filename]. If search_documents finds nothing "
    "relevant, say so plainly -- never guess or fabricate contract terms.\n"
    "- Use check_deadline_urgency for any date math on a renewal or "
    "termination deadline -- never estimate days yourself.\n"
    "- Use draft_renewal_email when the user asks for a follow-up email "
    "about a renewal or risk.\n"
    "- Keep answers concise and cite sources."
)


def build_search_tool(retriever):
    """Wraps the hybrid retriever as a tool the agent can call. Built as a
    factory (not a module-level tool) because the retriever depends on a
    live vectorstore connection that's only available after ingestion."""
    from langchain_core.tools import tool

    from src.rag_chain import format_docs

    @tool
    def search_documents(query: str) -> str:
        """Search the ingested contract/policy documents and return the
        most relevant passages, each tagged with its source filename.

        Args:
            query: A natural-language question or search phrase about the
                contents of the contracts (e.g. "termination notice period
                for the vendor agreement", "liability cap amount").

        Returns the matching passages as citable text, or an explicit
        'no relevant documents found' message if nothing matches -- treat
        that as a real answer, not a reason to guess.
        """
        results = retriever.invoke(query)
        return format_docs(results)

    return search_documents


def build_agent(retriever, checkpointer=None):
    from langchain.agents import create_agent

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    from src.tools import check_deadline_urgency, draft_renewal_email

    search_documents = build_search_tool(retriever)

    return create_agent(
        model=get_chat_model(),
        tools=[search_documents, check_deadline_urgency, draft_renewal_email],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
