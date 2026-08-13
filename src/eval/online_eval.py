"""
Bonus: a basic online evaluator. Scores real queries as they're handled,
logging feedback directly onto the trace in LangSmith -- separate from the
offline golden-set experiments in run_eval.py. This is "is it still
healthy right now" monitoring, not "did this change improve things"
regression testing; both matter, but they answer different questions.

Usage: swap src.cli's plain agent.invoke() call for
monitored_agent_query() below to get live faithfulness scores on real
usage, without changing anything else about the CLI's behavior.
"""

import uuid

# Explicit, not relied-upon-via-import-order -- same reasoning as
# src/eval/dataset.py, since this module instantiates Client() directly.
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.run_helpers import get_current_run_tree

from src.eval.evaluators import faithfulness_evaluator
from src.rag_chain import format_docs


def monitored_agent_query(agent, retriever, question: str, thread_id: str | None = None) -> str:
    """Same shape as a normal agent call (question in, answer out), but
    additionally retrieves context, runs the faithfulness evaluator
    against this specific real interaction, and logs the result as
    feedback on the trace -- so a drop in live faithfulness shows up in
    the LangSmith UI without waiting for the next scheduled eval run."""
    thread_id = thread_id or str(uuid.uuid4())

    retrieved = retriever.invoke(question)
    context = format_docs(retrieved)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = result["messages"][-1].content

    _log_online_feedback(answer=answer, context=context)
    return answer


def _log_online_feedback(answer: str, context: str) -> None:
    """Fires the faithfulness evaluator against this one interaction and
    attaches the score as feedback on the currently-running trace. Fails
    silently (logs a warning, doesn't raise) -- a monitoring hiccup should
    never take down the actual user-facing query."""
    try:
        run_tree = get_current_run_tree()
        if run_tree is None:
            # No active trace (e.g. tracing not enabled this session) --
            # nothing to attach feedback to.
            return

        from types import SimpleNamespace

        fake_run = SimpleNamespace(outputs={"answer": answer, "context": context})
        fake_example = SimpleNamespace(outputs={})
        grade = faithfulness_evaluator(fake_run, fake_example)

        client = Client()
        client.create_feedback(
            run_id=run_tree.id,
            key="faithfulness_live",
            score=grade["score"],
            comment=grade.get("comment"),
        )
    except Exception as e:
        print(f"  [WARN] online faithfulness monitoring failed (non-fatal): {e}")
