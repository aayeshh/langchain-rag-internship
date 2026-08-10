"""
Agent tools. Per the brief: one tool that calculates days-remaining and
flags urgency on a renewal/termination deadline, and one tool that drafts a
short follow-up email about an upcoming renewal or a flagged risk.

Docstrings here are load-bearing -- the model reads them to decide whether
and how to call each tool, per the coding guide's #1 intern mistake to
avoid ("vague tool docstrings").
"""

from datetime import date

from langchain_core.tools import tool


@tool
def check_deadline_urgency(target_date: str) -> str:
    """Calculate how many days remain until a contract renewal or
    termination deadline, and classify its urgency.

    Args:
        target_date: The deadline date in YYYY-MM-DD format (e.g. the
            renewal or termination-notice date found in a contract).

    Returns a string with the days remaining and one of three urgency
    labels: OVERDUE (deadline already passed), URGENT (fewer than 30 days
    remain), or OK (30+ days remain). Always use this tool instead of
    estimating dates yourself -- date arithmetic must be exact.
    """
    try:
        d = date.fromisoformat(target_date)
    except ValueError:
        return (
            f"Could not parse '{target_date}' as a date. Expected format: "
            "YYYY-MM-DD."
        )

    days_remaining = (d - date.today()).days
    if days_remaining < 0:
        status = "OVERDUE"
    elif days_remaining < 30:
        status = "URGENT"
    else:
        status = "OK"

    return f"{days_remaining} day(s) remaining until {target_date}. Status: {status}."


@tool
def draft_renewal_email(
    recipient_or_party: str,
    contract_name: str,
    context: str,
) -> str:
    """Draft a short, professional follow-up email about an upcoming
    contract renewal deadline or a flagged risk clause, ready to send or
    lightly edit.

    Args:
        recipient_or_party: Who the email is to/about (e.g. a vendor name,
            counterparty, or internal stakeholder).
        contract_name: The contract or document this email concerns (e.g.
            the source filename or a short contract description).
        context: The specific situation to write about -- e.g. "renewal
            deadline is 2026-09-01, 23 days remaining, URGENT" or "flagged
            risk: liability cap of $5,000 per incident is unusually low".

    Returns the drafted email as plain text (subject line + body).
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from src.config import get_chat_model

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You draft short, professional business emails about "
                "contract renewals and risk flags. Keep it under 150 "
                "words, one clear ask, no filler. Output a Subject line "
                "followed by the email body.",
            ),
            (
                "human",
                "Recipient/party: {recipient_or_party}\n"
                "Contract: {contract_name}\n"
                "Situation: {context}\n\n"
                "Draft the email.",
            ),
        ]
    )
    chain = prompt | get_chat_model(temperature=0.3) | StrOutputParser()
    return chain.invoke(
        {
            "recipient_or_party": recipient_or_party,
            "contract_name": contract_name,
            "context": context,
        }
    )
