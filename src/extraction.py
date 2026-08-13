"""
Structured extraction: for every ingested document, pull a typed summary
(parties, effective date, renewal/termination notice period, payment terms,
key risk) via .with_structured_output(). This is the piece that makes
ClauseIQ feel like a product rather than a chatbot -- reliable, typed data,
not a paragraph you have to re-parse.

Summaries are cached to data/summaries/<source>.json, keyed off the same
content-hash manifest ingest.py uses, so re-running ingestion doesn't
re-extract (and re-pay for) documents that haven't changed.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.config import BASE_DIR

SUMMARIES_DIR = BASE_DIR / "data" / "summaries"


class ContractSummary(BaseModel):
    """Typed extraction target. Optional fields default to None instead of
    forcing the model to invent a value for a document missing that
    field (e.g. the dental policy sample has no explicit termination
    notice period) -- required by the brief's robustness criterion."""

    parties: list[str] = Field(description="Names of the contracting parties")
    effective_date: Optional[str] = Field(
        default=None, description="ISO date the contract starts, if stated"
    )
    renewal_or_termination_notice_days: Optional[int] = Field(
        default=None,
        description=(
            "Number of days' written notice required to renew or terminate "
            "the agreement, if stated. Use the shorter/more restrictive "
            "value if multiple notice periods are mentioned."
        ),
    )
    payment_terms: Optional[str] = Field(
        default=None,
        description="A one-sentence summary of payment amount/frequency/terms, if stated",
    )
    key_risk: str = Field(
        description=(
            "The single biggest risk, unusual clause, or exposure for the "
            "receiving party, in one sentence. Always populate this even "
            "if other fields are missing."
        )
    )
    source: str = Field(description="Source filename this summary was extracted from")


def _combine_pages(docs) -> str:
    """docs: the raw (unsplit) Documents for a single file, as returned by
    a loader (one per page for PDFs). Joined back into one string for
    extraction, since a contract's parties/dates/risk usually need
    whole-document context, not a single chunk."""
    return "\n\n".join(d.page_content for d in docs)


def extract_summary(source_filename: str, full_text: str, model=None) -> ContractSummary:
    from langsmith import traceable

    @traceable(name="extract_contract_summary", run_type="chain")
    def _extract() -> ContractSummary:
        nonlocal model
        if model is None:
            from src.config import get_chat_model

            model = get_chat_model()

        structured_model = model.with_structured_output(ContractSummary)
        result = structured_model.invoke(
            "Extract the key details from this contract. If a field isn't "
            "stated in the document, leave it unset rather than guessing.\n\n"
            f"Document filename: {source_filename}\n\n"
            f"Contract text:\n{full_text}"
        )
        # Belt-and-suspenders: force the source field to the real filename
        # rather than trusting the model to copy it verbatim.
        result.source = source_filename
        return result

    return _extract()


def save_summary(summary: ContractSummary) -> Path:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARIES_DIR / f"{Path(summary.source).stem}.json"
    out_path.write_text(summary.model_dump_json(indent=2))
    return out_path


def load_cached_summary(source_filename: str) -> Optional[ContractSummary]:
    path = SUMMARIES_DIR / f"{Path(source_filename).stem}.json"
    if not path.exists():
        return None
    return ContractSummary.model_validate(json.loads(path.read_text()))


def extract_and_cache_for_files(files_with_docs: dict[str, list]) -> list[ContractSummary]:
    """files_with_docs: {source_filename: [Document, ...]} for files that
    were newly ingested this run (from src.ingest's changed-file set).
    Only extracts for files without a cached summary already -- same
    re-work guard the vector store gets."""
    summaries = []
    for filename, docs in files_with_docs.items():
        cached = load_cached_summary(filename)
        if cached is not None:
            summaries.append(cached)
            continue
        full_text = _combine_pages(docs)
        summary = extract_summary(filename, full_text)
        save_summary(summary)
        summaries.append(summary)
    return summaries
