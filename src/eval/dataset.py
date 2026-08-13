"""
Golden dataset creation for LangSmith evaluation.

Two datasets, because they test two different pipelines:

- clauseiq-golden-set: question/answer pairs for the RAG agent (tests
  retrieval + generation). Used by retrieval_hit_evaluator and
  faithfulness_evaluator.
- clauseiq-extraction-set: one row per sample contract with the actual
  ground-truth field values (this project's author wrote all 6 sample
  contracts, so these are exact, not guessed). Used by
  extraction_field_evaluator to test src.extraction.extract_summary
  directly -- a different pipeline than RAG Q&A, so it needs its own
  ground truth rather than being folded into the QA set.

Run directly to create/update both datasets: python -m src.eval.dataset
Idempotent -- re-running won't duplicate an existing dataset by name, but
note create_examples() itself will add duplicate *examples* on a second
run against an existing dataset. If you need to rebuild from scratch,
delete the dataset in the LangSmith UI first.
"""

# Loaded explicitly here (not just via importing src.config) because this
# module calls langsmith.Client() directly, and relying on some other
# module having already imported src.config first is fragile -- exactly
# the bug that caused a 401 "Invalid token" here even with a correct key
# in .env: nothing had loaded it into the environment yet.
from dotenv import load_dotenv

load_dotenv()

QA_EXAMPLES = [
    # --- residential_lease.docx ---------------------------------------
    {
        "inputs": {"question": "What is the monthly rent in the residential lease?"},
        "outputs": {"answer": "$2,150.00 per month", "expected_source": "residential_lease.docx"},
    },
    {
        "inputs": {"question": "How much notice does the tenant need to give to not renew the residential lease?"},
        "outputs": {"answer": "60 days written notice prior to expiration of the current term", "expected_source": "residential_lease.docx"},
    },
    {
        "inputs": {"question": "What is the security deposit on the residential lease and when is it refunded?"},
        "outputs": {"answer": "$2,150.00, refundable within 30 days of move-out less damage deductions", "expected_source": "residential_lease.docx"},
    },
    {
        "inputs": {"question": "What pet policy applies to the residential lease?"},
        "outputs": {"answer": "One pet under 40 lbs allowed with a non-refundable $300 pet fee; aggressive breeds prohibited", "expected_source": "residential_lease.docx"},
    },
    {
        "inputs": {"question": "By how much can the landlord raise rent at renewal on the residential lease?"},
        "outputs": {"answer": "Up to 8% per renewal term, with 45 days written notice", "expected_source": "residential_lease.docx"},
    },
    {
        "inputs": {"question": "Who are the parties in the residential lease?"},
        "outputs": {"answer": "Meridian Properties LLC (Landlord) and Sarah Chen (Tenant)", "expected_source": "residential_lease.docx"},
    },
    # --- vendor_service_agreement.pdf ---------------------------------
    {
        "inputs": {"question": "What is the termination notice period on the vendor service agreement?"},
        "outputs": {"answer": "30 days written notice of non-renewal, or 90 days for termination for convenience", "expected_source": "vendor_service_agreement.pdf"},
    },
    {
        "inputs": {"question": "What is the liability cap in the vendor service agreement, and why is it flagged as a risk?"},
        "outputs": {"answer": "$5,000 per incident, flagged as unusually low relative to potential losses from a service failure", "expected_source": "vendor_service_agreement.pdf"},
    },
    {
        "inputs": {"question": "What are the payment terms in the vendor service agreement?"},
        "outputs": {"answer": "$8,400.00 per month, invoiced monthly in arrears, net 30", "expected_source": "vendor_service_agreement.pdf"},
    },
    {
        "inputs": {"question": "What on-time delivery rate does the vendor guarantee?"},
        "outputs": {"answer": "98% on-time delivery, with a 10% invoice credit if missed in a given month", "expected_source": "vendor_service_agreement.pdf"},
    },
    {
        "inputs": {"question": "Who are the parties in the vendor service agreement?"},
        "outputs": {"answer": "Harborview Hotel Group (Client) and CrispClean Linen Services Inc. (Vendor)", "expected_source": "vendor_service_agreement.pdf"},
    },
    {
        "inputs": {"question": "What law governs the vendor service agreement?"},
        "outputs": {"answer": "The laws of the State of Florida", "expected_source": "vendor_service_agreement.pdf"},
    },
    # --- mutual_nda.pdf -------------------------------------------------
    {
        "inputs": {"question": "Who are the parties in the NDA?"},
        "outputs": {"answer": "Northgate Analytics Inc. (Party A) and BrightPath Consulting LLC (Party B)", "expected_source": "mutual_nda.pdf"},
    },
    {
        "inputs": {"question": "How long do confidentiality obligations survive after the NDA terminates?"},
        "outputs": {"answer": "Three (3) years after termination", "expected_source": "mutual_nda.pdf"},
    },
    {
        "inputs": {"question": "What is the term length of the mutual NDA?"},
        "outputs": {"answer": "Two (2) years from the effective date", "expected_source": "mutual_nda.pdf"},
    },
    {
        "inputs": {"question": "What notice is required to terminate the NDA?"},
        "outputs": {"answer": "Thirty (30) days written notice", "expected_source": "mutual_nda.pdf"},
    },
    {
        "inputs": {"question": "Why is the NDA's survival clause flagged as a risk?"},
        "outputs": {"answer": "The 3-year survival period is longer than the typical 1-2 year industry standard for mutual NDAs", "expected_source": "mutual_nda.pdf"},
    },
    # --- dental_insurance_policy.pdf ------------------------------------
    {
        "inputs": {"question": "What is the annual maximum benefit on the dental insurance policy?"},
        "outputs": {"answer": "$2,000 per covered individual per year", "expected_source": "dental_insurance_policy.pdf"},
    },
    {
        "inputs": {"question": "What is the waiting period for major dental procedures?"},
        "outputs": {"answer": "Twelve (12) months from the effective date of coverage, except for accidental injury", "expected_source": "dental_insurance_policy.pdf"},
    },
    {
        "inputs": {"question": "What is the monthly premium on the dental insurance policy?"},
        "outputs": {"answer": "$46.50 per covered employee per month", "expected_source": "dental_insurance_policy.pdf"},
    },
    {
        "inputs": {"question": "What is the deductible on the dental insurance policy?"},
        "outputs": {"answer": "$50 per individual, $150 family maximum, not applied to preventive care", "expected_source": "dental_insurance_policy.pdf"},
    },
    {
        "inputs": {"question": "What is the policy number of the dental insurance policy?"},
        "outputs": {"answer": "GDT-2026-04471", "expected_source": "dental_insurance_policy.pdf"},
    },
    {
        # Deliberately "should say not specified" -- the dental policy never
        # states a termination/renewal notice period (see src/extraction.py's
        # ContractSummary.renewal_or_termination_notice_days = None for this doc).
        "inputs": {"question": "What termination notice period applies to the dental insurance policy?"},
        "outputs": {"answer": "Not specified in the provided document", "expected_source": None},
    },
    # --- ecommerce_supplier_terms.pdf -----------------------------------
    {
        "inputs": {"question": "What notice is required to non-renew the e-commerce supplier terms agreement?"},
        "outputs": {"answer": "At least ninety (90) days written notice before the end of the current term", "expected_source": "ecommerce_supplier_terms.pdf"},
    },
    {
        "inputs": {"question": "What are the payment terms in the e-commerce supplier terms agreement?"},
        "outputs": {"answer": "Net 45 from the invoice date", "expected_source": "ecommerce_supplier_terms.pdf"},
    },
    {
        "inputs": {"question": "How can the buyer charge back defective goods under the supplier terms agreement?"},
        "outputs": {"answer": "Full invoice value plus a 15% handling fee, without prior authorization from the supplier", "expected_source": "ecommerce_supplier_terms.pdf"},
    },
    {
        "inputs": {"question": "Who owns the product designs under the e-commerce supplier terms agreement?"},
        "outputs": {"answer": "Buyer (Northline Commerce Inc.) retains exclusive ownership of all product designs it provides", "expected_source": "ecommerce_supplier_terms.pdf"},
    },
    {
        "inputs": {"question": "What is the initial term length of the e-commerce supplier terms agreement?"},
        "outputs": {"answer": "Eighteen (18) months from the effective date, ending November 30, 2027", "expected_source": "ecommerce_supplier_terms.pdf"},
    },
    {
        "inputs": {"question": "Who are the parties in the e-commerce supplier terms agreement?"},
        "outputs": {"answer": "Northline Commerce Inc. d/b/a Kettlewell Goods (Buyer) and Pinecrest Manufacturing Co. (Supplier)", "expected_source": "ecommerce_supplier_terms.pdf"},
    },
    # --- garbled_scan_office_lease.pdf (robustness case) -----------------
    {
        "inputs": {"question": "What is the monthly base rent in the office lease with CedarRidge?"},
        "outputs": {"answer": "$9,800.00 per month base rent, plus an estimated $1,400.00/month operating expense pass-through", "expected_source": "garbled_scan_office_lease.pdf"},
    },
    {
        "inputs": {"question": "How much notice does the tenant need to renew the CedarRidge office lease?"},
        "outputs": {"answer": "At least six (6) months prior to expiration of the initial term; no automatic renewal applies", "expected_source": "garbled_scan_office_lease.pdf"},
    },
    {
        "inputs": {"question": "Why is the operating expense clause in the CedarRidge office lease flagged as a risk?"},
        "outputs": {"answer": "The operating expense pass-through is only an estimate and is not capped, so actual monthly cost could rise substantially", "expected_source": "garbled_scan_office_lease.pdf"},
    },
    {
        "inputs": {"question": "What is the square footage of the premises in the CedarRidge office lease?"},
        "outputs": {"answer": "Approximately 4,200 square feet", "expected_source": "garbled_scan_office_lease.pdf"},
    },
    # --- Cross-document / genuinely out-of-scope (no matching source) ----
    {
        "inputs": {"question": "What is the office parking policy?"},
        "outputs": {"answer": "Not specified in the provided documents", "expected_source": None},
    },
    {
        "inputs": {"question": "What is the company's remote work policy?"},
        "outputs": {"answer": "Not specified in the provided documents", "expected_source": None},
    },
    {
        "inputs": {"question": "What is the CEO's salary?"},
        "outputs": {"answer": "Not specified in the provided documents", "expected_source": None},
    },
    {
        "inputs": {"question": "What warranty period applies to the ceramic kitchenware products themselves?"},
        # The supplier terms agreement covers quality/chargebacks but never
        # states a product warranty period -- a "should say not specified"
        # case that's easy to mistake for the chargeback clause if the
        # system isn't being careful.
        "outputs": {"answer": "Not specified in the provided documents", "expected_source": None},
    },
]

# One row per sample contract, ground-truthed against what was actually
# written into each document (see data/contracts_txt/ source text).
EXTRACTION_EXAMPLES = [
    {
        "inputs": {"filename": "residential_lease.docx"},
        "outputs": {
            "parties": ["Meridian Properties LLC", "Sarah Chen"],
            "effective_date": "2026-03-01",
            "renewal_or_termination_notice_days": 60,
        },
    },
    {
        "inputs": {"filename": "vendor_service_agreement.pdf"},
        "outputs": {
            "parties": ["Harborview Hotel Group", "CrispClean Linen Services Inc."],
            "effective_date": "2025-09-01",
            "renewal_or_termination_notice_days": 30,
        },
    },
    {
        "inputs": {"filename": "mutual_nda.pdf"},
        "outputs": {
            "parties": ["Northgate Analytics Inc.", "BrightPath Consulting LLC"],
            "effective_date": "2026-01-15",
            "renewal_or_termination_notice_days": 30,
        },
    },
    {
        "inputs": {"filename": "dental_insurance_policy.pdf"},
        "outputs": {
            "parties": ["Riverside Family Dental Associates", "Guardian Trust Dental Insurance Co."],
            "effective_date": "2026-04-01",
            # Deliberately None -- this document never states one. This row
            # is what proves extraction_field_evaluator actually checks
            # correctness rather than just "did a value come back."
            "renewal_or_termination_notice_days": None,
        },
    },
    {
        "inputs": {"filename": "ecommerce_supplier_terms.pdf"},
        "outputs": {
            "parties": ["Northline Commerce Inc.", "Pinecrest Manufacturing Co."],
            "effective_date": "2026-06-01",
            "renewal_or_termination_notice_days": 90,
        },
    },
    {
        "inputs": {"filename": "garbled_scan_office_lease.pdf"},
        "outputs": {
            "parties": ["CedarRidge Commercial Properties LLC", "Vertex Dynamics Inc."],
            "effective_date": "2025-07-01",
            "renewal_or_termination_notice_days": 180,  # 6 months
        },
    },
]

QA_DATASET_NAME = "clauseiq-golden-set"
EXTRACTION_DATASET_NAME = "clauseiq-extraction-set"


def get_or_create_dataset(client, name: str, description: str = ""):
    for ds in client.list_datasets():
        if ds.name == name:
            return ds
    return client.create_dataset(dataset_name=name, description=description)


def _dataset_is_populated(client, dataset_id) -> bool:
    """True if the dataset already has at least one example. Used to make
    build_qa_dataset/build_extraction_dataset genuinely idempotent --
    client.create_examples() itself has no dedup logic and will happily
    append a full duplicate copy of every example on every call, which is
    exactly what happened running this repeatedly (37 examples x 3 runs
    = 111 in the dataset). Checking first means re-running
    `python -m src.eval.run_eval` any number of times is now safe."""
    for _ in client.list_examples(dataset_id=dataset_id, limit=1):
        return True
    return False


def build_qa_dataset(client=None, force: bool = False):
    from langsmith import Client

    client = client or Client()
    dataset = get_or_create_dataset(
        client,
        QA_DATASET_NAME,
        description=(
            "Golden Q&A set for ClauseIQ RAG agent evaluation -- 37 examples "
            "across all 6 sample contracts, including 'should say not "
            "specified' cases."
        ),
    )
    if not force and _dataset_is_populated(client, dataset.id):
        print(
            f"QA dataset '{QA_DATASET_NAME}' already has examples -- skipping "
            f"(pass force=True to add anyway, e.g. after deliberately "
            f"editing QA_EXAMPLES)"
        )
        return dataset

    client.create_examples(
        inputs=[e["inputs"] for e in QA_EXAMPLES],
        outputs=[e["outputs"] for e in QA_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"QA dataset '{QA_DATASET_NAME}': {len(QA_EXAMPLES)} examples added")
    return dataset


def build_extraction_dataset(client=None, force: bool = False):
    from langsmith import Client

    client = client or Client()
    dataset = get_or_create_dataset(
        client,
        EXTRACTION_DATASET_NAME,
        description=(
            "Ground-truth structured-extraction values for each of the 6 "
            "sample contracts -- tests src.extraction.extract_summary "
            "directly, including a document with a genuinely null field."
        ),
    )
    if not force and _dataset_is_populated(client, dataset.id):
        print(
            f"Extraction dataset '{EXTRACTION_DATASET_NAME}' already has "
            f"examples -- skipping (pass force=True to add anyway)"
        )
        return dataset

    client.create_examples(
        inputs=[e["inputs"] for e in EXTRACTION_EXAMPLES],
        outputs=[e["outputs"] for e in EXTRACTION_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"Extraction dataset '{EXTRACTION_DATASET_NAME}': {len(EXTRACTION_EXAMPLES)} examples added")
    return dataset


if __name__ == "__main__":
    build_qa_dataset()
    build_extraction_dataset()
