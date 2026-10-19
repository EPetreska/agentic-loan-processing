"""verify_identity: the agent's tool for checking the applicant's ID.

The mock identity vendor compares the uploaded ID with the form and reports
only its findings (document type, readable, expired, name and date of birth
match). There is no verdict: the agent combines the findings with the bank's
requirements, which are part of its instructions, and decides what to do.
The findings are kept on file in the store, as a bank would keep a vendor's
answer, so finish can later check them without redoing the comparison.

The agent never sees the ID itself: no name, date of birth or ID number is
returned. When registered as an agent tool, the store, application_id and
today are injected by the system and are not part of the tool's parameters
that the LLM sees. Text in a prompt, including an ID injected by an attacker,
therefore cannot change which application is checked: the tool only ever
works on the current application (protection against cross-tenant access).
The LLM is not given the application ID at all; it has no use for it. The
date is supplied by the system because time is simulated in the evals.
"""

from datetime import date

from loans.services.identity import check_identity
from loans.store import ApplicationStore


def verify_identity(store: ApplicationStore, application_id: str, today: date) -> str:
    """Check the current application's ID and return the findings as text."""
    stored = store.get(application_id)
    document = stored.documents.get("id_front")
    if document is None:
        return "no ID uploaded yet (the ID front slot is empty)"
    findings = check_identity(
        document,
        stored.raw,
        id_front_version=stored.document_versions["id_front"],
        form_version=stored.form_version,
        today=today,
    )
    store.record_identity_findings(application_id, findings)
    return findings.to_llm_text()
