"""verify_identity: the agent's tool for checking the applicant's ID.

The mock identity vendor compares the uploaded ID with the form and reports
only its findings (document type, readable, expired, name and date of birth
match). There is no verdict: the agent combines the findings with the bank's
requirements, which are part of the frame (in the sense of the Agentic BPM
manifesto: the rules the agent works within, stated in its instructions and
enforced in code by finish), and decides what to do. The findings are kept on
file in the store, as a bank would keep a vendor's answer, so finish can later
check them without redoing the comparison.

The agent never sees the ID itself: no name, date of birth or ID number is
returned. The store, application_id and now are injected by the system and
are not part of the tool's parameters that the LLM sees; the tool has no
parameters the LLM chooses. Text in a prompt, including an ID injected by an
attacker, therefore cannot change which application is checked: the tool only
ever works on the current application (protection against cross-tenant
access). The LLM is not given the application ID at all; it has no use for it.
The time is supplied by the system because time is simulated in the evals.
Each check is recorded in the audit log with the findings (yes or no only).

Once the case has ended, the tool refuses and the attempt is logged as
blocked.
"""

from datetime import datetime

from loans.services.identity import check_identity
from loans.store import ApplicationStore


def verify_identity(store: ApplicationStore, application_id: str, now: datetime) -> str:
    """Check the current application's ID and return the findings as text."""
    if refusal := store.refuse_if_ended(application_id, now, "verify_identity"):
        return refusal
    stored = store.get(application_id)
    document = stored.documents.get("id_front")
    if document is None:
        store.log_event(application_id, now, "agent", "identity_checked", "no ID uploaded")
        return "no ID uploaded yet (the ID front slot is empty)"
    findings = check_identity(
        document,
        stored.raw,
        id_front_version=stored.document_versions["id_front"],
        form_version=stored.form_version,
        today=now.date(),
    )
    store.record_identity_findings(application_id, findings)
    text = findings.to_llm_text()
    store.log_event(application_id, now, "agent", "identity_checked", "; ".join(text.splitlines()))
    return text
