"""read_document: the agent's tool for reading an uploaded income document.

The agent names an income document slot (pay stub, W-2 or tax return) and
gets back what the mock extraction service found: what the document really
is, whether it is in the right slot, whether it could be read, its allowed
fields with identifiers removed, and values calculated in code.

Below the findings it adds the results of the bank's income rules (current
document, within 5% of the claimed income), from the same code that
mark_income_verified uses to refuse. The agent never compares numbers itself:
it reads "not met: too old" and decides what to do next. The rules are the
bank's, not the vendor's, so they are added here, not in the extraction
service.

ID documents cannot be read with this tool: everything on an ID is an
identifier, and the agent only ever sees the findings from
verify_identity.

The store, application_id and now are injected by the system and are not
part of the tool's parameters that the LLM sees; the LLM only chooses the
slot. Text in a prompt, including an ID injected by an attacker, therefore
cannot change which application is read: the tool only ever works on the
current application (protection against cross-tenant access). The LLM is not
given the application ID at all; it has no use for it. The time is supplied
by the system because time is simulated in the evals. Each read is recorded
in the audit log (the slot and whether the rules were met); an attempt to read
an ID is recorded as blocked.
"""

from datetime import datetime
from typing import get_args

from loans.checks.income_verification import check_income_document
from loans.schemas.documents import IncomeSlot, UploadSlot
from loans.services.extraction import extract_document
from loans.store import ApplicationStore


def read_document(
    store: ApplicationStore, application_id: str, slot: str, now: datetime
) -> str:
    """Read the document in one income document slot and return the result as
    text for the agent."""
    if slot not in get_args(IncomeSlot):
        tried = slot if slot in get_args(UploadSlot) else "an unknown slot"
        store.log_event(
            application_id, now, "guardrail", "action_blocked", f"read_document: {tried}"
        )
        return (
            f"slot: {slot}\n"
            "cannot be read with this tool: only pay_stub, w2 and tax_return can be "
            "read. ID documents are checked with verify_identity."
        )
    stored = store.get(application_id)
    document = stored.documents.get(slot)
    if document is None:
        store.log_event(application_id, now, "agent", "document_read", f"{slot}: no file")
        return f"slot: {slot}\nno file uploaded in this slot"
    findings = extract_document(document, slot, now.date()).to_llm_text()
    rules = check_income_document(document, slot, stored.raw, now.date())
    met = "met" if rules.met else "not met"
    store.log_event(
        application_id, now, "agent", "document_read", f"{slot}: bank income rules {met}"
    )
    return f"{findings}\n{rules.to_llm_text()}"
