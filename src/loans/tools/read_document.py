"""read_document: the agent's tool for reading an uploaded income document.

The agent names an income document slot (pay stub, W-2 or tax return) and
gets back what the mock extraction service found: what the document really
is, whether it is in the right slot, whether it could be read, its allowed
fields with identifiers removed, and values calculated in code.

ID documents cannot be read with this tool: everything on an ID is an
identifier, and the agent only ever sees the findings from
verify_identity.

The store, application_id and today are injected by the system and are not
part of the tool's parameters that the LLM sees; the LLM only chooses the
slot. Text in a prompt, including an ID injected by an attacker, therefore
cannot change which application is read: the tool only ever works on the
current application (protection against cross-tenant access). The LLM is not
given the application ID at all; it has no use for it. The date is supplied
by the system because time is simulated in the evals.
"""

from datetime import date
from typing import Literal, get_args

from loans.services.extraction import extract_document
from loans.store import ApplicationStore

IncomeSlot = Literal["pay_stub", "w2", "tax_return"]


def read_document(
    store: ApplicationStore, application_id: str, slot: str, today: date
) -> str:
    """Read the document in one income document slot and return the result as
    text for the agent."""
    if slot not in get_args(IncomeSlot):
        return (
            f"slot: {slot}\n"
            "cannot be read with this tool: only pay_stub, w2 and tax_return can be "
            "read. ID documents are checked with verify_identity."
        )
    document = store.get(application_id).documents.get(slot)
    if document is None:
        return f"slot: {slot}\nno file uploaded in this slot"
    return extract_document(document, slot, today).to_llm_text()
