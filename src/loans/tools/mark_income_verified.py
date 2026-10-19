"""mark_income_verified: the agent's tool for recording that a document
verifies the claimed income.

The agent decides: it chooses the document (slot) and states its reason, for
example why the employer on the pay stub matches the form or how the
customer's explanation accounts for a difference. That judgment is the
agent's; the reason is kept on file and shown to the human approver.

Frame rule, enforced in code: the bank's income rules (the document fits the
employment status, is current, and is within 5% of the claimed income) must
be met. The tool applies the same rules that read_document shows, and refuses
if one is not met, whatever the agent was told or believes. The verification
is kept with the document version and form version it refers to, so it goes
out of date after a new upload or a form correction, and finish can tell.

Idempotent: a retried tool call (same call ID) does not record the
verification again; it returns the same confirmation.

The store, application_id, now and call_id are injected by the system and
are not part of the tool's parameters that the LLM sees; the LLM chooses only
the slot and the reason. Text in a prompt, including an ID injected by an
attacker, therefore cannot change which application is marked: the tool only
ever works on the current application (protection against cross-tenant
access). The time is supplied by the system because time is simulated in the
evals.

Each verification is recorded in the audit log with the agent's reason; a
refused one is recorded as blocked, with the rules that were not met.
"""

from datetime import datetime
from typing import get_args

from loans.checks.income_verification import check_income_document
from loans.schemas.documents import IncomeSlot
from loans.store import ApplicationStore, IncomeVerification, StoredApplication


def mark_income_verified(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    call_id: str,
    slot: str,
    reason: str,
) -> str:
    """Record that the document in one income slot verifies the claimed income."""
    stored = store.get(application_id)

    earlier = stored.income_verification
    if earlier is not None and earlier.call_id == call_id:
        return f"income verified with the {earlier.slot} (kept on file)"

    problem = _problem(stored, slot, reason, now)
    if problem:
        store.log_event(
            application_id,
            now,
            "guardrail",
            "action_blocked",
            "mark_income_verified: " + "; ".join(problem.splitlines()),
            reason or None,
        )
        separator = "\n" if "\n" in problem else " "
        return f"not verified:{separator}{problem}"

    store.record_income_verification(
        application_id,
        IncomeVerification(
            call_id=call_id,
            slot=slot,
            reason=reason,
            document_version=stored.document_versions[slot],
            form_version=stored.form_version,
            verified_on=now.date(),
        ),
    )
    store.log_event(application_id, now, "agent", "income_verified", slot, reason)
    return f"income verified with the {slot} (kept on file)"


def _problem(stored: StoredApplication, slot: str, reason: str, now: datetime) -> str | None:
    if slot not in get_args(IncomeSlot):
        return "only pay_stub, w2 and tax_return can verify income"
    if not reason.strip():
        return "a reason is required"
    document = stored.documents.get(slot)
    if document is None:
        return f"no file uploaded in the {slot} slot"
    rules = check_income_document(document, slot, stored.raw, now.date())
    if not rules.met:
        return rules.to_llm_text()
    return None
