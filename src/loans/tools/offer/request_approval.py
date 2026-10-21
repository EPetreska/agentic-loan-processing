"""request_approval: the offer agent's tool for asking a loan officer.

The bank makes the customer one offer, the amount and term they asked for.
Any change the customer asks for (more or less money, another term, a lower
rate, a later first payment) is not the agent's to decide: the agent
describes the request and why it asks, and a loan officer decides. The
request is kept on file as pending, and the case waits for the decision; the
agent passes it on to the customer once it arrives. If the officer approves,
code makes the revised offer; the agent never calculates one.

One request at a time: while one is waiting, another is refused, which keeps
the wait simple. Also refused before the case has an offer, once the offer
phase has closed, and without a request or a reason, or with a request
longer than MAX_REQUEST_LENGTH; refusals are logged as blocked.

Idempotent: a retried tool call (same call ID) does not ask again; it
returns the same confirmation.

As for every agent tool, the store, application_id, now and call_id are
injected by the system; the LLM chooses only the request and the reason.
"""

from datetime import datetime

from loans.store import ApplicationStore, ApprovalRequest

# A limit on the request's length: a short description, not a letter.
MAX_REQUEST_LENGTH = 1000


def request_approval(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    call_id: str,
    request: str,
    reason: str,
) -> str:
    """Ask a loan officer about the customer's request; return a confirmation."""
    stored = store.get(application_id)

    for number, earlier in enumerate(stored.approval_requests, start=1):
        if earlier.request_id == call_id:
            return _confirmation(number)
    if refusal := store.refuse_if_offer_closed(application_id, now, "request_approval"):
        return refusal

    problem = _problem(stored.pending_approval() is not None, request, reason)
    if problem:
        store.log_event(
            application_id, now, "guardrail", "action_blocked", f"request_approval: {problem}"
        )
        return f"not sent: {problem}"

    # Counted from the case as read above, before the request is recorded:
    # in memory that case is the stored object itself and would change.
    number = len(stored.approval_requests) + 1
    store.record_approval_request(
        application_id,
        ApprovalRequest(request_id=call_id, requested_at=now, request=request, reason=reason),
    )
    return _confirmation(number)


def _confirmation(number: int) -> str:
    return (
        f"sent to a loan officer (request {number}); the case waits for the "
        "officer's decision"
    )


def _problem(pending: bool, request: str, reason: str) -> str | None:
    if pending:
        return "a request is already waiting for a loan officer's decision"
    if not request.strip():
        return "the request is empty"
    if not reason.strip():
        return "a reason is required"
    if len(request) > MAX_REQUEST_LENGTH:
        return f"the request is longer than {MAX_REQUEST_LENGTH} characters"
    return None
