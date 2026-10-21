"""send_message: the offer agent's tool for writing to the customer, to
answer questions and to pass on a loan officer's decision.

The rules for writing to the customer are shared with the application agent
and live in loans.tools.messages. This tool adds its own limits: it refuses
before the case has an offer, and once the offer phase has closed (the
customer signed, or the offer expired); the attempt is logged as blocked. The
offer itself is sent by the bank's workflow with its disclosures, never
written by the agent.

As for every agent tool, the store, application_id, now and call_id are
injected by the system; the LLM chooses only the text and the reason.
"""

from datetime import datetime

from loans.store import ApplicationStore
from loans.tools.messages import send


def send_message(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    call_id: str,
    text: str,
    reason: str,
) -> str:
    """Send a message to the current customer and return a confirmation."""
    return send(
        store, application_id, now, call_id, text, reason,
        refusal=lambda: store.refuse_if_offer_closed(application_id, now, "send_message"),
    )  # fmt: skip
