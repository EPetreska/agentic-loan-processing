"""send_message: the application agent's tool for writing to the customer.

The rules for writing to the customer (the agent chooses only the text and
the reason, e-sign consent, length, a reason, no duplicate on retry, the
audit log) are shared with the offer agent and live in loans.tools.messages.
This tool adds its own limit: once the case has ended, it refuses and the
attempt is logged as blocked.

The store, application_id, now and call_id are injected by the system and are
not part of the tool's parameters that the LLM sees; the LLM chooses only the
text and the reason. Text in a prompt, including an ID injected by an
attacker, therefore cannot change which customer receives the message: the
tool only ever works on the current application (protection against
cross-tenant access). The time is supplied by the system because time is
simulated in the evals, and the deadlines count from the last message.
"""

from datetime import datetime

from loans.store import ApplicationStore
from loans.tools.messages import MAX_MESSAGE_LENGTH, send

__all__ = ["MAX_MESSAGE_LENGTH", "send_message"]


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
        refusal=lambda: store.refuse_if_ended(application_id, now, "send_message"),
    )  # fmt: skip
