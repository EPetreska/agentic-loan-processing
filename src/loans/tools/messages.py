"""Writing to the customer: the rules both agents' send_message follow.

The application agent and the offer agent each have their own send_message,
because each refuses at a different time (after the application ends, or
after the offer phase closes). Everything else is the same and lives here,
so the two cannot drift apart:

- The agent chooses only what to say (text) and why (reason). Where the
  message goes is never the agent's choice: in a real bank the system would
  look up the customer's contact details and send it; in this project the
  message is kept on file in the store, and the simulation hands it to the
  scripted customer. The agent never sees the customer's email or phone.
- Frame rule, enforced in code: no electronic messages without the
  customer's e-sign consent (E-SIGN Act). The consent timestamp the system
  recorded is checked, not anything the agent says.
- An empty message, a message without a reason, or a message longer than
  MAX_MESSAGE_LENGTH is refused.
- Idempotent: a retried tool call (same call ID) does not send the message
  again; it returns the same confirmation.
- Each sent message is recorded in the audit log with the agent's reason
  (the text itself is kept with the messages); a refused message is recorded
  as blocked.
"""

from collections.abc import Callable
from datetime import datetime

from loans.store import ApplicationStore, SentMessage

# A limit on message length, against runaway output and token waste.
MAX_MESSAGE_LENGTH = 2000


def send(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    call_id: str,
    text: str,
    reason: str,
    refusal: Callable[[], str | None],
) -> str:
    """Send a message to the customer and return a confirmation, or the
    refusal. refusal is the calling tool's own check for whether it may still
    act on the case; it returns the refusal text (already logged), or None."""
    stored = store.get(application_id)

    for position, message in enumerate(stored.messages, start=1):
        if message.message_id == call_id:
            return f"message sent (message {position} in this case)"
    if refused := refusal():
        return refused

    problem = _problem(stored.metadata.esign_consented_at is None, text, reason)
    if problem:
        store.log_event(
            application_id, now, "guardrail", "action_blocked", f"send_message: {problem}"
        )
        return f"not sent: {problem}"

    # Counted from the case as read above, which does not change when the
    # message is recorded (in Postgres it is a copy).
    position = len(stored.messages) + 1
    store.record_message(
        application_id,
        SentMessage(message_id=call_id, sent_at=now, text=text, reason=reason),
    )
    store.log_event(
        application_id, now, "agent", "message_sent", f"message {position}", reason
    )
    return f"message sent (message {position} in this case)"


def _problem(no_esign_consent: bool, text: str, reason: str) -> str | None:
    if no_esign_consent:
        return (
            "the customer has not given e-sign consent, so no electronic "
            "messages may be sent"
        )
    if not text.strip():
        return "the message is empty"
    if not reason.strip():
        return "a reason is required"
    if len(text) > MAX_MESSAGE_LENGTH:
        return f"the message is longer than {MAX_MESSAGE_LENGTH} characters"
    return None
