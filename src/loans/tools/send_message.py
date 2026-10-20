"""send_message: the agent's tool for writing to the customer.

The agent chooses only what to say (text) and why (reason). Where the message
goes is never the agent's choice: in a real bank the system would look up the
customer's contact details and send it; in this project the message is kept
on file in the store, and the eval runner hands it to the scripted customer.
The agent never sees the customer's email or phone.

Frame rule, enforced in code: no electronic messages without the customer's
e-sign consent (E-SIGN Act). The tool checks the consent timestamp the system
recorded at intake, not anything the agent says.

Idempotent: a retried tool call (same call ID) does not send the message
again; it returns the same confirmation.

The store, application_id, now and call_id are injected by the system and are
not part of the tool's parameters that the LLM sees; the LLM chooses only the
text and the reason. Text in a prompt, including an ID injected by an
attacker, therefore cannot change which customer receives the message: the
tool only ever works on the current application (protection against
cross-tenant access). The time is supplied by the system because time is
simulated in the evals, and the deadlines count from the last message.

Each sent message is recorded in the audit log with the agent's reason (the
text itself is kept with the messages); a refused message is recorded as
blocked.

Once the case has ended, the tool refuses and the attempt is logged as
blocked.
"""

from datetime import datetime

from loans.store import ApplicationStore, SentMessage

# A limit on message length, against runaway output and token waste.
MAX_MESSAGE_LENGTH = 2000


def send_message(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    call_id: str,
    text: str,
    reason: str,
) -> str:
    """Send a message to the current customer and return a confirmation."""
    stored = store.get(application_id)

    for position, message in enumerate(stored.messages, start=1):
        if message.message_id == call_id:
            return f"message sent (message {position} in this case)"
    if refusal := store.refuse_if_ended(application_id, now, "send_message"):
        return refusal

    problem = _problem(stored.metadata.esign_consented_at is None, text, reason)
    if problem:
        store.log_event(
            application_id, now, "guardrail", "action_blocked", f"send_message: {problem}"
        )
        return f"not sent: {problem}"

    store.record_message(
        application_id,
        SentMessage(message_id=call_id, sent_at=now, text=text, reason=reason),
    )
    position = len(stored.messages)
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
