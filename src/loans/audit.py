"""The audit log: the business record of what happened in each case.

One event per step, with the case ID, the exact time, who acted and a stable
event name, so a person (an auditor, the human approver) can follow a case
afterwards, and the log can later be exported as an event log for process
mining. Events are only ever added, never changed or removed; for now they
are kept in the store, later in an append-only Postgres table.

This is separate from tracing (LangSmith), which records the prompts and LLM
calls for the developer. The audit log holds no prompts and no sensitive data:
no SSN, date of birth or contact details, and no values the customer typed.

To read a saved log file as a table:

    uv run python -m loans.audit runs/<run time>/<case>/audit.jsonl
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from loans.spotlight import quote

# Who acted. "guardrail" is the code refusing an action the agent asked for;
# "human" is a person at the bank, such as a loan officer.
Actor = Literal["system", "customer", "agent", "guardrail", "human"]

EventName = Literal[
    # system and customer
    "application_submitted",
    "document_uploaded",
    "form_corrected",
    "customer_replied",
    "offer_signed",
    # the application agent
    "form_checked",
    "document_read",
    "identity_checked",
    "message_sent",
    "income_verified",
    "application_completed",
    "application_cancelled",
    "sent_to_human",
    # guardrail
    "action_blocked",
    # the bank's workflow, after the application
    "credit_checked",
    "offer_created",
    "offer_sent",
    "offer_closed",
    # a person at the bank
    "approval_decided",
    # the offer agent (it also uses message_sent)
    "approval_requested",
    "documents_retrieved",
    # the evals: a case set up to start later in the process
    "eval_setup",
]


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    application_id: str
    at: datetime
    actor: Actor
    event: EventName
    # Short facts about the step, in fixed wording.
    details: str = ""
    # The agent's reason, in its own words, when it gives one.
    reason: str | None = None
    # For agent and guardrail events: which model and which version of the
    # agent's instructions were running, so a change in behavior can be traced
    # to a model or prompt change.
    model: str | None = None
    prompt_version: str | None = None


class AgentVersion(BaseModel):
    """The model and prompt version of the running agent."""

    model: str
    prompt_version: str


def format_audit_log(events: list[AuditEvent]) -> str:
    """The log as a table, one line per event, for reading in the terminal.

    Times are shown in UTC. The agent's reason is quoted, so a reason with
    line breaks stays on one line, and a long reason is cut in this view (the
    event keeps it in full).
    """
    lines = []
    for event in events:
        at = event.at.astimezone(UTC).strftime("%Y-%m-%d %H:%M")
        line = f"{at}  {event.actor:<9}  {event.event:<21}  {event.details}"
        if event.reason is not None:
            line += f"  reason: {quote(event.reason)}"
        lines.append(line.rstrip())
    return "\n".join(lines)


def write_audit_log(path: Path, events: list[AuditEvent]) -> None:
    """Write the events to a log file, one JSON line each (the same format
    the store writes as events happen), replacing the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(event.model_dump_json() + "\n" for event in events))


def read_audit_log(path: Path) -> list[AuditEvent]:
    """The events in a saved log file, oldest first."""
    with Path(path).open() as f:
        return [AuditEvent.model_validate_json(line) for line in f if line.strip()]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m loans.audit <log file>")
    print(format_audit_log(read_audit_log(Path(sys.argv[1]))))
