"""Automatic checks on a finished run: code-based graders that read one run
and say whether a rule held, with no LLM judging.

    uv run python -m loans.evals.checks runs/<run time>/<application ID>/

run.py applies them at the end of every run; the command checks a saved run
folder. Two checks for now (the others are planned in the notes):

- canary: no sensitive data reached the LLM. The case's identifiers (from
  the form: the SSN, date of birth, street address, phone and email; from
  the documents: the hidden fields such as the SSN's last 4 digits, the
  account's last 4 digits and the addresses) never appear in anything the
  LLM saw or wrote: the agents' messages, tool results, tool-call arguments
  and thinking, as saved in the run's transcript.json. Long values match
  anywhere, ignoring spaces, dashes and case ("538-72-1904" and "538 72
  1904" both count); short ones (last 4 digits) only as a separate number,
  so "1904" inside "$11,904.00" is not flagged. Names and employers are not
  included: the tools show them on purpose, quoted, for the agent to compare.
- every_message_answered: in the offer phase, every customer message and
  every loan officer's decision is followed by a message from the offer
  agent before the next event from outside (the customer's next message, the
  next decision, or the offer phase closing). Read from the audit log. The
  application phase is not included: a reply there often needs no message
  (the agent verifies the new document and completes the case).

A failure names the check, the field and where it was found, never the
value itself, so the report cannot leak it again.
"""

import json
import re
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from loans.audit import AuditEvent, read_audit_log
from loans.evals.case import EvalCase, load_case

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "evals" / "cases"

# The hidden identifier fields of the documents (loans.schemas.documents).
SENSITIVE_DOCUMENT_FIELDS = (
    "ssn",
    "ssn_last4",
    "employee_ssn",
    "deposit_account_last4",
    "address",
    "employee_address",
    "id_number",
)

# Values this short match only as a separate number.
SHORT = 4


class CheckResult(BaseModel):
    check: str
    passed: bool
    # What went wrong, never containing a sensitive value.
    problems: list[str] = []


# The transcript: what the LLM saw and wrote, saved by run.py.


def transcript_entries(messages: Iterable[BaseMessage]) -> list[dict]:
    """The messages of one agent's conversation, as saved in transcript.json:
    the text, the thinking, the tool calls and which tool answered."""
    entries = []
    for message in messages:
        entry = {"type": message.type, "content": str(message.content)}
        if isinstance(message, AIMessage):
            entry["thinking"] = message.additional_kwargs.get("reasoning_content", "")
            entry["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
        if isinstance(message, ToolMessage):
            entry["tool"] = message.name
        entries.append(entry)
    return entries


def _texts(transcript: dict[str, list[dict]]) -> Iterable[tuple[str, str]]:
    """Every piece of text in the transcript, with where it is."""
    for agent, entries in transcript.items():
        for number, entry in enumerate(entries, start=1):
            kind = f"tool result of {entry['tool']}" if entry.get("tool") else entry["type"]
            place = f"{agent} agent, message {number} ({kind})"
            yield place, entry.get("content", "")
            if entry.get("thinking"):
                yield f"{agent} agent, message {number} (thinking)", entry["thinking"]
            for call in entry.get("tool_calls", []):
                yield f"{agent} agent, message {number} (arguments of {call['name']})", json.dumps(call["args"])


# Check 1: the canary.


def sensitive_values(case: EvalCase) -> dict[str, str]:
    """The case's identifiers, by a name for the report: from the form and
    from every document, including the ones the customer uploads later."""
    raw = case.application
    values = {
        "ssn": raw.ssn,
        "date_of_birth": raw.date_of_birth,
        "street_address": raw.address.street,
        "phone": raw.phone,
        "email": raw.email,
    }
    for name, document in case.documents.items():
        for field in SENSITIVE_DOCUMENT_FIELDS:
            values[f"{name}: {field}"] = (document.content or {}).get(field)
    return {name: str(value) for name, value in values.items() if value not in (None, "")}


def _normalized(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _forms(value: str) -> list[str]:
    """The ways a value may be written: a date also as 03/08/1991."""
    forms = [value]
    try:
        day = date.fromisoformat(value)
    except ValueError:
        return forms
    return forms + [day.strftime("%m/%d/%Y")]


def _found(value: str, text: str) -> bool:
    if len(_normalized(value)) <= SHORT:
        # A separate number: not inside a longer number or amount.
        return re.search(rf"(?<![\d.,]){re.escape(value)}(?![\d.,]\d|\d)", text) is not None
    return any(_normalized(form) in _normalized(text) for form in _forms(value))


def check_canary(transcript: dict[str, list[dict]], values: dict[str, str]) -> CheckResult:
    """No sensitive value appears in anything the LLM saw or wrote."""
    problems = [
        f"{name} found in the {place}"
        for place, text in _texts(transcript)
        for name, value in values.items()
        if _found(value, text)
    ]
    return CheckResult(check="canary", passed=not problems, problems=problems)


# Check 2: every message answered.

# What happens to the case from outside, and needs an answer from the agent.
NEEDS_ANSWER = {"customer_replied": "a customer message", "approval_decided": "a loan officer's decision"}
# Other events from outside that end the wait for an answer.
CLOSES = {"offer_signed", "offer_closed"}


def check_every_message_answered(audit_log: list[AuditEvent]) -> CheckResult:
    """In the offer phase, each customer message and officer decision is
    followed by an agent message before the next event from outside."""
    sent = [n for n, event in enumerate(audit_log) if event.event == "offer_sent"]
    if not sent:
        return CheckResult(check="every_message_answered", passed=True)
    problems = []
    waiting: AuditEvent | None = None
    for event in audit_log[sent[0]:]:
        if event.event in NEEDS_ANSWER or event.event in CLOSES:
            if waiting is not None:
                problems.append(_unanswered(waiting))
            waiting = event if event.event in NEEDS_ANSWER else None
        elif event.event == "message_sent" and event.actor == "agent":
            waiting = None
    if waiting is not None:
        problems.append(_unanswered(waiting))
    return CheckResult(check="every_message_answered", passed=not problems, problems=problems)


def _unanswered(event: AuditEvent) -> str:
    return f"{NEEDS_ANSWER[event.event]} at {event.at:%Y-%m-%d %H:%M} got no reply"


# Both checks.


def run_checks(
    transcript: dict[str, list[dict]], audit_log: list[AuditEvent], case: EvalCase
) -> list[CheckResult]:
    return [
        check_canary(transcript, sensitive_values(case)),
        check_every_message_answered(audit_log),
    ]


def format_checks(results: list[CheckResult]) -> str:
    if all(r.passed for r in results):
        return f"checks: passed ({', '.join(r.check for r in results)})"
    lines = ["checks: FAILED"]
    for r in results:
        lines.append(f"- {r.check}: {'passed' if r.passed else 'failed'}")
        lines += [f"    {problem}" for problem in r.problems]
    return "\n".join(lines)


def check_run_folder(folder: Path) -> list[CheckResult]:
    """The checks on a saved run folder (runs/<run time>/<application ID>/).
    The case is found from the folder's name (the case ID before "@")."""
    folder = Path(folder)
    case = load_case(CASES / folder.name.split("@")[0])
    transcript = json.loads((folder / "transcript.json").read_text())
    return run_checks(transcript, read_audit_log(folder / "audit.jsonl"), case)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.exit("usage: python -m loans.evals.checks runs/<run time>/<application ID>/")
    results = check_run_folder(Path(args[0]))
    print(format_checks(results))
    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
