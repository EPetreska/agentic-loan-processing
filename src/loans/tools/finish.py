"""finish: the agent's tool for ending a case.

The agent decides how the case ends: it chooses the outcome, the reason (from
the fixed list in schemas/outcome.py) and writes an explanation for whoever
takes the case next. Frame rule, enforced in code: the ending must be allowed
by the bank's rules in checks/ending.py (for example, complete only with
current identity findings and a current income verification, and no
cancellation for no answer before 26 days). If it is not, finish refuses and
lists every problem, so the agent can fix them or choose another ending.

A case ends only once. Idempotent: a retried tool call (same call ID) returns
the same confirmation.

The store, application_id, now and call_id are injected by the system and are
not part of the tool's parameters that the LLM sees; the LLM chooses only the
outcome, the reason and the explanation. Text in a prompt, including an ID
injected by an attacker, therefore cannot change which case is ended: the
tool only ever works on the current application (protection against
cross-tenant access). The time is supplied by the system because time is
simulated in the evals, and the waiting period counts from the last message.
"""

from datetime import datetime

from pydantic import ValidationError

from loans.checks.ending import ending_problems
from loans.schemas.outcome import Outcome
from loans.store import ApplicationStore, CaseEnding


def finish(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    call_id: str,
    outcome: str,
    reason: str,
    explanation: str,
) -> str:
    """End the current case, if the chosen ending is allowed."""
    stored = store.get(application_id)

    ending = stored.ending
    if ending is not None:
        if ending.call_id == call_id:
            return f"case ended: {ending.outcome} ({ending.reason})"
        return f"not ended: the case has already ended ({ending.outcome}, {ending.reason})"

    try:
        chosen = Outcome(outcome=outcome, reason=reason)
    except ValidationError:
        return f"not ended: {reason!r} is not a valid reason for the outcome {outcome!r}"
    if not explanation.strip():
        return "not ended: an explanation is required"

    problems = ending_problems(stored, chosen, now)
    if problems:
        return "\n".join(["not ended:", *(f"- {problem}" for problem in problems)])

    store.record_ending(
        application_id,
        CaseEnding(
            call_id=call_id,
            outcome=chosen.outcome,
            reason=chosen.reason,
            explanation=explanation,
            ended_at=now,
        ),
    )
    return f"case ended: {chosen.outcome} ({chosen.reason})"
