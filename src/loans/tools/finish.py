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

The ending is recorded in the audit log with the reason and the agent's
explanation; a refused ending is recorded as blocked, with every problem.
"""

from datetime import datetime

from pydantic import ValidationError

from loans.checks.ending import ending_problems
from loans.schemas.outcome import Outcome
from loans.store import ApplicationStore, CaseEnding

EVENT_BY_OUTCOME = {
    "complete": "application_completed",
    "cancelled": "application_cancelled",
    "sent_to_human": "sent_to_human",
}


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

    def blocked(text: str, problems: list[str]) -> str:
        store.log_event(
            application_id,
            now,
            "guardrail",
            "action_blocked",
            "finish: " + "; ".join(problems),
            explanation or None,
        )
        return text

    ending = stored.ending
    if ending is not None:
        if ending.call_id == call_id:
            return f"case ended: {ending.outcome} ({ending.reason})"
        problem = f"the case has already ended ({ending.outcome}, {ending.reason})"
        return blocked(f"not ended: {problem}", [problem])

    try:
        chosen = Outcome(outcome=outcome, reason=reason)
    except ValidationError:
        problem = f"{reason!r} is not a valid reason for the outcome {outcome!r}"
        return blocked(f"not ended: {problem}", [problem])
    if not explanation.strip():
        problem = "an explanation is required"
        return blocked(f"not ended: {problem}", [problem])

    problems = ending_problems(stored, chosen, now)
    if problems:
        text = "\n".join(["not ended:", *(f"- {problem}" for problem in problems)])
        return blocked(text, problems)

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
    store.log_event(
        application_id,
        now,
        "agent",
        EVENT_BY_OUTCOME[chosen.outcome],
        chosen.reason,
        explanation,
    )
    return f"case ended: {chosen.outcome} ({chosen.reason})"
