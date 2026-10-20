"""Tests that every tool refuses once the case has ended."""

from pathlib import Path

import pytest

from loans.evals.case import load_case, submit_case
from loans.store import ApplicationStore
from loans.tools.check_application import check_application
from loans.tools.finish import finish
from loans.tools.mark_income_verified import mark_income_verified
from loans.tools.read_document import read_document
from loans.tools.send_message import send_message
from loans.tools.verify_identity import verify_identity

CASES = Path(__file__).parent.parent / "evals" / "cases"
REFUSAL = "not done: the case has already ended (sent_to_human, agent_unsure)"

CALLS = {
    "check_application": lambda s, a, now: check_application(s, a, now),
    "read_document": lambda s, a, now: read_document(s, a, "pay_stub", now),
    "verify_identity": lambda s, a, now: verify_identity(s, a, now),
    "send_message": lambda s, a, now: send_message(s, a, now, "c9", "Approved!", "x"),
    "mark_income_verified": lambda s, a, now: mark_income_verified(
        s, a, now, "c9", "pay_stub", "Matches."
    ),
}


@pytest.mark.parametrize("tool", CALLS)
def test_tool_refuses_after_the_case_ended(tool):
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    app_id = submit_case(store, case)
    now = case.info.start_time
    finish(store, app_id, now, "c1", "sent_to_human", "agent_unsure", "Unsure.")

    assert CALLS[tool](store, app_id, now) == REFUSAL
    stored = store.get(app_id)
    assert stored.messages == []
    assert stored.income_verification is None
    assert stored.identity_findings is None
    last = stored.audit_log[-1]
    assert (last.actor, last.event) == ("guardrail", "action_blocked")
    assert last.details == f"{tool}: the case has already ended"
