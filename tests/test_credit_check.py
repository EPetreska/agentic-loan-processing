"""Tests for the credit check: the mock bureau, and the workflow running it
only after a complete ending."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from loans.evals.case import load_case, submit_case
from loans.evals.simulate import simulate_case
from loans.services.credit_bureau import NoCreditConsent, check_credit
from loans.store import ApplicationStore
from test_simulate import MARIA, FakeModel, ask, step, wait

CASES = Path(__file__).parent.parent / "evals" / "cases"


def simulate(case_name, answers):
    store = ApplicationStore()
    case = load_case(CASES / case_name)
    result = asyncio.run(simulate_case(case, store, FakeModel(messages=iter(answers))))
    return store.get(result.application_id)


def test_a_complete_case_is_credit_checked():
    stored = simulate("02_maria_job_change", MARIA)
    assert stored.credit_check.passed
    assert stored.credit_check.checked_at == stored.ending.ended_at + timedelta(minutes=1)
    event = next(e for e in stored.audit_log if e.event == "credit_checked")
    assert (event.actor, event.details) == ("system", "passed")


@pytest.mark.parametrize(
    "case_name, answers",
    [
        (
            "03_no_answer",
            [ask(), wait(), step("finish", {"outcome": "cancelled", "reason": "no_answer", "explanation": "No reply."})],
        ),
        (
            "04_retiree_other_employment",
            [step("finish", {"outcome": "sent_to_human", "reason": "other_employment", "explanation": "Retired."})],
        ),
    ],
    ids=["cancelled", "sent to a human"],
)
def test_other_endings_go_no_further(case_name, answers):
    stored = simulate(case_name, answers)
    assert stored.ending.outcome != "complete"
    assert stored.credit_check is None
    assert "credit_checked" not in [e.event for e in stored.audit_log]


def test_no_credit_check_without_consent():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    app_id = submit_case(store, case)
    store.correct_fields(app_id, {"credit_check_consent": "no"}, now=case.info.start_time)
    with pytest.raises(NoCreditConsent):
        check_credit(store.get(app_id).metadata, case.info.start_time)


def test_the_result_is_kept_on_file_in_postgres(pg_store):
    case = load_case(CASES / "01_clean_application")
    app_id = submit_case(pg_store, case)
    pg_store.record_credit_check(app_id, check_credit(pg_store.get(app_id).metadata, case.info.start_time))
    stored = pg_store.get(app_id)
    assert stored.credit_check.passed
    assert stored.audit_log[-1].event == "credit_checked"
