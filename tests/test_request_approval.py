"""Tests for the offer agent's request_approval tool."""

from datetime import timedelta
from pathlib import Path

import pytest

from loans.evals.case import load_case, submit_case
from loans.offers import offer_for
from loans.store import ApplicationStore, OfferClosing
from loans.tools.offer.request_approval import MAX_REQUEST_LENGTH, request_approval

CASES = Path(__file__).parent.parent / "evals" / "cases"
REQUEST = "The customer asks to borrow $17,000 instead of $15,000, still over 48 months."
REASON = "More money than the offer; only an officer can approve it."


@pytest.fixture
def offered():
    """Maria's case with her offer made, as after the credit check."""
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    return store, app_id, case


def ask(store, app_id, case, call_id="call-1", request=REQUEST, reason=REASON):
    return request_approval(store, app_id, case.info.start_time + timedelta(days=1), call_id, request, reason)


def blocked(store, app_id) -> str:
    event = store.get(app_id).audit_log[-1]
    assert (event.actor, event.event) == ("guardrail", "action_blocked")
    return event.details


def test_a_request_waits_for_the_officer(offered):
    store, app_id, case = offered
    assert ask(store, app_id, case) == (
        "sent to a loan officer (request 1); the case waits for the officer's decision"
    )
    stored = store.get(app_id)
    [request] = stored.approval_requests
    assert (request.request, request.reason, request.request_id) == (REQUEST, REASON, "call-1")
    assert request.status == "pending" and request.decided_at is None
    assert stored.pending_approval() == request


def test_the_request_is_logged_without_its_text(offered):
    store, app_id, case = offered
    ask(store, app_id, case)
    event = store.get(app_id).audit_log[-1]
    assert (event.actor, event.event, event.details, event.reason) == (
        "agent", "approval_requested", "request 1", REASON
    )
    assert "17,000" not in event.model_dump_json()  # the text stays with the request


def test_a_retry_does_not_ask_twice(offered):
    store, app_id, case = offered
    first = ask(store, app_id, case)
    assert ask(store, app_id, case) == first
    assert len(store.get(app_id).approval_requests) == 1


def test_one_request_at_a_time(offered):
    store, app_id, case = offered
    ask(store, app_id, case)
    other = ask(store, app_id, case, call_id="call-2", request="The customer asks for 60 months.")
    assert other == "not sent: a request is already waiting for a loan officer's decision"
    assert len(store.get(app_id).approval_requests) == 1
    assert blocked(store, app_id) == (
        "request_approval: a request is already waiting for a loan officer's decision"
    )


def test_a_new_request_once_the_last_was_decided(offered):
    store, app_id, case = offered
    ask(store, app_id, case)
    # As the officer's decision will do, once built.
    with store._changing(app_id) as stored:
        stored.approval_requests[0].status = "declined"
    assert ask(store, app_id, case, call_id="call-2", request="The customer asks for 60 months.") == (
        "sent to a loan officer (request 2); the case waits for the officer's decision"
    )


@pytest.mark.parametrize(
    "request_text, reason, problem",
    [
        (" ", REASON, "the request is empty"),
        (REQUEST, " ", "a reason is required"),
        ("x" * (MAX_REQUEST_LENGTH + 1), REASON, f"the request is longer than {MAX_REQUEST_LENGTH} characters"),
    ],
    ids=["empty", "no reason", "too long"],
)
def test_a_request_needs_text_and_a_reason(offered, request_text, reason, problem):
    store, app_id, case = offered
    assert ask(store, app_id, case, request=request_text, reason=reason) == f"not sent: {problem}"
    assert store.get(app_id).approval_requests == []
    assert blocked(store, app_id) == f"request_approval: {problem}"


def test_no_request_before_there_is_an_offer():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    assert ask(store, app_id, case) == "not done: the case has no offer yet"
    assert store.get(app_id).approval_requests == []


def test_no_request_after_the_offer_phase_closed(offered):
    store, app_id, case = offered
    store.record_offer_closing(app_id, OfferClosing(outcome="accepted", closed_at=case.info.start_time))
    assert ask(store, app_id, case) == "not done: the offer phase has closed (accepted)"
    assert store.get(app_id).approval_requests == []


def test_requests_are_kept_in_postgres(pg_store):
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)
    pg_store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    ask(pg_store, app_id, case)
    stored = pg_store.get(app_id)
    assert stored.pending_approval().request == REQUEST
    assert stored.audit_log[-1].event == "approval_requested"
