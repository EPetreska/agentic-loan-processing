"""Tests for the offer agent's send_message tool. The rules shared with the
application agent are tested in test_send_message.py; these check that they
hold here too, and the offer agent's own limits."""

from datetime import timedelta
from pathlib import Path

import pytest

from loans.evals.case import load_case, submit_case
from loans.offers import offer_for
from loans.store import ApplicationStore, OfferClosing
from loans.tools.messages import MAX_MESSAGE_LENGTH
from loans.tools.application.send_message import send_message as application_send
from loans.tools.offer.send_message import send_message

CASES = Path(__file__).parent.parent / "evals" / "cases"
TEXT = "There is no fee for paying the loan off early."
REASON = "The customer asked about early repayment."


@pytest.fixture
def offered():
    """Maria's case with her offer made, as after the credit check."""
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    return store, app_id, case


def send(store, app_id, case, call_id="call-1", text=TEXT, reason=REASON):
    return send_message(store, app_id, case.info.start_time + timedelta(days=1), call_id, text, reason)


def blocked(store, app_id) -> str:
    event = store.get(app_id).audit_log[-1]
    assert (event.actor, event.event) == ("guardrail", "action_blocked")
    return event.details


def test_a_message_about_the_offer_is_sent(offered):
    store, app_id, case = offered
    assert send(store, app_id, case) == "message sent (message 1 in this case)"
    [message] = store.get(app_id).messages
    assert (message.text, message.reason, message.message_id) == (TEXT, REASON, "call-1")
    event = store.get(app_id).audit_log[-1]
    assert (event.actor, event.event, event.reason) == ("agent", "message_sent", REASON)


def test_the_count_continues_from_the_application_messages(offered):
    # The customer's conversation with the bank is one: the application
    # agent's messages come first.
    store, app_id, case = offered
    application_send(store, app_id, case.info.start_time, "app-1", "Please upload a pay stub.", "Too old.")
    assert send(store, app_id, case) == "message sent (message 2 in this case)"


def test_a_retry_is_not_sent_twice(offered):
    store, app_id, case = offered
    first = send(store, app_id, case)
    assert send(store, app_id, case) == first
    assert len(store.get(app_id).messages) == 1


@pytest.mark.parametrize(
    "text, reason, problem",
    [
        (" ", REASON, "the message is empty"),
        (TEXT, " ", "a reason is required"),
        ("x" * (MAX_MESSAGE_LENGTH + 1), REASON, f"the message is longer than {MAX_MESSAGE_LENGTH} characters"),
    ],
    ids=["empty", "no reason", "too long"],
)
def test_the_shared_rules_hold(offered, text, reason, problem):
    store, app_id, case = offered
    assert send(store, app_id, case, text=text, reason=reason) == f"not sent: {problem}"
    assert store.get(app_id).messages == []
    assert blocked(store, app_id) == f"send_message: {problem}"


def test_no_message_without_e_sign_consent(offered):
    store, app_id, case = offered
    store.correct_fields(app_id, {"esign_consent": "no"}, now=case.info.start_time)
    assert send(store, app_id, case).startswith("not sent: the customer has not given e-sign consent")


def test_no_message_before_there_is_an_offer():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    assert send(store, app_id, case) == "not done: the case has no offer yet"
    assert store.get(app_id).messages == []
    assert blocked(store, app_id) == "send_message: the case has no offer yet"


@pytest.mark.parametrize("outcome", ["accepted", "expired"])
def test_no_message_after_the_offer_phase_closed(offered, outcome):
    store, app_id, case = offered
    closed_at = case.info.start_time + timedelta(hours=1)
    store.record_offer_closing(app_id, OfferClosing(outcome=outcome, closed_at=closed_at))
    assert send(store, app_id, case) == f"not done: the offer phase has closed ({outcome})"
    assert store.get(app_id).messages == []
    assert blocked(store, app_id) == f"send_message: the offer phase has closed ({outcome})"


def test_closing_is_logged_once_and_kept_in_postgres(pg_store):
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)
    pg_store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    pg_store.record_offer_closing(app_id, OfferClosing(outcome="accepted", closed_at=case.info.start_time))
    stored = pg_store.get(app_id)
    assert stored.offer_closing.outcome == "accepted"
    event = stored.audit_log[-1]
    assert (event.actor, event.event, event.details) == ("system", "offer_closed", "accepted")
