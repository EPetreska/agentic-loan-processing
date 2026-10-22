"""Tests for the workflow's offer phase on its own, without an LLM: sending
the offer, signing, expiry, and the loan officer's decision."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from loans.evals.case import load_case, submit_case
from loans.evals.simulate import SimulatedClock
from loans.store import ApplicationStore, CaseEnding
from loans.tools.offer.request_approval import request_approval
from loans.workflow import (
    OfferPhaseError,
    after_application,
    decide_approval,
    expire_if_due,
    receive_reply,
    send_offer,
    sign_offer,
    valid_until,
)

CASES = Path(__file__).parent.parent / "evals" / "cases"
DAY = timedelta(days=1)


def offered(store=None):
    """Maria's case, completed, credit-checked, with her offer sent: (store,
    ID, the time the offer was sent)."""
    store = store or ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    start = case.info.start_time + 2 * DAY
    store.record_ending(app_id, CaseEnding(call_id="f", outcome="complete", reason="all_checks_passed",
                                           explanation="x", ended_at=start))  # fmt: skip
    clock = SimulatedClock(start)
    after_application(store, app_id, clock)
    return store, app_id, store.get(app_id).offers_sent[-1].sent_at


def ask_officer(store, app_id, sent_at, call_id="call-1"):
    request_approval(store, app_id, sent_at + DAY, call_id,
                     "The customer asks to borrow $17,000 instead of $15,000.", "More than offered.")  # fmt: skip


def last_event(store, app_id):
    event = store.get(app_id).audit_log[-1]
    return event.actor, event.event, event.details


# Sending the offer


def test_the_offer_is_sent_with_its_notice():
    store, app_id, sent_at = offered()
    [sent] = store.get(app_id).offers_sent
    assert sent.offer_index == 0
    assert sent.notice.splitlines() == [
        "Your loan offer",
        "Amount: $15,000.00",
        "Annual percentage rate (APR): 10.99%",
        "Term: 48 monthly payments",
        "Monthly payment: $387.61",
        "Total of payments: $18,605.28",
        "Finance charge (interest): $3,605.28",
        f"This offer is valid until {(sent_at + 30 * DAY).date().isoformat()}. To accept it, sign it"
        " electronically in your online account; a reply by message does not accept it.",
    ]
    assert last_event(store, app_id) == ("system", "offer_sent", "offer 1: $15,000, 48 months")


# Signing


def test_signing_the_offer_accepts_it():
    store, app_id, sent_at = offered()
    sign_offer(store, app_id, 0, sent_at + 3 * DAY)
    stored = store.get(app_id)
    assert (stored.signature.offer_index, stored.signature.signed_at) == (0, sent_at + 3 * DAY)
    assert stored.offer_closing.outcome == "accepted"
    assert [e.event for e in stored.audit_log][-2:] == ["offer_signed", "offer_closed"]
    assert stored.audit_log[-2].actor == "customer"


@pytest.mark.parametrize("index", [1, -1], ids=["not sent", "no such offer"])
def test_only_the_latest_offer_sent_can_be_signed(index):
    store, app_id, sent_at = offered()
    with pytest.raises(OfferPhaseError, match="only the latest offer sent"):
        sign_offer(store, app_id, index, sent_at + DAY)
    assert store.get(app_id).signature is None


def test_an_offer_can_be_signed_until_its_last_moment():
    store, app_id, sent_at = offered()
    with pytest.raises(OfferPhaseError, match="no longer valid"):
        sign_offer(store, app_id, 0, sent_at + 30 * DAY)
    sign_offer(store, app_id, 0, sent_at + 30 * DAY - timedelta(minutes=1))
    assert store.get(app_id).offer_closing.outcome == "accepted"


def test_nothing_after_the_phase_closed():
    store, app_id, sent_at = offered()
    sign_offer(store, app_id, 0, sent_at + DAY)
    with pytest.raises(OfferPhaseError, match="has closed"):
        sign_offer(store, app_id, 0, sent_at + 2 * DAY)
    with pytest.raises(OfferPhaseError, match="has closed"):
        send_offer(store, app_id, sent_at + 2 * DAY)
    assert not expire_if_due(store, app_id, sent_at + 40 * DAY)


# Expiry


def test_the_offer_expires_after_30_days_not_before():
    store, app_id, sent_at = offered()
    assert not expire_if_due(store, app_id, sent_at + 30 * DAY - timedelta(minutes=1))
    assert store.get(app_id).offer_closing is None
    assert expire_if_due(store, app_id, sent_at + 30 * DAY)
    assert store.get(app_id).offer_closing.outcome == "expired"
    assert last_event(store, app_id) == ("system", "offer_closed", "expired")


# The loan officer's decision


def test_an_approval_creates_and_sends_the_revised_offer():
    store, app_id, sent_at = offered()
    ask_officer(store, app_id, sent_at)
    decided_at = sent_at + 2 * DAY
    decide_approval(store, app_id, "officer-17", "approve", "Income supports it.", decided_at,
                    amount=Decimal("17000"), term_months=48)  # fmt: skip
    stored = store.get(app_id)
    [request] = stored.approval_requests
    assert (request.status, request.officer_id, request.officer_note) == ("approved", "officer-17", "Income supports it.")
    revised = stored.offers[1]
    assert (revised.amount, revised.term_months, revised.monthly_payment) == (Decimal("17000"), 48, Decimal("439.29"))
    assert [s.offer_index for s in stored.offers_sent] == [0, 1]
    decision = next(e for e in stored.audit_log if e.event == "approval_decided")
    assert (decision.actor, decision.details) == ("human", "officer-17: request 1 approved")
    assert "Income supports it." not in decision.model_dump_json()  # the note is kept, not logged


def test_the_revised_offer_replaces_the_first_and_restarts_the_clock():
    store, app_id, sent_at = offered()
    ask_officer(store, app_id, sent_at)
    decided_at = sent_at + 10 * DAY
    decide_approval(store, app_id, "officer-17", "approve", "ok", decided_at,
                    amount=Decimal("17000"), term_months=48)  # fmt: skip
    with pytest.raises(OfferPhaseError, match="only the latest"):
        sign_offer(store, app_id, 0, decided_at + DAY)
    # 30 days from the revised offer, not the first one.
    assert not expire_if_due(store, app_id, sent_at + 30 * DAY)
    assert valid_until(store.get(app_id).offers_sent[-1]) == decided_at + 30 * DAY
    sign_offer(store, app_id, 1, decided_at + DAY)
    assert store.get(app_id).offer_closing.outcome == "accepted"


def test_a_decline_leaves_the_offer_as_it_is():
    store, app_id, sent_at = offered()
    ask_officer(store, app_id, sent_at)
    decide_approval(store, app_id, "officer-17", "decline", "Too much for the income.", sent_at + 2 * DAY)
    stored = store.get(app_id)
    assert stored.approval_requests[0].status == "declined"
    assert len(stored.offers) == 1 and len(stored.offers_sent) == 1
    assert last_event(store, app_id) == ("human", "approval_decided", "officer-17: request 1 declined")


@pytest.mark.parametrize(
    "amount, term, problem",
    [
        (None, 48, "needs an amount and a term"),
        (Decimal("60000"), 48, "amount must be between"),
        (Decimal("17000"), 72, "term must be one of"),
    ],
    ids=["no amount", "too much", "no such term"],
)
def test_an_approval_must_be_a_product_the_bank_offers(amount, term, problem):
    store, app_id, sent_at = offered()
    ask_officer(store, app_id, sent_at)
    with pytest.raises(OfferPhaseError, match=problem):
        decide_approval(store, app_id, "officer-17", "approve", "ok", sent_at + 2 * DAY, amount=amount, term_months=term)
    stored = store.get(app_id)
    assert stored.approval_requests[0].status == "pending" and len(stored.offers) == 1


def test_a_decision_needs_a_waiting_request():
    store, app_id, sent_at = offered()
    with pytest.raises(OfferPhaseError, match="no request waiting"):
        decide_approval(store, app_id, "officer-17", "decline", "x", sent_at + DAY)


def test_a_decision_after_the_phase_closed_is_refused_and_logged():
    store, app_id, sent_at = offered()
    ask_officer(store, app_id, sent_at)
    sign_offer(store, app_id, 0, sent_at + 2 * DAY)  # she signed while the request waited
    with pytest.raises(OfferPhaseError, match="has closed"):
        decide_approval(store, app_id, "officer-17", "approve", "ok", sent_at + 3 * DAY,
                        amount=Decimal("17000"), term_months=48)  # fmt: skip
    stored = store.get(app_id)
    assert stored.approval_requests[0].status == "pending"  # left on file, unanswered
    assert last_event(store, app_id) == (
        "guardrail", "action_blocked", "approval decision: the offer phase has closed (accepted)"
    )


# The customer's messages


def test_a_message_during_the_offer_phase_is_kept():
    store, app_id, sent_at = offered()
    receive_reply(store, app_id, sent_at + DAY, "Is there a fee if I pay early?")
    reply = store.get(app_id).replies[-1]
    assert (reply.message, reply.received_at) == ("Is there a fee if I pay early?", sent_at + DAY)


def test_the_offer_phase_is_kept_in_postgres(pg_store):
    store, app_id, sent_at = offered(pg_store)
    ask_officer(store, app_id, sent_at)
    decide_approval(store, app_id, "officer-17", "approve", "ok", sent_at + DAY,
                    amount=Decimal("17000"), term_months=48)  # fmt: skip
    sign_offer(store, app_id, 1, sent_at + 2 * DAY)
    stored = pg_store.get(app_id)
    assert [s.offer_index for s in stored.offers_sent] == [0, 1]
    assert stored.signature.offer_index == 1 and stored.offer_closing.outcome == "accepted"
    assert stored.approval_requests[0].officer_id == "officer-17"
