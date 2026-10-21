"""Tests for the offer: the loan math, and the workflow creating it only
after a passed credit check."""

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from loans import policy
from loans.evals.case import load_case, submit_case
from loans.evals.simulate import simulate_case
from loans.offers import create_offer, monthly_payment, offer_for
from loans.schemas.application import ALLOWED_TERMS
from loans.store import ApplicationStore
from test_simulate import MARIA, FakeModel, ask, step, wait

CASES = Path(__file__).parent.parent / "evals" / "cases"


@pytest.mark.parametrize(
    "months, apr, payment, total, interest",
    [
        (36, "0.0999", "483.94", "17421.84", "2421.84"),
        (48, "0.1099", "387.61", "18605.28", "3605.28"),
        (60, "0.1199", "333.59", "20015.40", "5015.40"),
    ],
)
def test_the_numbers_for_15000(months, apr, payment, total, interest):
    # Values from the standard loan formula.
    offer = create_offer(Decimal("15000"), months)
    assert offer.amount == Decimal("15000")
    assert (offer.apr, offer.monthly_payment, offer.total_of_payments, offer.finance_charge) == (
        Decimal(apr), Decimal(payment), Decimal(total), Decimal(interest)
    )


def test_every_allowed_term_has_a_rate():
    assert set(policy.OFFER_RATES) == ALLOWED_TERMS


def test_a_longer_term_costs_more_in_total_but_less_a_month():
    offers = [create_offer(Decimal("15000"), months) for months in sorted(ALLOWED_TERMS)]
    payments = [o.monthly_payment for o in offers]
    totals = [o.total_of_payments for o in offers]
    assert payments == sorted(payments, reverse=True)
    assert totals == sorted(totals)


def test_the_payments_pay_off_the_loan():
    # Paying the monthly payment for the whole term leaves (almost) nothing:
    # at most what the rounding to cents adds up to.
    amount, apr, months = Decimal("15000"), Decimal("0.1099"), 48
    payment = monthly_payment(amount, apr, months)
    balance = amount
    for _ in range(months):
        balance = balance * (1 + apr / 12) - payment
    assert abs(balance) < Decimal("0.01") * months


def test_the_offer_is_what_the_customer_asked_for():
    case = load_case(CASES / "02_maria_job_change")  # $15,000 over 48 months
    offer = offer_for(case.application)
    assert (offer.amount, offer.term_months) == (Decimal("15000"), 48)


def simulate(case_name, answers):
    store = ApplicationStore()
    case = load_case(CASES / case_name)
    result = asyncio.run(simulate_case(case, store, FakeModel(messages=iter(answers))))
    return store.get(result.application_id)


def test_a_complete_case_gets_its_offer():
    stored = simulate("02_maria_job_change", MARIA)
    assert [o.term_months for o in stored.offers] == [48]
    event = stored.audit_log[-1]
    assert (event.actor, event.event, event.details) == ("system", "offer_created", "48 months")
    assert event.at > stored.credit_check.checked_at


def test_other_endings_get_no_offer():
    no_answer = {"outcome": "cancelled", "reason": "no_answer", "explanation": "No reply."}
    stored = simulate("03_no_answer", [ask(), wait(), step("finish", no_answer)])
    assert stored.offers == []
    assert "offer_created" not in [e.event for e in stored.audit_log]


def test_offers_are_kept_on_file_in_postgres(pg_store):
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)
    first = offer_for(case.application)
    revised = create_offer(Decimal("17000"), 48)  # as after an officer's approval
    pg_store.record_offer(app_id, first, case.info.start_time)
    pg_store.record_offer(app_id, revised, case.info.start_time)
    assert pg_store.get(app_id).offers == [first, revised]
