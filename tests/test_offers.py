"""Tests for the offers: the terms, the loan math, and the workflow creating
them only after a passed credit check."""

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from loans import policy
from loans.evals.case import load_case, submit_case
from loans.evals.simulate import simulate_case
from loans.offers import create_offers, monthly_payment, offer_terms, offers_for
from loans.schemas.application import ALLOWED_TERMS
from loans.store import ApplicationStore
from test_simulate import MARIA, FakeModel, ask, step, wait

CASES = Path(__file__).parent.parent / "evals" / "cases"


def test_maria_gets_three_offers_with_the_right_numbers():
    # $15,000 asked over 48 months. Values from the standard loan formula.
    offers = create_offers(Decimal("15000"), 48)
    rows = [
        (o.term_months, o.apr, o.monthly_payment, o.total_of_payments, o.finance_charge)
        for o in offers
    ]
    assert rows == [
        (36, Decimal("0.0999"), Decimal("483.94"), Decimal("17421.84"), Decimal("2421.84")),
        (48, Decimal("0.1099"), Decimal("387.61"), Decimal("18605.28"), Decimal("3605.28")),
        (60, Decimal("0.1199"), Decimal("333.59"), Decimal("20015.40"), Decimal("5015.40")),
    ]
    assert all(o.amount == Decimal("15000") for o in offers)


@pytest.mark.parametrize(
    "asked, terms",
    [(12, [12, 24, 36]), (24, [12, 24, 36]), (36, [24, 36, 48]), (48, [36, 48, 60]), (60, [36, 48, 60])],
)
def test_the_asked_for_term_and_its_neighbors(asked, terms):
    assert offer_terms(asked) == terms
    assert asked in terms


def test_every_allowed_term_has_a_rate():
    assert set(policy.OFFER_RATES) == ALLOWED_TERMS


def test_a_longer_term_costs_more_in_total_but_less_a_month():
    offers = create_offers(Decimal("15000"), 48)
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


def test_offers_come_from_the_form():
    case = load_case(CASES / "02_maria_job_change")
    assert [o.term_months for o in offers_for(case.application)] == [36, 48, 60]


def simulate(case_name, answers):
    store = ApplicationStore()
    case = load_case(CASES / case_name)
    result = asyncio.run(simulate_case(case, store, FakeModel(messages=iter(answers))))
    return store.get(result.application_id)


def test_a_complete_case_gets_its_offers():
    stored = simulate("02_maria_job_change", MARIA)
    assert [o.term_months for o in stored.offers] == [36, 48, 60]
    event = stored.audit_log[-1]
    assert (event.actor, event.event, event.details) == ("system", "offers_created", "36, 48, 60 months")
    assert event.at > stored.credit_check.checked_at


def test_other_endings_get_no_offers():
    no_answer = {"outcome": "cancelled", "reason": "no_answer", "explanation": "No reply."}
    stored = simulate("03_no_answer", [ask(), wait(), step("finish", no_answer)])
    assert stored.offers == []
    assert "offers_created" not in [e.event for e in stored.audit_log]


def test_offers_are_kept_on_file_in_postgres(pg_store):
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)
    offers = offers_for(case.application)
    pg_store.record_offers(app_id, offers, case.info.start_time)
    assert pg_store.get(app_id).offers == offers
