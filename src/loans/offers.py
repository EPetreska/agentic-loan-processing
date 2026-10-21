"""create_offers: the bank's loan offers for a completed application.

After the credit check passes, the bank's workflow (code, never the LLM)
creates three offers for the amount the customer asked for: the term they
asked for, plus the next shorter and the next longer term the form allows. At
the edges the nearest terms are taken, so 12 months asked gives 12, 24 and
36. Each term has its rate from policy.OFFER_RATES.

For each offer the code calculates what truth-in-lending rules require a
lender to show: the APR (with no fees, the rate itself), the monthly payment
(the standard loan formula, in exact decimals, rounded to cents), the total
of payments and the finance charge (the interest cost). Every payment is the
same; a real lender would adjust the last one for the rounding, a
simplification here.

The three offers are the whole allowed space: the offer agent may present,
compare and recommend them, and anything else goes to a loan officer. The
agent never calculates an offer.
"""

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, TypeAdapter

from loans import policy
from loans.schemas.application import ALLOWED_TERMS, CleanApplication, RawApplication

CENT = Decimal("0.01")

_LOAN_AMOUNT = TypeAdapter(CleanApplication.model_fields["loan_amount"].rebuild_annotation())
_TERM = TypeAdapter(CleanApplication.model_fields["term_months"].rebuild_annotation())


class Offer(BaseModel):
    amount: Decimal
    term_months: int
    # The yearly rate. With no fees, also the APR.
    apr: Decimal
    monthly_payment: Decimal
    total_of_payments: Decimal
    # The interest cost: the total of payments minus the amount.
    finance_charge: Decimal


def offer_terms(requested: int) -> list[int]:
    """The asked-for term and its neighbors among the allowed terms."""
    terms = sorted(ALLOWED_TERMS)
    start = min(max(terms.index(requested) - 1, 0), len(terms) - 3)
    return terms[start : start + 3]


def monthly_payment(amount: Decimal, apr: Decimal, months: int) -> Decimal:
    """The fixed monthly payment that pays off the amount with interest."""
    rate = apr / 12
    payment = amount * rate / (1 - (1 + rate) ** -months)
    return payment.quantize(CENT, rounding=ROUND_HALF_UP)


def create_offers(amount: Decimal, requested_term: int) -> list[Offer]:
    """The three offers, shortest term first."""
    offers = []
    for months in offer_terms(requested_term):
        apr = policy.OFFER_RATES[months]
        payment = monthly_payment(amount, apr, months)
        total = payment * months
        offers.append(
            Offer(
                amount=amount,
                term_months=months,
                apr=apr,
                monthly_payment=payment,
                total_of_payments=total,
                finance_charge=total - amount,
            )
        )
    return offers


def offers_for(form: RawApplication) -> list[Offer]:
    """The offers for a completed application's form. Its amount and term
    have passed the form check, so they are read as checked values."""
    amount = _LOAN_AMOUNT.validate_python(form.loan_amount)
    term = _TERM.validate_python(form.term_months)
    return create_offers(amount, term)
