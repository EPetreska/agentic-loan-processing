"""create_offer: the bank's loan offer for a completed application.

After the credit check passes, the bank's workflow (code, never the LLM)
creates one offer: the amount and the term the customer asked for, at the
term's rate from policy.OFFER_RATES. A change the customer asks for later
goes to a loan officer; when the officer approves, code creates the revised
offer the same way.

For the offer the code calculates what truth-in-lending rules require a
lender to show: the APR (with no fees, the rate itself), the monthly payment
(the standard loan formula, in exact decimals, rounded to cents), the total
of payments and the finance charge (the interest cost). Every payment is the
same; a real lender would adjust the last one for the rounding, a
simplification here. The agent never calculates an offer.
"""

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, TypeAdapter

from loans import policy
from loans.schemas.application import CleanApplication, RawApplication

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


def monthly_payment(amount: Decimal, apr: Decimal, months: int) -> Decimal:
    """The fixed monthly payment that pays off the amount with interest."""
    rate = apr / 12
    payment = amount * rate / (1 - (1 + rate) ** -months)
    return payment.quantize(CENT, rounding=ROUND_HALF_UP)


def create_offer(amount: Decimal, term_months: int) -> Offer:
    """The offer for this amount and term, at the term's rate."""
    apr = policy.OFFER_RATES[term_months]
    payment = monthly_payment(amount, apr, term_months)
    total = payment * term_months
    return Offer(
        amount=amount,
        term_months=term_months,
        apr=apr,
        monthly_payment=payment,
        total_of_payments=total,
        finance_charge=total - amount,
    )


def offer_for(form: RawApplication) -> Offer:
    """The offer for a completed application's form: the amount and term
    asked for. Both have passed the form check, so they are read as checked
    values."""
    amount = _LOAN_AMOUNT.validate_python(form.loan_amount)
    term = _TERM.validate_python(form.term_months)
    return create_offer(amount, term)
