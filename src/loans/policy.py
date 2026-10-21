"""The bank's process rules, in one place.

These are frame rules enforced in code: the agent cannot change them, and the
evals check them. Each value notes where it comes from.
"""

from decimal import Decimal

# A customer who does not answer is cancelled only after this many days
# without an answer, counted from the agent's last follow-up message, never
# earlier. From BPI Challenge 2017: the bank cancelled applications
# automatically about 26 days after the last follow-up call.
NO_ANSWER_CANCEL_DAYS = 26

# The first request for missing information must be sent within this many
# days of receiving the application. Regulation B (12 CFR 1002.9) requires a
# lender to notify the applicant within 30 days of receiving an incomplete
# application.
INCOMPLETE_NOTICE_DAYS = 30

# Income verification. For employed applicants a pay stub or a W-2 can verify
# the claimed income; for self-employed applicants, a tax return. The rules
# are applied in src/loans/checks/income_verification.py.

# A pay stub counts only if its pay date is at most this many days old. A
# common lender requirement for "current" pay stubs.
PAY_STUB_MAX_AGE_DAYS = 30

# A W-2 counts only if it is for the most recent tax year: in 2026, the 2025
# W-2. Employers issue W-2s by the end of January, so a W-2 is always months
# old and the pay stub age rule cannot apply to it.
W2_MAX_YEARS_OLD = 1

# A tax return (Form 1040 with Schedule C) counts only if it is for the most
# recent tax year, like a W-2.
TAX_RETURN_MAX_YEARS_OLD = 1

# The yearly income shown by the document (the annualized pay stub, the W-2
# wages, or the tax return's net profit) must be within this fraction of the
# claimed income. Exactly 5% counts as within.
INCOME_TOLERANCE = Decimal("0.05")

# Loan offers. After the credit check, the bank offers the amount and the
# term the customer asked for (src/loans/offers.py). The yearly rate for each
# term the form allows: illustrative rates for a borrower with good credit,
# slightly higher for longer terms, as at real lenders. There are no fees, so
# the APR equals the rate.
OFFER_RATES = {
    12: Decimal("0.0849"),
    24: Decimal("0.0899"),
    36: Decimal("0.0999"),
    48: Decimal("0.1099"),
    60: Decimal("0.1199"),
}

# The loan's terms, as the bank states them to customers in its terms and
# FAQ document (src/loans/rag/bank_terms.py), which the offer agent searches.

# A payment counts as late if it has not been received this many days after
# its due date; the late fee is then charged once for that payment.
LATE_PAYMENT_GRACE_DAYS = 15
LATE_FEE = Decimal("25")

# Paying the loan off early, in part or in full, costs nothing extra.
PREPAYMENT_PENALTY = Decimal("0")

# The customer may move the monthly due date at most this many times in any
# 12 months.
DUE_DATE_CHANGES_PER_YEAR = 1

# An offer can be accepted (e-signed) for this many days after it was sent.
OFFER_VALID_DAYS = 30
