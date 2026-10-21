"""The bank's personal loan terms and FAQ: the bank's own document for the
offer agent's RAG.

It answers the questions customers ask about this bank's loan (the late fee,
paying early, changing the due date, autopay, hardship), which general
consumer guides cannot. Every number comes from the code (policy.py and the
application form's limits), so the document always says what the offers and
the rules do; a test checks that the committed file is up to date.

Each topic and each FAQ question has its own heading, so the document splits
naturally into sections for the search.

To write the file after a change to the numbers:

    uv run python -m loans.rag.bank_terms
"""

from decimal import Decimal
from pathlib import Path

from loans import policy
from loans.schemas.application import ALLOWED_TERMS, MAX_LOAN_AMOUNT, MIN_LOAN_AMOUNT

PATH = Path(__file__).resolve().parents[3] / "documents" / "bank" / "loan_terms.md"


def _money(amount: Decimal) -> str:
    return f"${amount:,.0f}" if amount == amount.to_integral() else f"${amount:,.2f}"


def _percent(rate: Decimal) -> str:
    return f"{rate * 100:.2f}%"


def render() -> str:
    """The document as Markdown."""
    terms = sorted(ALLOWED_TERMS)
    term_list = ", ".join(str(t) for t in terms[:-1]) + f" or {terms[-1]}"
    rate_rows = "\n".join(
        f"| {t} months | {_percent(policy.OFFER_RATES[t])} |" for t in terms
    )
    lowest, highest = policy.OFFER_RATES[terms[0]], policy.OFFER_RATES[terms[-1]]
    late_fee = _money(policy.LATE_FEE)
    grace = policy.LATE_PAYMENT_GRACE_DAYS
    valid = policy.OFFER_VALID_DAYS
    changes = policy.DUE_DATE_CHANGES_PER_YEAR
    changes_text = "once" if changes == 1 else f"{changes} times"
    if policy.PREPAYMENT_PENALTY == 0:
        penalty = "There is no prepayment penalty: paying early never costs a fee."
        fees = (
            "There is no origination fee and no prepayment penalty. The only fee is the\n"
            f"late fee of {late_fee}, charged if a payment is more than {grace} days late."
        )
        early = "Yes, at any time and without a fee."
    else:
        prepayment_fee = _money(policy.PREPAYMENT_PENALTY)
        penalty = f"Paying early costs a prepayment fee of {prepayment_fee}."
        fees = (
            f"There is no origination fee. There is a prepayment fee of {prepayment_fee},\n"
            f"and a late fee of {late_fee}, charged if a payment is more than {grace} days late."
        )
        early = f"Yes, at any time, with a prepayment fee of {prepayment_fee}."

    return f"""# Personal Loan Terms and Frequently Asked Questions

This document describes the terms of our personal loans. The offer you receive
shows the exact amount, rate and payments of your own loan.

## The loan

Our personal loan is an unsecured installment loan: you do not need to put up
any property as security. You borrow a fixed amount and repay it in equal
monthly payments over a fixed term. You can borrow from {_money(MIN_LOAN_AMOUNT)}
to {_money(MAX_LOAN_AMOUNT)}, over {term_list} months.

## Interest rates and APR

The rate is fixed: it does not change for the life of the loan. We charge no
origination fee or other loan fees, so the annual percentage rate (APR) is the
same as the interest rate. The rate depends on the term:

| Term | APR |
|---|---|
{rate_rows}

A longer term means lower monthly payments but more interest in total.

## Your offer

Your offer shows the amount, the APR, the monthly payment, the number of
payments, the total of all payments and the finance charge (the total
interest you will pay). The offer is valid for {valid} days after we send it.
You accept it by signing it electronically in your online account; a reply
by message does not accept the offer. If you would like different terms, for
example another amount or term, ask us: a loan officer reviews every change.

## Monthly payments

Your payment is due on the same day each month and is the same amount every
month. You can move your due date {changes_text} in any 12 months. You can pay
by automatic payment (autopay) from your bank account, or pay each month
yourself. Autopay is a convenience: it does not change your rate.

## Late payments

If we have not received a payment {grace} days after its due date, we charge a
late fee of {late_fee} for that payment. The fee is charged once per late
payment. Late payments may also be reported to the credit bureaus and can
lower your credit score.

## Paying early

You can pay extra or pay off the whole loan at any time. {penalty} When you pay
early, you pay interest only for the time you borrowed the money.

## Financial hardship

If you lose your job, fall ill or have another hardship that makes it hard to
pay, contact our Loan Support team as soon as possible, ideally before you
miss a payment. We may be able to agree a temporary payment arrangement. Every
arrangement is reviewed and approved by a loan officer.

## Frequently asked questions

### What is the difference between the interest rate and the APR?

The APR is the yearly cost of the loan including fees. Because we charge no
loan fees, your APR is the same as your interest rate.

### Can my rate change during the loan?

No. The rate is fixed for the whole term.

### Are there any fees?

{fees}

### Can I pay off my loan early?

{early} You pay interest only for the time you borrowed
the money.

### What happens if I pay late?

If a payment is more than {grace} days late, we charge a late fee of
{late_fee}. Late payments may be reported to the credit bureaus.

### Can I change my payment due date?

Yes, {changes_text} in any 12 months. Contact Loan Support to move it.

### Do I get a lower rate with autopay?

No. Autopay makes paying easier, but your rate stays the one in your offer.

### How long is my offer valid?

{valid} days after we send it. After that, it expires and can no longer be
accepted.

### How do I accept my offer?

By signing it electronically in your online account. Replying to a message
does not accept the offer.

### Can I borrow a different amount or choose a different term?

You can ask. A loan officer reviews every change to an offer and lets you
know the decision.

### What if I can't make a payment?

Contact Loan Support as soon as possible, ideally before the payment is due.
We may be able to agree a temporary payment arrangement, reviewed by a loan
officer.

### Which term has the lowest rate?

The shortest term. Rates go from {_percent(lowest)} for {terms[0]} months to
{_percent(highest)} for {terms[-1]} months.
"""


def write(path: Path = PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render())


if __name__ == "__main__":
    write()
    print(f"written: {PATH}")
