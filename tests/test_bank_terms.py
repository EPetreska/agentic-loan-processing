"""Tests for the bank's terms and FAQ document: it states exactly what the
code does, and the committed file is up to date."""

from decimal import Decimal

from loans import policy
from loans.rag import bank_terms
from loans.schemas.application import ALLOWED_TERMS


def test_the_committed_document_is_up_to_date():
    # After changing a number in policy.py, rewrite the document:
    #     uv run python -m loans.rag.bank_terms
    assert bank_terms.PATH.read_text() == bank_terms.render()


def test_every_term_and_its_rate_is_listed():
    text = bank_terms.render()
    for months in ALLOWED_TERMS:
        assert f"| {months} months | {policy.OFFER_RATES[months] * 100:.2f}% |" in text


def test_the_terms_state_the_policy_numbers():
    text = bank_terms.render()
    assert "late fee of $25" in text
    assert "15 days after its due date" in text
    assert "no prepayment penalty" in text
    assert "valid for 30 days" in text
    assert "move your due date once in any 12 months" in text
    assert "from $1,000\nto $50,000" in text


def test_a_changed_number_changes_the_document(monkeypatch):
    monkeypatch.setattr(policy, "LATE_FEE", Decimal("30"))
    monkeypatch.setattr(policy, "PREPAYMENT_PENALTY", Decimal("100"))
    text = bank_terms.render()
    assert "late fee of $30" in text and "late fee of $25" not in text
    assert "prepayment fee of $100" in text and "no prepayment penalty" not in text


def test_each_faq_question_is_its_own_section():
    # One heading per question, so the search can return one answer.
    questions = [line for line in bank_terms.render().splitlines() if line.startswith("### ")]
    assert len(questions) >= 8
    assert all(q.endswith("?") for q in questions)
