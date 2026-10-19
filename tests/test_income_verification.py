"""Tests for the bank's income rules."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from loans.checks.income_verification import check_income_document
from loans.evals.case import load_case
from loans.schemas.documents import UploadedDocument

CASES = Path(__file__).parent.parent / "evals" / "cases"
DANIEL = load_case(CASES / "01_clean_application")
MARIA = load_case(CASES / "02_maria_job_change")
DANIEL_DAY = DANIEL.info.start_time.date()  # 2026-09-14
MARIA_DAY = MARIA.info.start_time.date()  # 2026-09-14


def check(case, file, slot, today, **form_changes):
    raw = case.application.model_copy(update=form_changes)
    return check_income_document(case.documents[file], slot, raw, today)


def tax_return(net_profit="58000", gross_receipts="120000", tax_year=2025):
    return UploadedDocument(
        actual_type="tax_return",
        content={
            "tax_year": tax_year,
            "taxpayer_name": "Daniel Kim",
            "business_name": "Kim Consulting",
            "gross_receipts": gross_receipts,
            "net_profit": net_profit,
        },
    )


def check_tax_return(document, today=DANIEL_DAY, **form_changes):
    form = {"employment_status": "self_employed", "annual_income": "60000"} | form_changes
    raw = DANIEL.application.model_copy(update=form)
    return check_income_document(document, "tax_return", raw, today)


@pytest.mark.parametrize("file, slot", [("pay_stub.json", "pay_stub"), ("w2.json", "w2")])
def test_daniel_documents_meet_the_rules(file, slot):
    assert check(DANIEL, file, slot, DANIEL_DAY).met


def test_maria_old_pay_stub_is_too_old():
    # $74,100 is exactly 5% below her claimed $78,000, which counts as within:
    # only the age fails.
    result = check(MARIA, "pay_stub.json", "pay_stub", MARIA_DAY)
    assert result.not_met == ["too old: the pay date was 45 days ago (the limit is 30 days)"]


def test_maria_w2_is_too_far_below():
    result = check(MARIA, "w2.json", "w2", MARIA_DAY)
    assert result.not_met == [
        "the yearly income on the document (61200.00) is 21.5% below "
        "the claimed income (78000) (the limit is 5%)"
    ]


def test_maria_new_pay_stub_meets_the_rules():
    assert check(MARIA, "pay_stub_new.json", "pay_stub", MARIA_DAY + timedelta(days=2)).met


@pytest.mark.parametrize(
    "claimed, met",
    [
        ("72000", True),
        ("75780", True),  # 71,999.98 is 4.99% below
        ("75800", False),  # 5.01% below
        ("68600", True),  # 4.96% above
        ("68500", False),  # 5.11% above
    ],
)
def test_five_percent_limit(claimed, met):
    result = check(DANIEL, "pay_stub.json", "pay_stub", DANIEL_DAY, annual_income=claimed)
    assert result.met is met


def test_income_above_the_claim_is_reported_as_above():
    result = check(DANIEL, "pay_stub.json", "pay_stub", DANIEL_DAY, annual_income="60000")
    assert result.not_met == [
        "the yearly income on the document (71999.98) is 20.0% above "
        "the claimed income (60000) (the limit is 5%)"
    ]


@pytest.mark.parametrize("days, met", [(30, True), (31, False)])
def test_pay_stub_age_limit(days, met):
    # Daniel's pay stub is dated 2026-09-04.
    today = date(2026, 9, 4) + timedelta(days=days)
    assert check(DANIEL, "pay_stub.json", "pay_stub", today).met is met


def test_w2_from_two_tax_years_ago_is_too_old():
    result = check(DANIEL, "w2.json", "w2", date(2027, 2, 1))
    assert result.not_met == ["too old: tax year 2025 (tax year 2026 or later is required)"]


def test_tax_return_meets_the_rules():
    assert check_tax_return(tax_return()).met


def test_tax_return_uses_net_profit_not_gross_receipts():
    result = check_tax_return(tax_return(net_profit="30000", gross_receipts="60000"))
    assert result.not_met == [
        "the yearly income on the document (30000) is 50.0% below "
        "the claimed income (60000) (the limit is 5%)"
    ]


def test_old_tax_return_is_too_old():
    result = check_tax_return(tax_return(tax_year=2024))
    assert result.not_met == ["too old: tax year 2024 (tax year 2025 or later is required)"]


def test_tax_return_does_not_count_for_an_employed_applicant():
    result = check_tax_return(tax_return(), employment_status="employed")
    assert result.not_met == [
        "for an employed applicant, income is verified with a pay stub or a W-2"
    ]


def test_pay_stub_does_not_count_for_a_self_employed_applicant():
    result = check(
        DANIEL, "pay_stub.json", "pay_stub", DANIEL_DAY, employment_status="self_employed"
    )
    assert result.not_met == [
        "for a self-employed applicant, income is verified with a tax return"
    ]


def test_no_document_counts_for_other_employment():
    result = check(DANIEL, "pay_stub.json", "pay_stub", DANIEL_DAY, employment_status="other")
    assert result.not_met == [
        "income is verified with documents only for employed or self-employed applicants"
    ]


def test_invalid_employment_status():
    result = check(DANIEL, "pay_stub.json", "pay_stub", DANIEL_DAY, employment_status="retired")
    assert result.not_met == ["the employment status on the form is missing or not valid"]


def test_invalid_claimed_income():
    result = check(DANIEL, "pay_stub.json", "pay_stub", DANIEL_DAY, annual_income="72k")
    assert result.not_met == ["the claimed income on the form is missing or not valid"]


def test_document_of_the_wrong_type():
    raw = DANIEL.application
    result = check_income_document(DANIEL.documents["pay_stub.json"], "w2", raw, DANIEL_DAY)
    assert result.not_met == ["the document in this slot is not a w2"]


def test_unreadable_document():
    raw = DANIEL.application
    blurry = UploadedDocument(actual_type="pay_stub", readable=False)
    result = check_income_document(blurry, "pay_stub", raw, DANIEL_DAY)
    assert result.not_met == ["the document could not be read"]


def test_several_problems_are_all_listed():
    result = check(MARIA, "pay_stub.json", "pay_stub", MARIA_DAY, annual_income="90000")
    assert len(result.not_met) == 2
    assert result.to_llm_text().splitlines() == [
        "bank income rules: not met",
        "- too old: the pay date was 45 days ago (the limit is 30 days)",
        "- the yearly income on the document (74100.00) is 17.7% below "
        "the claimed income (90000) (the limit is 5%)",
    ]


def test_result_text_when_met():
    assert check(DANIEL, "w2.json", "w2", DANIEL_DAY).to_llm_text() == "bank income rules: met"
