"""Tests for outcomes and the eval case format."""

import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from loans.evals.case import load_case
from loans.schemas.outcome import REASONS_BY_OUTCOME, Outcome, Reason

# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_every_reason_belongs_to_exactly_one_outcome():
    all_reasons = [r for reasons in REASONS_BY_OUTCOME.values() for r in reasons]
    assert sorted(all_reasons) == sorted(get_args(Reason))


@pytest.mark.parametrize(
    "outcome, reason",
    [
        ("complete", "all_checks_passed"),
        ("cancelled", "no_answer"),
        ("sent_to_human", "agent_unsure"),
    ],
)
def test_valid_outcome(outcome, reason):
    assert Outcome(outcome=outcome, reason=reason).reason == reason


@pytest.mark.parametrize(
    "outcome, reason",
    [
        ("complete", "no_answer"),
        ("cancelled", "under_18"),
        ("sent_to_human", "all_checks_passed"),
        ("complete", "made_up_reason"),
    ],
)
def test_reason_must_fit_outcome(outcome, reason):
    with pytest.raises(ValidationError):
        Outcome(outcome=outcome, reason=reason)


# ---------------------------------------------------------------------------
# Eval case folders
# ---------------------------------------------------------------------------

PDF = {"filename": "doc.pdf", "content_type": "application/pdf"}

PAY_STUB = {
    "actual_type": "pay_stub",
    "content": {
        "employer_name": "Brightline Health Services LLC",
        "employee_name": "Maria Lopez",
        "pay_date": "2026-08-26",
        "pay_period_start": "2026-08-10",
        "pay_period_end": "2026-08-23",
        "pay_frequency": "biweekly",
        "gross_pay": "2850.00",
        "net_pay": "2140.55",
        "ytd_gross": "45600.00",
    },
}

FILES = {
    "case.json": {
        "title": "Job change explained by the customer",
        "kind": "reasoning",
        "start_time": "2026-09-14T10:00:00Z",
        "expected": {
            "outcome": "complete",
            "reason": "all_checks_passed",
            "income_verified": True,
            "must_request": ["pay_stub"],
            "max_messages_to_customer": 1,
        },
    },
    "application.json": {
        "full_name": "Maria Lopez",
        "employment_status": "employed",
        "income_documents": {"pay_stub": PDF},
    },
    "customer.json": {
        "first_message": "I switched jobs in March.",
        "replies": [{"after_days": 3, "uploads": {"pay_stub": "pay_stub_new.json"}}],
    },
    "documents/pay_stub.json": PAY_STUB,
    "documents/pay_stub_new.json": PAY_STUB,
}


def write_case(folder: Path, files: dict) -> Path:
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    return folder


def edited(name: str, **changes) -> dict:
    """FILES with one file's top-level keys changed."""
    return FILES | {name: FILES[name] | changes}


def test_loads_a_valid_case(tmp_path):
    case = load_case(write_case(tmp_path / "maria", FILES))
    assert case.case_id == "maria"
    assert case.info.expected.reason == "all_checks_passed"
    assert case.application.full_name == "Maria Lopez"
    assert set(case.documents) == {"pay_stub.json", "pay_stub_new.json"}
    assert case.customer.replies[0].uploads == {"pay_stub": "pay_stub_new.json"}


def test_misspelled_key_is_an_error(tmp_path):
    expected = FILES["case.json"]["expected"] | {"max_messages_to_custmer": 1}
    files = edited("case.json", expected=expected)
    with pytest.raises(ValidationError, match="max_messages_to_custmer"):
        load_case(write_case(tmp_path / "c", files))


def test_expected_reason_must_fit_outcome(tmp_path):
    expected = FILES["case.json"]["expected"] | {"reason": "no_answer"}
    with pytest.raises(ValidationError):
        load_case(write_case(tmp_path / "c", edited("case.json", expected=expected)))


def test_start_time_needs_a_time_zone(tmp_path):
    files = edited("case.json", start_time="2026-09-14T10:00:00")
    with pytest.raises(ValidationError, match="time zone"):
        load_case(write_case(tmp_path / "c", files))


def test_upload_in_form_needs_its_document(tmp_path):
    files = {k: v for k, v in FILES.items() if k != "documents/pay_stub.json"}
    with pytest.raises(ValueError, match="missing documents/pay_stub.json"):
        load_case(write_case(tmp_path / "c", files))


def test_later_upload_needs_its_document(tmp_path):
    files = {k: v for k, v in FILES.items() if k != "documents/pay_stub_new.json"}
    with pytest.raises(ValueError, match="pay_stub_new.json"):
        load_case(write_case(tmp_path / "c", files))


def test_reply_can_only_change_real_form_fields(tmp_path):
    customer = {"replies": [{"changes": {"anual_income": "78000"}}]}
    with pytest.raises(ValidationError, match="anual_income"):
        load_case(write_case(tmp_path / "c", FILES | {"customer.json": customer}))


def test_customer_who_never_answers(tmp_path):
    case = load_case(write_case(tmp_path / "c", FILES | {"customer.json": {}}))
    assert case.customer.replies == []
