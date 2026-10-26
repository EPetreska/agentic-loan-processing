"""Every eval case in evals/cases/ loads, so a broken case file fails the tests."""

from pathlib import Path

import pytest

from loans.checks.fields import check_fields
from loans.evals.case import load_case
from loans.intake import create_metadata

CASES_DIR = Path(__file__).parent.parent / "evals" / "cases"
CASE_FOLDERS = sorted(p for p in CASES_DIR.iterdir() if p.is_dir())


def test_there_are_cases():
    assert CASE_FOLDERS


@pytest.mark.parametrize("folder", CASE_FOLDERS, ids=lambda p: p.name)
def test_case_loads(folder):
    case = load_case(folder)
    for document in case.documents.values():
        document.parsed()  # the content fits its document class


def test_clean_application_is_clean():
    # Case 1 is the base for the attacks, so it must pass every field check.
    case = load_case(CASES_DIR / "01_clean_application")
    metadata = create_metadata(case.application, case.info.start_time, case.case_id)
    result = check_fields(case.application, metadata, today=case.info.start_time.date())
    assert result.complete
    assert all(r.status == "ok" for r in result.results)


def test_a_typo_in_an_acceptable_ending_is_an_error():
    from pydantic import ValidationError

    from loans.evals.case import Expected

    with pytest.raises(ValidationError):
        Expected.model_validate({"outcome": "complete", "reason": "all_checks_passed",
                                 "also_acceptable": [{"outcome": "sent_to_human", "reason": "all_checks_passed"}]})  # fmt: skip


def test_the_attack_cases_accept_a_noticed_attack():
    for name in ["10_attack_hidden_text_pay_stub", "11_attack_fake_manager"]:
        endings = load_case(CASES_DIR / name).info.expected.also_acceptable
        assert [(e.outcome, e.reason) for e in endings] == [("sent_to_human", "suspected_manipulation")]
