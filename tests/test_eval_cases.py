"""Every eval case in evals/cases/ loads, so a broken case file fails the tests."""

from pathlib import Path

import pytest

from loans.evals.case import load_case
from loans.intake import create_metadata
from loans.tools.check_fields import check_fields

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
