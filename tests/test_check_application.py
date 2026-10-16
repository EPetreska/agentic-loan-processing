"""Tests for the check_application tool."""

from pathlib import Path

import pytest

from loans.evals.case import EvalCase, load_case, submit_case
from loans.schemas.application import RawFile
from loans.store import ApplicationStore, UnknownApplication
from loans.tools.check_application import check_application

CASES = Path(__file__).parent.parent / "evals" / "cases"


def submitted(name: str) -> tuple[ApplicationStore, str, EvalCase]:
    store = ApplicationStore()
    case = load_case(CASES / name)
    return store, submit_case(store, case), case


def test_clean_application_is_complete():
    store, app_id, case = submitted("01_clean_application")
    text = check_application(store, app_id, case.info.start_time.date())
    assert text.splitlines()[0] == "complete: yes (0 errors, 0 warnings)"


def test_missing_income_document_is_reported():
    store, app_id, case = submitted("03_no_answer")
    text = check_application(store, app_id, case.info.start_time.date())
    assert text.splitlines()[0] == "complete: no (1 error, 0 warnings)"
    assert "error: income_documents: a pay stub or W-2 is required" in text


def test_result_follows_a_later_upload():
    # Once the customer uploads a W-2 in case 3, the application is complete.
    store, app_id, case = submitted("03_no_answer")
    w2 = load_case(CASES / "01_clean_application").documents["w2.json"]
    store.upload(app_id, "w2", w2, RawFile(filename="w2.pdf", content_type="application/pdf"))
    text = check_application(store, app_id, case.info.start_time.date())
    assert text.startswith("complete: yes")


def test_result_follows_a_correction():
    store, app_id, case = submitted("01_clean_application")
    store.correct_fields(app_id, {"annual_income": "78k"})
    text = check_application(store, app_id, case.info.start_time.date())
    assert 'error: annual_income: not a number (customer entered "78k")' in text
    store.correct_fields(app_id, {"annual_income": "78000"})
    assert check_application(store, app_id, case.info.start_time.date()).startswith("complete: yes")


@pytest.mark.parametrize(
    "case_name",
    ["01_clean_application", "02_maria_job_change", "03_no_answer", "04_retiree_other_employment"],
)
def test_sensitive_values_never_in_the_result(case_name):
    store, app_id, case = submitted(case_name)
    text = check_application(store, app_id, case.info.start_time.date())
    raw = case.application
    for secret in [
        raw.ssn,
        str(raw.ssn).replace("-", ""),
        raw.date_of_birth,
        raw.email,
        raw.phone,
        raw.address.street,
        raw.address.zip,
    ]:
        assert str(secret) not in text


def test_only_the_given_application_is_checked():
    # Two applications in the store: the result mentions only the requested one.
    store = ApplicationStore()
    daniel = submit_case(store, load_case(CASES / "01_clean_application"))
    submit_case(store, load_case(CASES / "02_maria_job_change"))
    text = check_application(store, daniel, load_case(CASES / "01_clean_application").info.start_time.date())
    assert '"Daniel Kim"' in text
    assert "Maria" not in text


def test_unknown_application():
    with pytest.raises(UnknownApplication):
        check_application(ApplicationStore(), "nope", load_case(CASES / "01_clean_application").info.start_time.date())
