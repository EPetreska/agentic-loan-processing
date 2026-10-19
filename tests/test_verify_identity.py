"""Tests for the verify_identity tool and the mock identity vendor."""

from pathlib import Path

import pytest

from loans.evals.case import EvalCase, load_case, submit_case
from loans.schemas.application import RawFile
from loans.schemas.documents import UploadedDocument
from loans.services.identity import normalize_name
from loans.store import ApplicationStore
from loans.tools.verify_identity import verify_identity

CASES = Path(__file__).parent.parent / "evals" / "cases"
JPG = RawFile(filename="id.jpg", content_type="image/jpeg")
ALL_YES = [
    "document_type_matches_declared: yes",
    "readable: yes",
    "expired: no",
    "name_matches_form: yes",
    "date_of_birth_matches_form: yes",
]


def submitted(name: str = "01_clean_application") -> tuple[ApplicationStore, str, EvalCase]:
    store = ApplicationStore()
    case = load_case(CASES / name)
    return store, submit_case(store, case), case


def verify(store, app_id, case) -> list[str]:
    return verify_identity(store, app_id, case.info.start_time.date()).splitlines()


def upload_id(store, app_id, case, **content_changes):
    """Replace the ID front with a copy of the case's ID, with some fields changed."""
    doc = case.documents["id_front.json"]
    changed = doc.model_copy(update={"content": doc.content | content_changes})
    store.upload(app_id, "id_front", changed, JPG)


@pytest.mark.parametrize(
    "case_name",
    ["01_clean_application", "02_maria_job_change", "03_no_answer", "04_retiree_other_employment"],
)
def test_every_case_has_a_matching_id(case_name):
    store, app_id, case = submitted(case_name)
    assert verify(store, app_id, case) == ALL_YES


def test_expired_id():
    store, app_id, case = submitted()
    upload_id(store, app_id, case, expiry_date="2025-12-31")
    assert "expired: yes" in verify(store, app_id, case)


def test_id_expiring_today_is_not_expired():
    store, app_id, case = submitted()
    upload_id(store, app_id, case, expiry_date=str(case.info.start_time.date()))
    assert "expired: no" in verify(store, app_id, case)


def test_wrong_type_of_id():
    # A passport uploaded where a driver's license was declared.
    store, app_id, case = submitted()
    doc = case.documents["id_front.json"]
    store.upload(app_id, "id_front", doc.model_copy(update={"actual_type": "passport"}), JPG)
    lines = verify(store, app_id, case)
    assert "document_type_matches_declared: no" in lines
    assert "name_matches_form: yes" in lines  # the passport itself can still be read


def test_pay_stub_in_the_id_slot():
    store, app_id, case = submitted()
    store.upload(app_id, "id_front", case.documents["pay_stub.json"], JPG)
    lines = verify(store, app_id, case)
    assert "document_type_matches_declared: no" in lines
    assert "name_matches_form: cannot check (the document is not an ID)" in lines


def test_unreadable_id():
    store, app_id, case = submitted()
    doc = case.documents["id_front.json"]
    store.upload(app_id, "id_front", doc.model_copy(update={"readable": False}), JPG)
    lines = verify(store, app_id, case)
    assert "readable: no" in lines
    assert "expired: cannot check (the document could not be read)" in lines


def test_name_difference():
    store, app_id, case = submitted()
    upload_id(store, app_id, case, full_name="Daniel Park")
    assert "name_matches_form: no" in verify(store, app_id, case)


@pytest.mark.parametrize(
    "name_on_id, matches",
    [
        ("DANIEL KIM", True),
        ("  daniel   kim ", True),
        ("Daniel Kim.", True),
        ("Daniel J Kim", False),  # a middle initial counts as a difference
        ("Dan Kim", False),
    ],
)
def test_name_formatting(name_on_id, matches):
    store, app_id, case = submitted()
    upload_id(store, app_id, case, full_name=name_on_id)
    expected = "yes" if matches else "no"
    assert f"name_matches_form: {expected}" in verify(store, app_id, case)


def test_normalize_name():
    assert normalize_name("Mary-Jane O'Brien") == "mary jane obrien"


def test_date_of_birth_difference():
    store, app_id, case = submitted()
    upload_id(store, app_id, case, date_of_birth="1988-04-23")
    assert "date_of_birth_matches_form: no" in verify(store, app_id, case)


def test_invalid_date_of_birth_on_the_form():
    store, app_id, case = submitted()
    store.correct_fields(app_id, {"date_of_birth": "22/04/1988"})
    lines = verify(store, app_id, case)
    assert (
        "date_of_birth_matches_form: cannot check "
        "(the date of birth on the form is missing or not valid)"
    ) in lines
    assert "name_matches_form: yes" in lines  # the other checks still run


def test_no_id_uploaded():
    store, app_id, case = submitted()
    store.get(app_id).documents.pop("id_front")
    assert verify_identity(store, app_id, case.info.start_time.date()) == (
        "no ID uploaded yet (the ID front slot is empty)"
    )


@pytest.mark.parametrize(
    "case_name",
    ["01_clean_application", "02_maria_job_change", "03_no_answer", "04_retiree_other_employment"],
)
def test_no_id_data_in_the_result(case_name):
    store, app_id, case = submitted(case_name)
    text = verify_identity(store, app_id, case.info.start_time.date())
    id_content = case.documents["id_front.json"].content
    for value in id_content.values():
        assert str(value) not in text
    assert case.application.full_name not in text


def test_findings_are_kept_on_file():
    store, app_id, case = submitted()
    assert store.get(app_id).identity_findings is None
    verify(store, app_id, case)
    findings = store.get(app_id).identity_findings
    assert findings.name_matches_form is True
    assert findings.checked_on == case.info.start_time.date()
    assert store.get(app_id).identity_findings_are_current()


def test_findings_out_of_date_after_a_new_id_upload():
    store, app_id, case = submitted()
    verify(store, app_id, case)
    upload_id(store, app_id, case)  # the same ID again, but a new upload
    assert not store.get(app_id).identity_findings_are_current()
    verify(store, app_id, case)
    assert store.get(app_id).identity_findings_are_current()


def test_findings_out_of_date_after_a_form_correction():
    store, app_id, case = submitted()
    verify(store, app_id, case)
    store.correct_fields(app_id, {"full_name": "Daniel J Kim"})
    assert not store.get(app_id).identity_findings_are_current()


def test_findings_unaffected_by_other_uploads():
    store, app_id, case = submitted()
    verify(store, app_id, case)
    store.upload(app_id, "pay_stub", case.documents["pay_stub.json"], JPG)
    assert store.get(app_id).identity_findings_are_current()


def test_no_findings_means_not_current():
    store, app_id, _ = submitted()
    assert not store.get(app_id).identity_findings_are_current()


def test_only_the_current_application_is_checked():
    store = ApplicationStore()
    daniel_case = load_case(CASES / "01_clean_application")
    daniel = submit_case(store, daniel_case)
    maria = submit_case(store, load_case(CASES / "02_maria_job_change"))
    verify(store, daniel, daniel_case)
    assert store.get(daniel).identity_findings is not None
    assert store.get(maria).identity_findings is None


def test_unsupported_upload_in_the_id_slot():
    store, app_id, case = submitted()
    store.upload(app_id, "id_front", UploadedDocument(actual_type="other"), JPG)
    lines = verify(store, app_id, case)
    assert "document_type_matches_declared: no" in lines
    assert "readable: yes" in lines
