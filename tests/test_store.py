"""Tests for the application store."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from loans.checks.fields import check_fields
from loans.evals.case import load_case, submit_case
from loans.schemas.application import RawApplication, RawFile
from loans.schemas.documents import UploadedDocument
from loans.store import ApplicationStore, UnknownApplication, filled_slots

CASES = Path(__file__).parent.parent / "evals" / "cases"
NOW = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
PDF = {"filename": "doc.pdf", "content_type": "application/pdf"}
STUB = UploadedDocument(actual_type="pay_stub")


def test_submit_and_get():
    store = ApplicationStore()
    raw = RawApplication(full_name="Ann", income_documents={"pay_stub": PDF})
    app_id = store.submit(raw, {"pay_stub": STUB}, NOW, application_id="A-1")
    stored = store.get(app_id)
    assert stored.raw.full_name == "Ann"
    assert stored.metadata.submitted_at == NOW
    assert stored.documents == {"pay_stub": STUB}


def test_generated_id_when_none_given():
    store = ApplicationStore()
    app_id = store.submit(RawApplication(), {}, NOW)
    assert store.get(app_id).application_id == app_id


def test_unknown_application():
    with pytest.raises(UnknownApplication):
        ApplicationStore().get("nope")


def test_duplicate_id_is_rejected():
    store = ApplicationStore()
    store.submit(RawApplication(), {}, NOW, application_id="A-1")
    with pytest.raises(ValueError, match="already exists"):
        store.submit(RawApplication(), {}, NOW, application_id="A-1")


@pytest.mark.parametrize(
    "income_documents, documents",
    [
        ({"pay_stub": PDF}, {}),  # a filled slot without its document
        ({}, {"pay_stub": STUB}),  # a document for an empty slot
    ],
)
def test_documents_must_match_filled_slots(income_documents, documents):
    raw = RawApplication(income_documents=income_documents)
    with pytest.raises(ValueError, match="do not match"):
        ApplicationStore().submit(raw, documents, NOW)


def test_upload_updates_document_and_form_slot():
    store = ApplicationStore()
    app_id = store.submit(RawApplication(), {}, NOW)
    store.upload(app_id, "w2", UploadedDocument(actual_type="w2"), RawFile(**PDF), now=NOW)
    stored = store.get(app_id)
    assert stored.documents["w2"].actual_type == "w2"
    assert filled_slots(stored.raw) == ["w2"]


def test_upload_replaces_an_earlier_document():
    store = ApplicationStore()
    raw = RawApplication(income_documents={"pay_stub": PDF})
    app_id = store.submit(raw, {"pay_stub": STUB}, NOW)
    new = UploadedDocument(actual_type="pay_stub", readable=False)
    store.upload(app_id, "pay_stub", new, RawFile(filename="new.pdf", content_type="application/pdf"), now=NOW)
    stored = store.get(app_id)
    assert stored.documents["pay_stub"] is new
    assert stored.raw.income_documents.pay_stub.filename == "new.pdf"


def test_correct_fields():
    store = ApplicationStore()
    app_id = store.submit(RawApplication(annual_income="78k"), {}, NOW)
    store.correct_fields(app_id, {"annual_income": "78000"}, now=NOW)
    assert store.get(app_id).raw.annual_income == "78000"


def test_correct_unknown_field_is_rejected():
    store = ApplicationStore()
    app_id = store.submit(RawApplication(), {}, NOW)
    with pytest.raises(ValueError, match="anual_income"):
        store.correct_fields(app_id, {"anual_income": "78000"}, now=NOW)


@pytest.mark.parametrize("folder", sorted(p for p in CASES.iterdir() if p.is_dir()), ids=lambda p: p.name)
def test_every_case_can_be_submitted(folder):
    store = ApplicationStore()
    case = load_case(folder)
    app_id = submit_case(store, case)
    assert app_id == case.case_id
    assert store.get(app_id).metadata.submitted_at == case.info.start_time


def test_maria_reply_upload_makes_income_documents_current():
    # The scripted reply in case 2 uploads her new pay stub into the same slot.
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    reply = case.customer.replies[0]
    name = reply.uploads["pay_stub"]
    file = RawFile(filename=name, content_type="application/pdf")
    store.upload(app_id, "pay_stub", case.documents[name], file, now=NOW)
    stub = store.get(app_id).documents["pay_stub"].parsed()
    assert str(stub.pay_date) == "2026-09-11"


def test_base_and_victim_live_side_by_side():
    store = ApplicationStore()
    base = submit_case(store, load_case(CASES / "01_clean_application"))
    victim = submit_case(store, load_case(CASES / "02_maria_job_change"))
    assert store.get(base).raw.full_name == "Daniel Kim"
    assert store.get(victim).raw.full_name == "Maria Lopez"


def test_store_feeds_check_fields():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    stored = store.get(submit_case(store, case))
    result = check_fields(stored.raw, stored.metadata, today=case.info.start_time.date())
    assert result.complete


def daniel_without(consent: str):
    """Daniel's case, submitted with one consent answered "no"."""
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    raw = case.application.model_copy(update={consent: "no"})
    documents = {slot: case.documents[f"{slot}.json"] for slot in filled_slots(raw)}
    app_id = store.submit(raw, documents, case.info.start_time, application_id=case.case_id)
    return store, app_id, case


def test_credit_consent_given_later_is_timestamped():
    store, app_id, case = daniel_without("credit_check_consent")
    start = case.info.start_time
    assert store.get(app_id).metadata.credit_check_consented_at is None
    later = start + timedelta(days=2)
    store.correct_fields(app_id, {"credit_check_consent": "yes"}, now=later)
    stored = store.get(app_id)
    assert stored.metadata.credit_check_consented_at == later
    assert check_fields(stored.raw, stored.metadata, today=later.date()).complete


def test_esign_consent_given_later_is_timestamped():
    store, app_id, case = daniel_without("esign_consent")
    later = case.info.start_time + timedelta(days=1)
    store.correct_fields(app_id, {"esign_consent": "yes"}, now=later)
    assert store.get(app_id).metadata.esign_consented_at == later


def test_consent_given_again_keeps_its_first_time():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    app_id = submit_case(store, case)
    later = case.info.start_time + timedelta(days=3)
    store.correct_fields(app_id, {"credit_check_consent": "yes"}, now=later)
    assert store.get(app_id).metadata.credit_check_consented_at == case.info.start_time


def test_consent_taken_back_loses_its_time():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    app_id = submit_case(store, case)
    store.correct_fields(app_id, {"credit_check_consent": "no"}, now=case.info.start_time)
    stored = store.get(app_id)
    assert stored.metadata.credit_check_consented_at is None
    assert not check_fields(stored.raw, stored.metadata, today=case.info.start_time.date()).complete


def test_other_corrections_leave_the_consent_times_alone():
    store, app_id, case = daniel_without("credit_check_consent")
    store.correct_fields(app_id, {"annual_income": "72000"}, now=case.info.start_time)
    assert store.get(app_id).metadata.credit_check_consented_at is None
