"""Tests for the mark_income_verified tool."""

from datetime import timedelta
from pathlib import Path

from loans.evals.case import EvalCase, load_case, submit_case
from loans.schemas.application import RawFile
from loans.store import ApplicationStore
from loans.tools.application.mark_income_verified import mark_income_verified

CASES = Path(__file__).parent.parent / "evals" / "cases"
PDF = RawFile(filename="doc.pdf", content_type="application/pdf")
REASON = "The pay stub is from the employer on the form and matches the claimed income."


def submitted(name: str = "01_clean_application") -> tuple[ApplicationStore, str, EvalCase]:
    store = ApplicationStore()
    case = load_case(CASES / name)
    return store, submit_case(store, case), case


def mark(store, app_id, case, slot="pay_stub", call_id="call-1", reason=REASON, days=0):
    now = case.info.start_time + timedelta(days=days)
    return mark_income_verified(store, app_id, now, call_id, slot, reason)


def test_verification_is_kept_on_file():
    store, app_id, case = submitted()
    assert store.get(app_id).income_verification is None
    assert mark(store, app_id, case) == "income verified with the pay_stub (kept on file)"
    verification = store.get(app_id).income_verification
    assert verification.slot == "pay_stub"
    assert verification.reason == REASON
    assert verification.call_id == "call-1"
    assert verification.verified_on == case.info.start_time.date()
    assert store.get(app_id).income_verification_is_current()


def test_refused_when_a_rule_is_not_met():
    store, app_id, case = submitted("02_maria_job_change")
    text = mark(store, app_id, case, reason="The customer said she got a raise.")
    assert text == (
        "not verified:\n"
        "bank income rules: not met\n"
        "- too old: the pay date was 45 days ago (the limit is 30 days)"
    )
    assert store.get(app_id).income_verification is None


def test_maria_is_verified_after_the_new_pay_stub():
    store, app_id, case = submitted("02_maria_job_change")
    store.upload(app_id, "pay_stub", case.documents["pay_stub_new.json"], PDF, now=case.info.start_time)
    assert mark(store, app_id, case, days=2) == "income verified with the pay_stub (kept on file)"


def test_reason_is_required():
    store, app_id, case = submitted()
    assert mark(store, app_id, case, reason="  ") == "not verified: a reason is required"
    assert store.get(app_id).income_verification is None


def test_empty_slot():
    store, app_id, case = submitted("03_no_answer")
    assert mark(store, app_id, case) == "not verified: no file uploaded in the pay_stub slot"


def test_id_slot_cannot_verify_income():
    store, app_id, case = submitted()
    text = mark(store, app_id, case, slot="id_front")
    assert text == "not verified: only pay_stub, w2 and tax_return can verify income"
    assert store.get(app_id).income_verification is None


def test_retried_call_is_not_recorded_twice():
    store, app_id, case = submitted()
    first = mark(store, app_id, case, call_id="call-1")
    recorded = store.get(app_id).income_verification
    retry = mark(store, app_id, case, call_id="call-1", reason="something else")
    assert retry == first
    assert store.get(app_id).income_verification is recorded


def test_a_new_call_replaces_the_earlier_verification():
    store, app_id, case = submitted()
    mark(store, app_id, case, slot="pay_stub", call_id="call-1")
    mark(store, app_id, case, slot="w2", call_id="call-2")
    assert store.get(app_id).income_verification.slot == "w2"


def test_out_of_date_after_a_new_upload_of_the_same_document():
    store, app_id, case = submitted()
    mark(store, app_id, case)
    store.upload(app_id, "pay_stub", case.documents["pay_stub.json"], PDF, now=case.info.start_time)
    assert not store.get(app_id).income_verification_is_current()


def test_unaffected_by_uploads_of_other_documents():
    store, app_id, case = submitted()
    mark(store, app_id, case)
    store.upload(app_id, "w2", case.documents["w2.json"], PDF, now=case.info.start_time)
    assert store.get(app_id).income_verification_is_current()


def test_out_of_date_after_an_income_correction():
    store, app_id, case = submitted()
    mark(store, app_id, case)
    store.correct_fields(app_id, {"annual_income": "90000"}, now=case.info.start_time)
    assert not store.get(app_id).income_verification_is_current()


def test_no_verification_means_not_current():
    store, app_id, _ = submitted()
    assert not store.get(app_id).income_verification_is_current()


def test_only_the_current_application_is_marked():
    store = ApplicationStore()
    daniel_case = load_case(CASES / "01_clean_application")
    daniel = submit_case(store, daniel_case)
    maria = submit_case(store, load_case(CASES / "02_maria_job_change"))
    mark(store, daniel, daniel_case)
    assert store.get(daniel).income_verification is not None
    assert store.get(maria).income_verification is None
