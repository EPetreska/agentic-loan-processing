"""Tests for the finish tool."""

from datetime import timedelta
from pathlib import Path

import pytest

from loans.evals.case import EvalCase, load_case, submit_case
from loans.schemas.application import RawFile
from loans.checks.ending import last_contact
from loans.store import ApplicationStore, CustomerReply, filled_slots
from loans.tools.finish import finish
from loans.tools.mark_income_verified import mark_income_verified
from loans.tools.send_message import send_message
from loans.tools.verify_identity import verify_identity

CASES = Path(__file__).parent.parent / "evals" / "cases"
JPG = RawFile(filename="id.jpg", content_type="image/jpeg")
PDF = RawFile(filename="doc.pdf", content_type="application/pdf")
EXPLANATION = "All checks passed."


def submitted(name: str = "01_clean_application") -> tuple[ApplicationStore, str, EvalCase]:
    store = ApplicationStore()
    case = load_case(CASES / name)
    return store, submit_case(store, case), case


def ready(name: str = "01_clean_application") -> tuple[ApplicationStore, str, EvalCase]:
    """A case with identity checked and income verified, ready to complete."""
    store, app_id, case = submitted(name)
    now = case.info.start_time
    verify_identity(store, app_id, now)
    mark_income_verified(store, app_id, now, "call-income", "pay_stub", "Matches the form.")
    return store, app_id, case


def end(store, app_id, case, outcome="complete", reason="all_checks_passed",
        call_id="call-end", explanation=EXPLANATION, days=0):  # fmt: skip
    now = case.info.start_time + timedelta(days=days)
    return finish(store, app_id, now, call_id, outcome, reason, explanation)


def message(store, app_id, case, call_id="call-msg", days=0):
    now = case.info.start_time + timedelta(days=days)
    send_message(store, app_id, now, call_id, "Please upload a pay stub.", "No income document.")


def problems(text: str) -> list[str]:
    lines = text.splitlines()
    assert lines[0] == "not ended:"
    return [line.removeprefix("- ") for line in lines[1:]]


# complete


def test_complete_when_all_checks_passed():
    store, app_id, case = ready()
    assert end(store, app_id, case) == "case ended: complete (all_checks_passed)"
    ending = store.get(app_id).ending
    assert ending.outcome == "complete"
    assert ending.explanation == EXPLANATION
    assert ending.ended_at == case.info.start_time


def test_complete_refused_without_identity_and_income():
    store, app_id, case = submitted()
    assert problems(end(store, app_id, case)) == [
        "the identity findings are missing or out of date",
        "the income has not been verified",
    ]
    assert store.get(app_id).ending is None


def test_complete_refused_when_identity_findings_are_not_as_required():
    store, app_id, case = submitted()
    doc = case.documents["id_front.json"]
    changed = doc.model_copy(update={"content": doc.content | {"full_name": "Daniel Park"}})
    store.upload(app_id, "id_front", changed, JPG, now=case.info.start_time)
    now = case.info.start_time
    verify_identity(store, app_id, now)
    mark_income_verified(store, app_id, now, "call-income", "pay_stub", "Matches the form.")
    assert problems(end(store, app_id, case)) == ["identity: name_matches_form is not as required"]


def test_complete_refused_after_a_new_id_upload():
    store, app_id, case = ready()
    store.upload(app_id, "id_front", case.documents["id_front.json"], JPG, now=case.info.start_time)
    assert problems(end(store, app_id, case)) == [
        "the identity findings are missing or out of date"
    ]


def test_complete_refused_after_an_income_correction():
    store, app_id, case = ready()
    store.correct_fields(app_id, {"annual_income": "72500"}, now=case.info.start_time)
    assert problems(end(store, app_id, case)) == [
        # the form changed, so the identity findings are out of date too
        "the identity findings are missing or out of date",
        "the income verification is out of date (the document or the form changed)",
    ]


def test_complete_refused_when_the_form_has_errors():
    store, app_id, case = ready()
    store.correct_fields(app_id, {"phone": "123"}, now=case.info.start_time)
    verify_identity(store, app_id, case.info.start_time)
    assert "the form still has errors" in problems(end(store, app_id, case))


def test_complete_refused_for_an_applicant_under_18():
    store, app_id, case = submitted()
    # Born 17 years before the start, on both the form and the ID.
    today = case.info.start_time.date()
    born = today.replace(year=today.year - 17).isoformat()
    store.correct_fields(app_id, {"date_of_birth": born}, now=case.info.start_time)
    doc = case.documents["id_front.json"]
    changed = doc.model_copy(update={"content": doc.content | {"date_of_birth": born}})
    store.upload(app_id, "id_front", changed, JPG, now=case.info.start_time)
    verify_identity(store, app_id, case.info.start_time)
    mark_income_verified(store, app_id, case.info.start_time, "call-income", "pay_stub", "Matches the form.")
    assert problems(end(store, app_id, case)) == ["the applicant is under 18"]


def test_complete_refused_without_esign_consent():
    store, app_id, case = ready()
    store.get(app_id).metadata.esign_consented_at = None
    assert "the customer has not given e-sign consent" in problems(end(store, app_id, case))


def test_maria_completes_after_the_new_pay_stub():
    store, app_id, case = submitted("02_maria_job_change")
    now = case.info.start_time
    verify_identity(store, app_id, now)
    mark_income_verified(store, app_id, now, "call-1", "pay_stub", "She got a raise.")
    assert problems(end(store, app_id, case)) == ["the income has not been verified"]
    store.upload(app_id, "pay_stub", case.documents["pay_stub_new.json"], PDF, now=case.info.start_time)
    later = now + timedelta(days=2)
    mark_income_verified(store, app_id, later, "call-2", "pay_stub", "New stub at the new rate.")
    assert end(store, app_id, case, days=2) == "case ended: complete (all_checks_passed)"


# cancelled


def test_no_answer_after_26_days():
    store, app_id, case = submitted("03_no_answer")
    message(store, app_id, case, days=0)
    assert end(store, app_id, case, "cancelled", "no_answer", days=26) == (
        "case ended: cancelled (no_answer)"
    )


def test_no_answer_too_early():
    store, app_id, case = submitted("03_no_answer")
    message(store, app_id, case, days=0)
    text = end(store, app_id, case, "cancelled", "no_answer", days=25)
    assert problems(text) == ["too early: 25 days since the last message or reply (the minimum is 26 days)"]
    assert store.get(app_id).ending is None


def test_waiting_counts_from_the_last_message():
    store, app_id, case = submitted("03_no_answer")
    message(store, app_id, case, call_id="first", days=0)
    message(store, app_id, case, call_id="follow-up", days=10)
    text = end(store, app_id, case, "cancelled", "no_answer", days=30)
    assert problems(text) == ["too early: 20 days since the last message or reply (the minimum is 26 days)"]
    assert end(store, app_id, case, "cancelled", "no_answer", days=36).startswith("case ended")


def test_no_answer_without_any_message():
    store, app_id, case = submitted("03_no_answer")
    text = end(store, app_id, case, "cancelled", "no_answer", days=40)
    assert problems(text) == ["no message has been sent to the customer"]


def test_documents_not_provided_waits_too():
    store, app_id, case = submitted("03_no_answer")
    message(store, app_id, case)
    assert end(store, app_id, case, "cancelled", "documents_not_provided", days=3).startswith(
        "not ended"
    )


def reply(store, app_id, case, text="I don't need the loan anymore.", days=1, **changes):
    """The customer replies; with only uploads or corrections, text is None."""
    store.record_reply(
        app_id,
        CustomerReply(received_at=case.info.start_time + timedelta(days=days), message=text, **changes),
    )


def without_credit_consent(store, app_id, case):
    """The customer takes back the credit check consent on the form."""
    store.correct_fields(app_id, {"credit_check_consent": "no"}, now=case.info.start_time)


def test_withdrawal_after_the_customer_wrote():
    store, app_id, case = submitted("03_no_answer")
    reply(store, app_id, case)
    assert end(store, app_id, case, "cancelled", "customer_withdrew") == (
        "case ended: cancelled (customer_withdrew)"
    )


def test_withdrawal_in_the_message_with_the_application():
    store = ApplicationStore()
    case = load_case(CASES / "03_no_answer")
    raw, documents = case.application, {s: case.documents[f"{s}.json"] for s in filled_slots(case.application)}
    app_id = store.submit(raw, documents, case.info.start_time, first_message="Please cancel, I changed my mind.")
    assert end(store, app_id, case, "cancelled", "customer_withdrew").startswith("case ended")


def test_no_withdrawal_when_the_customer_never_wrote():
    store, app_id, case = submitted("03_no_answer")
    message(store, app_id, case)  # the agent wrote; the customer did not
    assert problems(end(store, app_id, case, "cancelled", "customer_withdrew")) == [
        "the customer has not written to the bank, so a withdrawal cannot be recorded"
    ]
    assert store.get(app_id).ending is None


@pytest.mark.parametrize(
    "changes",
    [{"uploads": ["pay_stub"]}, {"corrected_fields": ["phone"]}],
    ids=["upload only", "correction only"],
)
def test_a_reply_without_text_is_not_the_customer_writing(changes):
    store, app_id, case = submitted("03_no_answer")
    reply(store, app_id, case, text=None, **changes)
    assert end(store, app_id, case, "cancelled", "customer_withdrew").startswith("not ended")


def test_injected_text_alone_cannot_cancel_a_case():
    # The idea of the cancel_maria attack: text in a document tells the agent
    # the customer wants to cancel. Even if the agent believes it, the
    # customer never wrote, so the code refuses.
    store, app_id, case = submitted("03_no_answer")
    for reason in ["customer_withdrew", "credit_consent_refused"]:
        assert end(store, app_id, case, "cancelled", reason, call_id=reason).startswith("not ended")
    assert store.get(app_id).ending is None


def test_consent_refusal_with_the_form_and_a_message():
    store, app_id, case = submitted("03_no_answer")
    without_credit_consent(store, app_id, case)
    reply(store, app_id, case, "No, I don't want a credit check.")
    assert end(store, app_id, case, "cancelled", "credit_consent_refused") == (
        "case ended: cancelled (credit_consent_refused)"
    )


def test_no_consent_refusal_while_the_form_shows_consent():
    # The customer says no, but the bank's record still says yes.
    store, app_id, case = submitted("03_no_answer")
    reply(store, app_id, case, "Actually, I don't want the credit check.")
    assert problems(end(store, app_id, case, "cancelled", "credit_consent_refused")) == [
        "the form shows the credit check consent as given; only the customer can change it"
    ]


def test_no_consent_refusal_without_a_message():
    store, app_id, case = submitted("03_no_answer")
    without_credit_consent(store, app_id, case)
    assert problems(end(store, app_id, case, "cancelled", "credit_consent_refused")) == [
        "the customer has not written to the bank, so a refusal cannot be recorded"
    ]


def test_a_consent_refusal_lists_both_problems():
    store, app_id, case = submitted("03_no_answer")
    assert problems(end(store, app_id, case, "cancelled", "credit_consent_refused")) == [
        "the form shows the credit check consent as given; only the customer can change it",
        "the customer has not written to the bank, so a refusal cannot be recorded",
    ]


# sent_to_human


def test_linda_is_sent_to_a_human():
    store, app_id, case = submitted("04_retiree_other_employment")
    assert end(store, app_id, case, "sent_to_human", "other_employment") == (
        "case ended: sent_to_human (other_employment)"
    )


def test_sending_to_a_human_is_always_allowed():
    store, app_id, case = submitted()
    assert end(store, app_id, case, "sent_to_human", "agent_unsure").startswith("case ended")


# the call itself


def test_reason_must_fit_the_outcome():
    store, app_id, case = ready()
    text = end(store, app_id, case, "complete", "no_answer")
    assert text == "not ended: 'no_answer' is not a valid reason for the outcome 'complete'"
    assert store.get(app_id).ending is None


def test_unknown_outcome():
    store, app_id, case = ready()
    assert end(store, app_id, case, "approved", "all_checks_passed").startswith(
        "not ended: 'all_checks_passed' is not a valid reason"
    )


def test_explanation_is_required():
    store, app_id, case = ready()
    assert end(store, app_id, case, explanation=" ") == "not ended: an explanation is required"


def test_retried_call_returns_the_same_confirmation():
    store, app_id, case = ready()
    first = end(store, app_id, case, call_id="call-end")
    assert end(store, app_id, case, call_id="call-end") == first


def test_a_case_ends_only_once():
    store, app_id, case = ready()
    end(store, app_id, case, call_id="call-1")
    text = end(store, app_id, case, "sent_to_human", "agent_unsure", call_id="call-2")
    assert text == "not ended: the case has already ended (complete, all_checks_passed)"
    assert store.get(app_id).ending.outcome == "complete"


def test_only_the_current_case_is_ended():
    store = ApplicationStore()
    daniel_case = load_case(CASES / "01_clean_application")
    daniel = submit_case(store, daniel_case)
    maria = submit_case(store, load_case(CASES / "02_maria_job_change"))
    end(store, daniel, daniel_case, "sent_to_human", "agent_unsure")
    assert store.get(daniel).ending is not None
    assert store.get(maria).ending is None


def test_a_reply_restarts_the_26_days():
    store, app_id, case = submitted("03_no_answer")
    message(store, app_id, case, days=0)
    replied = case.info.start_time + timedelta(days=10)
    store.record_reply(app_id, CustomerReply(received_at=replied, message="I'll send it soon."))
    # 30 days after the message, but only 20 after the reply: too early.
    text = end(store, app_id, case, "cancelled", "documents_not_provided", days=30)
    assert problems(text) == [
        "too early: 20 days since the last message or reply (the minimum is 26 days)"
    ]
    assert end(store, app_id, case, "cancelled", "documents_not_provided", days=36) == (
        "case ended: cancelled (documents_not_provided)"
    )


def test_an_old_reply_does_not_extend_the_wait():
    # The customer replied, then the agent wrote again: count from the message.
    store, app_id, case = submitted("03_no_answer")
    store.record_reply(app_id, CustomerReply(received_at=case.info.start_time, message="Hi"))
    message(store, app_id, case, days=5)
    assert end(store, app_id, case, "cancelled", "no_answer", days=31).startswith("case ended")


def test_last_contact():
    store, app_id, case = submitted("03_no_answer")
    stored = store.get(app_id)
    assert last_contact(stored) is None
    message(store, app_id, case, days=1)
    replied = case.info.start_time + timedelta(days=3)
    store.record_reply(app_id, CustomerReply(received_at=replied))
    assert last_contact(stored) == replied


def test_complete_refused_for_other_employment_says_why():
    store, app_id, case = submitted("04_retiree_other_employment")
    verify_identity(store, app_id, case.info.start_time)
    assert problems(end(store, app_id, case)) == [
        'the income cannot be verified: no document counts for the employment status "other"'
    ]
