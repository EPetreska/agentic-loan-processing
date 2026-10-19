"""Tests for the send_message tool."""

from datetime import timedelta
from pathlib import Path

from loans.evals.case import EvalCase, load_case, submit_case
from loans.store import ApplicationStore
from loans.tools.send_message import MAX_MESSAGE_LENGTH, send_message

CASES = Path(__file__).parent.parent / "evals" / "cases"
TEXT = "Please upload a recent pay stub."
REASON = "No income document was uploaded."


def submitted(name: str = "03_no_answer") -> tuple[ApplicationStore, str, EvalCase]:
    store = ApplicationStore()
    case = load_case(CASES / name)
    return store, submit_case(store, case), case


def send(store, app_id, case, call_id="call-1", text=TEXT, reason=REASON, minutes=0):
    now = case.info.start_time + timedelta(minutes=minutes)
    return send_message(store, app_id, now, call_id, text, reason)


def test_message_is_recorded():
    store, app_id, case = submitted()
    assert send(store, app_id, case, minutes=5) == "message sent (message 1 in this case)"
    [message] = store.get(app_id).messages
    assert message.text == TEXT
    assert message.reason == REASON
    assert message.message_id == "call-1"
    assert message.sent_at == case.info.start_time + timedelta(minutes=5)


def test_messages_are_counted_in_order():
    store, app_id, case = submitted()
    send(store, app_id, case, call_id="call-1")
    assert send(store, app_id, case, call_id="call-2", text="Reminder.") == (
        "message sent (message 2 in this case)"
    )
    assert [m.text for m in store.get(app_id).messages] == [TEXT, "Reminder."]


def test_retried_call_is_not_sent_twice():
    store, app_id, case = submitted()
    first = send(store, app_id, case, call_id="call-1")
    retry = send(store, app_id, case, call_id="call-1")
    assert retry == first
    assert len(store.get(app_id).messages) == 1


def test_retry_of_an_earlier_call_keeps_its_position():
    store, app_id, case = submitted()
    send(store, app_id, case, call_id="call-1")
    send(store, app_id, case, call_id="call-2")
    assert send(store, app_id, case, call_id="call-1") == "message sent (message 1 in this case)"
    assert len(store.get(app_id).messages) == 2


def test_no_message_without_esign_consent():
    store, app_id, case = submitted()
    store.get(app_id).metadata.esign_consented_at = None  # the customer declined e-sign
    text = send(store, app_id, case)
    assert text.startswith("not sent: the customer has not given e-sign consent")
    assert store.get(app_id).messages == []


def test_empty_message_is_not_sent():
    store, app_id, case = submitted()
    assert send(store, app_id, case, text="   ") == "not sent: the message is empty"
    assert store.get(app_id).messages == []


def test_reason_is_required():
    store, app_id, case = submitted()
    assert send(store, app_id, case, reason="") == "not sent: a reason is required"
    assert store.get(app_id).messages == []


def test_too_long_message_is_not_sent():
    store, app_id, case = submitted()
    text = send(store, app_id, case, text="x" * (MAX_MESSAGE_LENGTH + 1))
    assert text.startswith("not sent: the message is longer than")
    assert store.get(app_id).messages == []


def test_message_goes_only_to_the_current_application():
    store = ApplicationStore()
    robert_case = load_case(CASES / "03_no_answer")
    robert = submit_case(store, robert_case)
    maria = submit_case(store, load_case(CASES / "02_maria_job_change"))
    send(store, robert, robert_case)
    assert len(store.get(robert).messages) == 1
    assert store.get(maria).messages == []


def test_confirmation_contains_no_contact_details():
    store, app_id, case = submitted()
    text = send(store, app_id, case)
    assert case.application.email not in text
    assert case.application.phone not in text
