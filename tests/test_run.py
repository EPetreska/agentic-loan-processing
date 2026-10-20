"""Tests for the run script's --show output."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from loans.evals.case import load_case, submit_case
from loans.evals.simulate import simulate_case
from loans.run import LivePrinter, conversation, format_message, write_conversation
from loans.store import ApplicationStore

CASES = Path(__file__).parent.parent / "evals" / "cases"


def test_agent_message_with_thinking_and_a_tool_call():
    message = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "The form is fine.\nNext the pay stub."},
        tool_calls=[{"name": "read_document", "args": {"slot": "pay_stub"}, "id": "c1"}],
    )
    assert format_message(message).splitlines() == [
        "[thinking] The form is fine.",
        "    Next the pay stub.",
        "[agent] calls read_document(slot='pay_stub')",
    ]


def test_agent_text_only():
    assert format_message(AIMessage(content="Waiting for the pay stub.")) == (
        "[agent] Waiting for the pay stub."
    )


def test_tool_result():
    message = ToolMessage(
        content="complete: yes\nok: ssn: ok", name="check_application", tool_call_id="c1"
    )
    assert format_message(message).splitlines() == [
        "[tool check_application] complete: yes",
        "    ok: ssn: ok",
    ]


def test_news_that_starts_or_resumes_the_case():
    assert format_message(HumanMessage("26 days have passed.")) == "[news] 26 days have passed."


def test_without_text_only_the_tool_calls_are_shown():
    message = AIMessage(
        content="Let me check.",
        additional_kwargs={"reasoning_content": "Thinking..."},
        tool_calls=[{"name": "verify_identity", "args": {}, "id": "c1"}],
    )
    assert format_message(message, with_text=False) == "[agent] calls verify_identity()"


def test_live_printer_labels_each_stream_once(capsys):
    printer = LivePrinter()
    for word in ["The ", "form ", "is ", "fine."]:
        printer.token("thinking", word)
    printer.token("answer", "Checking now.")
    call = {"name": "verify_identity", "args": {}, "id": "c1"}
    printer.message(AIMessage(content="", tool_calls=[call]))
    out = capsys.readouterr().out
    assert out == (
        "[thinking] The form is fine.\n\n"
        "[agent] Checking now.\n\n"
        "[agent] calls verify_identity()\n\n"
    )


def maria_after_a_simulated_run():
    """Maria's case through the scripted customer, with a fake LLM."""
    from test_simulate import MARIA, FakeModel  # the same scripted run

    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    result = simulate_case(case, store, FakeModel(messages=iter(MARIA)))
    return case, store.get(result.application_id)


def test_conversation_in_order():
    case, stored = maria_after_a_simulated_run()
    entries = conversation(case, stored)
    assert [(e["from"], e["text"]) for e in entries] == [
        ("customer", case.customer.first_message),
        ("agent", "Please upload a recent pay stub."),
        ("customer", "Here is my latest pay stub with the new pay rate."),
    ]
    assert entries[0]["with_application"] is True
    assert entries[1]["reason"] == "Missing document."
    assert entries[2]["uploads"] == ["pay_stub"]
    assert entries[0]["at"] == "2026-09-14T15:00:00+00:00"


def test_conversation_file(tmp_path):
    case, stored = maria_after_a_simulated_run()
    path = tmp_path / "run" / "02_maria_job_change.conversation.json"
    write_conversation(path, case, stored)
    assert json.loads(path.read_text()) == conversation(case, stored)


def test_conversation_without_any_messages():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    stored = store.get(submit_case(store, case))
    assert conversation(case, stored) == []
