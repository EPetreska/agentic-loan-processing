"""Tests for the run script's --show output."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from loans.run import format_message


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
