"""Tests for the tools as the LLM sees them."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from loans.agent.tools import make_tools
from loans.evals.case import load_case, submit_case
from loans.store import ApplicationStore
from loans.tools.send_message import MAX_MESSAGE_LENGTH

CASES = Path(__file__).parent.parent / "evals" / "cases"
HIDDEN = {"store", "application_id", "now", "today", "clock", "call_id"}


def daniel():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    app_id = submit_case(store, case)
    tools = {t.name: t for t in make_tools(store, app_id, lambda: case.info.start_time)}
    return store, app_id, tools


def call(tool, args, call_id="call-1"):
    """Call a tool the way LangGraph does, with the LLM's tool call."""
    message = tool.invoke({"type": "tool_call", "id": call_id, "name": tool.name, "args": args})
    return message.content


def test_the_llm_sees_only_its_own_parameters():
    _, _, tools = daniel()
    params = {
        name: set(t.tool_call_schema.model_json_schema().get("properties", {}))
        for name, t in tools.items()
    }
    assert params == {
        "check_application": set(),
        "read_document": {"slot"},
        "verify_identity": set(),
        "mark_income_verified": {"slot", "reason"},
        "send_message": {"text", "reason"},
        "finish": {"outcome", "reason", "explanation"},
    }
    assert all(not (p & HIDDEN) for p in params.values())


def test_every_tool_has_a_description():
    _, _, tools = daniel()
    assert all(t.description for t in tools.values())


def test_slots_and_reasons_are_fixed_choices():
    _, _, tools = daniel()
    slot = tools["read_document"].tool_call_schema.model_json_schema()["properties"]["slot"]
    assert slot["enum"] == ["pay_stub", "w2", "tax_return"]
    finish = tools["finish"].tool_call_schema.model_json_schema()["properties"]
    assert finish["outcome"]["enum"] == ["complete", "cancelled", "sent_to_human"]
    assert "all_checks_passed" in finish["reason"]["enum"]


def test_the_call_id_comes_from_the_tool_call():
    store, app_id, tools = daniel()
    call(tools["send_message"], {"text": "Hello.", "reason": "A test."}, call_id="abc-123")
    assert store.get(app_id).messages[0].message_id == "abc-123"


def test_a_retried_tool_call_is_not_sent_twice():
    store, app_id, tools = daniel()
    for _ in range(2):
        call(tools["send_message"], {"text": "Hello.", "reason": "A test."}, call_id="abc-123")
    assert len(store.get(app_id).messages) == 1


def test_the_tools_work_only_on_their_own_case():
    store = ApplicationStore()
    daniel_case = load_case(CASES / "01_clean_application")
    maria_case = load_case(CASES / "02_maria_job_change")
    daniel_id = submit_case(store, daniel_case)
    maria_id = submit_case(store, maria_case)
    tools = {t.name: t for t in make_tools(store, daniel_id, lambda: daniel_case.info.start_time)}
    # An application ID in the arguments is not a parameter, so it is ignored.
    call(tools["send_message"], {"text": "Hi.", "reason": "Test.", "application_id": maria_id})
    assert len(store.get(daniel_id).messages) == 1
    assert store.get(maria_id).messages == []


def test_a_wrong_slot_is_rejected_before_the_tool_runs():
    # In the loop, LangGraph's ToolNode returns this error to the LLM.
    store, app_id, tools = daniel()
    with pytest.raises(ValidationError):
        call(tools["read_document"], {"slot": "id_front"})
    assert len(store.get(app_id).audit_log) == 1  # only the submission


def test_a_full_case_through_the_wrappers():
    store, app_id, tools = daniel()
    assert call(tools["check_application"], {}).startswith("complete: yes")
    assert "bank income rules: met" in call(tools["read_document"], {"slot": "pay_stub"})
    assert "name_matches_form: yes" in call(tools["verify_identity"], {})
    call(tools["mark_income_verified"], {"slot": "pay_stub", "reason": "Matches."}, "c1")
    text = call(
        tools["finish"],
        {"outcome": "complete", "reason": "all_checks_passed", "explanation": "All good."},
        "c2",
    )
    assert text == "case ended: complete (all_checks_passed)"


def test_send_message_description_states_the_real_limit():
    _, _, tools = daniel()
    description = tools["send_message"].description
    assert f"longer than {MAX_MESSAGE_LENGTH} characters" in description
    assert "e-sign consent" in description
