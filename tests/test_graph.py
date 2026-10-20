"""Tests for the agent loop, with a fake LLM that plays back fixed answers.

No real model is called: these tests check the wiring (tools are called, the
loop stops at the right time), not the agent's judgment. The judgment is
tested by the evals, with a real model.
"""

from datetime import date
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from loans.agent.graph import build_graph, make_model, thread_config
from loans.agent.prompt import first_turn
from loans.agent.tools import make_tools
from loans.evals.case import load_case, submit_case
from loans.store import ApplicationStore

CASES = Path(__file__).parent.parent / "evals" / "cases"
DAY = date(2026, 9, 14)


class FakeModel(GenericFakeChatModel):
    """Plays back a fixed list of answers; accepts tools like a real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def step(name, args=None, call_id=None):
    """An LLM answer that calls one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id or f"call-{name}"}],
    )


def run(answers, case_name="01_clean_application"):
    store = ApplicationStore()
    case = load_case(CASES / case_name)
    app_id = submit_case(store, case)
    model = FakeModel(messages=iter(answers))
    tools = make_tools(store, app_id, lambda: case.info.start_time)
    graph = build_graph(model, tools, store, app_id)
    result = graph.invoke({"messages": [HumanMessage(first_turn(None, DAY))]}, thread_config(app_id))
    return store, app_id, result


DANIEL = [
    step("check_application"),
    step("read_document", {"slot": "pay_stub"}),
    step("verify_identity"),
    step("mark_income_verified", {"slot": "pay_stub", "reason": "Same employer, amounts match."}),
    step(
        "finish",
        {"outcome": "complete", "reason": "all_checks_passed", "explanation": "All checks passed."},
    ),
]


def test_daniel_is_completed_through_the_loop():
    store, app_id, _ = run(DANIEL)
    assert store.get(app_id).ending.outcome == "complete"
    events = [e.event for e in store.get(app_id).audit_log]
    assert events == [
        "application_submitted",
        "form_checked",
        "document_read",
        "identity_checked",
        "income_verified",
        "application_completed",
    ]


def test_the_loop_stops_after_finish():
    # An extra answer after finish is never asked for.
    store, app_id, result = run([*DANIEL, step("send_message", {"text": "Hi", "reason": "x"})])
    assert store.get(app_id).messages == []
    assert result["messages"][-1].type == "tool"  # finish's result is the last message


def test_a_refused_finish_goes_back_to_the_llm():
    early = {"outcome": "complete", "reason": "all_checks_passed", "explanation": "Too early."}
    answers = [step("finish", early, "f1"), *DANIEL]
    store, app_id, result = run(answers)
    first_result = result["messages"][2]
    assert first_result.content.startswith("not ended:")
    assert store.get(app_id).ending.outcome == "complete"


WAITING = [
    step("send_message", {"text": "Please upload a pay stub.", "reason": "Missing."}),
    AIMessage(content="Waiting for the customer's pay stub."),
]


def build(answers, case_name="03_no_answer"):
    store = ApplicationStore()
    case = load_case(CASES / case_name)
    app_id = submit_case(store, case)
    tools = make_tools(store, app_id, lambda: case.info.start_time)
    graph = build_graph(FakeModel(messages=iter(answers)), tools, store, app_id)
    return store, app_id, graph


def test_the_run_pauses_when_the_llm_answers_without_a_tool():
    store, app_id, graph = build(WAITING)
    config = thread_config(app_id)
    result = graph.invoke({"messages": [HumanMessage(first_turn(None, DAY))]}, config)
    assert graph.get_state(config).next == ("wait",)  # paused in the wait box
    assert result["__interrupt__"][0].value == {"waiting": "Waiting for the customer's pay stub."}
    assert store.get(app_id).ending is None
    assert len(store.get(app_id).messages) == 1


def test_a_paused_run_resumes_with_the_news_as_a_new_message():
    handoff = {"outcome": "sent_to_human", "reason": "agent_unsure", "explanation": "Unsure."}
    store, app_id, graph = build([*WAITING, step("finish", handoff)])
    config = thread_config(app_id)
    graph.invoke({"messages": [HumanMessage(first_turn(None, DAY))]}, config)
    result = graph.invoke(Command(resume="2 days later, the customer replied."), config)
    news = result["messages"][-3]
    assert (news.type, news.content) == ("human", "2 days later, the customer replied.")
    assert store.get(app_id).ending.reason == "agent_unsure"
    assert graph.get_state(config).next == ()  # ended, not paused


def test_each_case_keeps_its_own_conversation():
    store, app_id, graph = build(WAITING)
    graph.invoke({"messages": [HumanMessage(first_turn(None, DAY))]}, thread_config(app_id))
    assert graph.get_state(thread_config("another-case")).values == {}


def test_a_wrong_tool_argument_is_returned_to_the_llm():
    answers = [step("read_document", {"slot": "id_front"}), *DANIEL]
    store, app_id, result = run(answers)
    error = result["messages"][2]
    assert error.type == "tool" and error.status == "error"
    assert store.get(app_id).ending.outcome == "complete"


def test_make_model_reads_the_setting():
    model = make_model("ollama:qwen3:4b")
    assert model.model == "qwen3:4b"
    assert model.num_ctx == 16_384  # not Ollama's small default
    assert model.reasoning is True  # thinking returned separately, not in the answer


def test_make_model_rejects_an_unknown_setting():
    with pytest.raises(ValueError):
        make_model("qwen3")
