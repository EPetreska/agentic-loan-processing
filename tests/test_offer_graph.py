"""Tests for the offer agent's loop, with a fake LLM that plays back fixed
answers: the wiring (tools are called, the run pauses and resumes, it never
ends after a tool step), not the agent's judgment."""

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from loans.agents.application.graph import build_application_graph
from loans.agents.application.prompt import first_turn as application_first_turn
from loans.agents.application.tools import make_application_tools
from loans.agents.loop import end_run, thread_config
from loans.agents.offer.graph import build_offer_graph, never_ends
from loans.agents.offer.prompt import SYSTEM_PROMPT, closed_turn, first_turn, officer_decision_turn
from loans.agents.offer.tools import make_offer_tools
from loans.evals.case import load_case, submit_case
from loans.offers import offer_for
from loans.rag.search import SearchResult
from loans.store import ApplicationStore
from test_graph import FakeModel, step

CASES = Path(__file__).parent.parent / "evals" / "cases"
DAY = date(2026, 9, 20)
SECTION = SearchResult(chunk_id="bank-07", document="bank", title="Paying early",
                       text="No prepayment penalty.", source="the bank's own document", distance=0.1)  # fmt: skip


class RecordingModel(FakeModel):
    """A fake LLM that also keeps what it was given on each call."""

    seen: list = []

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def offered():
    """Maria's case with her offer made."""
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    now = case.info.start_time + timedelta(days=1)
    return store, app_id, make_offer_tools(store, app_id, lambda: now, lambda query: [SECTION])


def start(graph, app_id, message):
    turn = first_turn("Your loan offer", DAY, message, DAY)
    config = thread_config(app_id, "offer")
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(turn)]}, config))
    return config


def test_a_question_is_looked_up_answered_and_then_the_agent_waits():
    store, app_id, tools = offered()
    model = FakeModel(messages=iter([
        step("retrieve_documents", {"query": "prepayment penalty"}),
        step("send_message", {"text": "There is no fee for paying early.", "reason": "Her question."}),
        AIMessage("Waiting for the customer."),
    ]))  # fmt: skip
    graph = build_offer_graph(model, tools)
    config = start(graph, app_id, "Is there a fee if I pay early?")
    assert graph.get_state(config).next == ("wait",)
    stored = store.get(app_id)
    assert [r.chunk_ids for r in stored.document_retrievals] == [["bank-07"]]
    assert [m.text for m in stored.messages] == ["There is no fee for paying early."]


def test_a_request_waits_for_the_officer_and_resumes_with_the_decision():
    store, app_id, tools = offered()
    model = FakeModel(messages=iter([
        step("request_approval", {"request": "Borrow $17,000.", "reason": "More than offered."}),
        step("send_message", {"text": "A loan officer will decide.", "reason": "Tell her."}, "call-m1"),
        AIMessage("Waiting for the loan officer's decision."),
        step("send_message", {"text": "The loan officer declined.", "reason": "The decision."}, "call-m2"),
        AIMessage("Waiting for the customer."),
    ]))  # fmt: skip
    graph = build_offer_graph(model, tools)
    config = start(graph, app_id, "Could I borrow $17,000?")
    assert graph.get_state(config).next == ("wait",)
    assert store.get(app_id).pending_approval() is not None

    decision = officer_decision_turn("declined", DAY)
    asyncio.run(graph.ainvoke(Command(resume=decision), config))
    messages = graph.get_state(config).values["messages"]
    assert graph.get_state(config).next == ("wait",)
    assert decision in [m.content for m in messages if isinstance(m, HumanMessage)]
    assert [m.text for m in store.get(app_id).messages] == ["A loan officer will decide.", "The loan officer declined."]


def test_the_workflow_ends_the_run_without_the_llm():
    _, app_id, tools = offered()
    # One answer only: a second call to the LLM would find none and fail.
    graph = build_offer_graph(FakeModel(messages=iter([AIMessage("Waiting for the customer.")])), tools)
    config = start(graph, app_id, "Thanks.")
    closing = closed_turn("accepted", DAY)
    asyncio.run(graph.ainvoke(end_run(closing), config))
    state = graph.get_state(config)
    assert state.next == ()  # ended, not paused
    assert state.values["messages"][-1].content == closing  # kept as the last message


def test_no_tool_answer_ends_the_run():
    # Not even one that looks like the application agent's finish.
    looks_final = ToolMessage("ended: complete", name="finish", tool_call_id="x")
    assert never_ends([looks_final]) is False


def test_the_agent_gets_the_offer_prompt():
    _, app_id, tools = offered()
    model = RecordingModel(messages=iter([AIMessage("Waiting for the customer.")]))
    model.seen = []
    start(build_offer_graph(model, tools), app_id, "Thanks.")
    [first, *_] = model.seen[0]
    assert first == SystemMessage(SYSTEM_PROMPT)


def test_each_agent_keeps_its_own_conversation_on_a_case():
    store, app_id, offer_tools = offered()
    shared = InMemorySaver()  # as a database checkpointer would be
    application = build_application_graph(
        FakeModel(messages=iter([AIMessage("Waiting.")])), make_application_tools(store, app_id, lambda: None), shared
    )
    asyncio.run(application.ainvoke(
        {"messages": [HumanMessage(application_first_turn(None, DAY))]}, thread_config(app_id, "application")
    ))  # fmt: skip
    offer = build_offer_graph(FakeModel(messages=iter([AIMessage("Waiting.")])), offer_tools, shared)
    config = start(offer, app_id, "Thanks.")
    messages = offer.get_state(config).values["messages"]
    assert len(messages) == 2 and "The customer wrote" in messages[0].content


def test_a_thread_name_never_contains_a_slash():
    # LangSmith's thread page breaks on a "/" in the thread ID.
    for agent in ["application", "offer"]:
        assert "/" not in thread_config("02_maria_job_change@2026-10-22T18-32-08", agent)["configurable"]["thread_id"]
    with pytest.raises(ValueError, match="cannot contain '/'"):
        thread_config("runs/02", "offer")
