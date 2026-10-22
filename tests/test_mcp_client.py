"""Tests for the MCP client: the server's tools as the agent will use them,
run by LangGraph's ToolNode as in the agent loop. They need Postgres running
(docker compose up -d) and are skipped without it."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from loans.agents.application.tools import make_application_tools
from loans.agents.offer.tools import make_offer_tools
from loans.evals.case import load_case, submit_case
from loans.offers import offer_for
from loans.mcp import client
from loans.mcp.client import ServerError, load_tools
from loans.mcp.sessions import close_session, open_session
from loans.store import ApplicationStore

pytestmark = pytest.mark.anyio

CASES = Path(__file__).parent.parent / "evals" / "cases"

MESSAGE = {"text": "Please upload a newer pay stub.", "reason": "Pay stub too old."}


async def run_tool(tools, name: str, args: dict, call_id: str):
    """Run one tool call the way the agent loop does (a graph with LangGraph's
    ToolNode), and return the tool's message to the LLM."""
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tools")
    request = AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])
    result = await graph.compile().ainvoke({"messages": [request]})
    return result["messages"][-1]


async def test_the_llm_sees_the_same_tools_through_mcp(application_url, maria):
    _, _, token = maria
    tools = await load_tools(application_url, token, "application")
    through_mcp = {t.name: convert_to_openai_tool(t) for t in tools}
    direct = make_application_tools(ApplicationStore(), "unused", lambda: datetime.now(UTC))
    assert through_mcp == {t.name: convert_to_openai_tool(t) for t in direct}


async def test_a_tool_call_reaches_the_server_with_its_call_id(application_url, maria, pg_store):
    _, app_id, token = maria
    tools = await load_tools(application_url, token, "application")
    reply = await run_tool(tools, "send_message", MESSAGE, "call_1")
    assert reply.content == "message sent (message 1 in this case)"
    assert [m.message_id for m in pg_store.get(app_id).messages] == ["call_1"]


async def test_a_retried_tool_call_is_not_repeated(application_url, maria, pg_store):
    _, app_id, token = maria
    tools = await load_tools(application_url, token, "application")
    first = await run_tool(tools, "send_message", MESSAGE, "call_1")
    retry = await run_tool(tools, "send_message", MESSAGE, "call_1")
    assert first.content == retry.content
    assert len(pg_store.get(app_id).messages) == 1


async def test_a_rule_refusal_goes_back_to_the_llm(application_url, maria, pg_store):
    _, app_id, token = maria
    tools = await load_tools(application_url, token, "application")
    reply = await run_tool(tools, "send_message", {**MESSAGE, "text": " "}, "call_1")
    assert reply.content == "not sent: the message is empty"
    assert pg_store.get(app_id).messages == []


async def test_a_wrong_argument_goes_back_to_the_llm(application_url, maria, pg_store):
    _, app_id, token = maria
    tools = await load_tools(application_url, token, "application")
    events = len(pg_store.get(app_id).audit_log)
    reply = await run_tool(tools, "read_document", {"slot": "passport"}, "call_1")
    assert reply.status == "error" and "slot" in reply.content
    assert len(pg_store.get(app_id).audit_log) == events  # caught before the call


async def test_a_bad_token_stops_before_any_tool(application_url, maria):
    with pytest.raises(ServerError, match="cannot load the tools"):
        await load_tools(application_url, "made-up-token", "application")


async def test_a_session_closed_during_the_run_stops_the_run(application_url, maria, pg_conn, pg_store):
    _, app_id, token = maria
    tools = await load_tools(application_url, token, "application")
    close_session(pg_conn, token)
    before = pg_store.get(app_id)
    with pytest.raises(ServerError, match="check_application failed"):
        await run_tool(tools, "check_application", {}, "call_1")
    assert pg_store.get(app_id) == before


async def test_a_server_down_stops_the_run(maria):
    _, _, token = maria
    with pytest.raises(ServerError, match="cannot load the tools"):
        await load_tools("http://127.0.0.1:9/mcp", token, "application")  # nothing listens there


async def test_an_unexpected_tool_list_is_refused(application_url, maria, monkeypatch):
    # Seen from the client, the server now offers a tool it does not expect.
    _, _, token = maria
    monkeypatch.setitem(client.EXPECTED_TOOLS, "application", client.EXPECTED_TOOLS["application"] - {"finish"})
    with pytest.raises(ServerError, match=r"offers \['finish'\]"):
        await load_tools(application_url, token, "application")


async def test_an_error_from_the_server_stops_the_run(application_url, maria, pg_store, monkeypatch):
    # The client and the server disagree on where the call ID goes, so the
    # server answers with an error. That is the system's fault, not the
    # agent's: the run stops and the LLM never sees it.
    _, app_id, token = maria
    monkeypatch.setattr(client, "CALL_ID_KEY", "wrong/key")
    tools = await load_tools(application_url, token, "application")
    with pytest.raises(ServerError, match="refused by the server: .*the request has no call ID"):
        await run_tool(tools, "send_message", MESSAGE, "call_1")
    assert pg_store.get(app_id).messages == []


# The offer agent's tools, from its own endpoint.


@pytest.fixture
def offered(pg_store, pg_conn):
    """Maria's case with her offer made, and a session for the offer agent:
    (ID, token)."""
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case, "maria-offer")
    pg_store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    return app_id, open_session(pg_conn, app_id, "offer", case.info.start_time)


async def test_the_offer_agent_sees_the_same_tools_through_mcp(offer_url, offered):
    _, token = offered
    through_mcp = {t.name: convert_to_openai_tool(t) for t in await load_tools(offer_url, token, "offer")}
    direct = make_offer_tools(ApplicationStore(), "unused", lambda: datetime.now(UTC), lambda query: [])
    assert through_mcp == {t.name: convert_to_openai_tool(t) for t in direct}


async def test_an_offer_tool_call_reaches_the_server(offer_url, offered, pg_store):
    app_id, token = offered
    tools = await load_tools(offer_url, token, "offer")
    reply = await run_tool(tools, "retrieve_documents", {"query": "prepayment penalty"}, "call_1")
    assert "[section 1] Paying early" in reply.content  # the test server's stand-in search
    [retrieval] = pg_store.get(app_id).document_retrievals
    assert (retrieval.retrieval_id, retrieval.query) == ("call_1", "prepayment penalty")


async def test_the_offer_agent_refuses_the_application_agents_tools(application_url, maria):
    # Pointed at the wrong endpoint (with a token that opens it), the offer
    # agent's client refuses tools that are not its own.
    _, _, token = maria
    with pytest.raises(ServerError, match=r"for the offer agent: offers \['check_application'"):
        await load_tools(application_url, token, "offer")
