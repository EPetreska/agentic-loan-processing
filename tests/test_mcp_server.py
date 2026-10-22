"""Tests for the MCP server, called over real HTTP as the agent will call it.
They need Postgres running (docker compose up -d) and are skipped without it."""

from datetime import timedelta
from pathlib import Path

import httpx2
import pytest
from mcp import ClientSession
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from psycopg_pool import ConnectionPool

from loans.evals.case import load_case, submit_case
from loans.evals.simulate import SimulatedClock
from loans.mcp import CALL_ID_KEY
from loans.mcp.client import connect as connected
from loans.mcp.server import NOT_ALLOWED, _case_opener, create_application_server
from loans.mcp.sessions import SessionClock, close_session, open_session
from loans.schemas.application import RawFile
from loans.store import ApplicationStore
from conftest import AGENT, SERVER_SECTION
from loans.offers import offer_for
from loans.tools.offer import descriptions as offer_descriptions
from loans.tools.offer.request_approval import request_approval
from loans.tools.offer.retrieve_documents import retrieve_documents
from loans.tools.offer.send_message import send_message as offer_send_message
from loans.tools.application import descriptions
from loans.tools.application.check_application import check_application
from loans.tools.application.finish import finish
from loans.tools.application.mark_income_verified import mark_income_verified
from loans.tools.application.read_document import read_document
from loans.tools.application.send_message import send_message
from loans.tools.application.verify_identity import verify_identity

CASES = Path(__file__).parent.parent / "evals" / "cases"
PDF = RawFile(filename="stub.pdf", content_type="application/pdf")
MESSAGE = "Please upload a newer pay stub."

pytestmark = pytest.mark.anyio


async def call(session: ClientSession, tool: str, args: dict | None = None, call_id: str | None = None):
    """The tool's text and whether the server refused the call as an error."""
    meta = {CALL_ID_KEY: call_id} if call_id else None
    result = await session.call_tool(tool, args or {}, meta=meta)
    return result.content[0].text, result.is_error


def status_without_valid_token(url: str, token: str | None) -> int:
    """The HTTP status of a tool call sent with this token, or none."""
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "check_application", "arguments": {}},
    }
    return httpx2.post(url, json=body, headers=headers).status_code


async def test_the_six_tools_as_the_llm_will_see_them(application_url, maria):
    _, _, token = maria
    async with connected(application_url, token) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    assert sorted(tools) == sorted(
        ["check_application", "read_document", "verify_identity",
         "mark_income_verified", "send_message", "finish"]
    )
    assert tools["read_document"].description == descriptions.READ_DOCUMENT
    assert tools["send_message"].description == descriptions.SEND_MESSAGE
    for tool in tools.values():
        params = set(tool.input_schema.get("properties", {}))
        # Neither the case nor the call ID is ever a tool argument.
        assert not params & {"application_id", "call_id", "ctx", "token"}, tool.name


async def test_maria_through_mcp_matches_the_direct_tools(application_url, maria, pg_store, pg_conn):
    case, app_id, token = maria
    start = case.info.start_time
    later = start + timedelta(days=2)
    new_stub = case.documents["pay_stub_new.json"]

    # The direct tools, in memory, with the same clock.
    memory = ApplicationStore(agent=AGENT)
    submit_case(memory, case)
    clock = SimulatedClock(start)
    direct = [
        check_application(memory, app_id, clock()),
        read_document(memory, app_id, "pay_stub", clock()),
        send_message(memory, app_id, clock(), "c1", MESSAGE, "Pay stub too old."),
    ]
    clock.jump_to(later)
    memory.upload(app_id, "pay_stub", new_stub, PDF, now=later)
    direct += [
        read_document(memory, app_id, "pay_stub", clock()),
        verify_identity(memory, app_id, clock()),
        mark_income_verified(memory, app_id, clock(), "c2", "pay_stub", "New stub at the new rate."),
        finish(memory, app_id, clock(), "c3", "complete", "all_checks_passed", "All checks passed."),
    ]

    # The same steps through the server; the customer's upload goes straight
    # to the database, as the bank's portal would.
    async with connected(application_url, token) as session:
        texts = [
            await call(session, "check_application"),
            await call(session, "read_document", {"slot": "pay_stub"}),
            await call(session, "send_message", {"text": MESSAGE, "reason": "Pay stub too old."}, "c1"),
        ]
        SessionClock(pg_conn, token).jump_to(later)
        pg_store.upload(app_id, "pay_stub", new_stub, PDF, now=later)
        texts += [
            await call(session, "read_document", {"slot": "pay_stub"}),
            await call(session, "verify_identity"),
            await call(session, "mark_income_verified", {"slot": "pay_stub", "reason": "New stub at the new rate."}, "c2"),
            await call(session, "finish", {"outcome": "complete", "reason": "all_checks_passed", "explanation": "All checks passed."}, "c3"),
        ]

    assert texts == [(text, False) for text in direct]
    assert pg_store.get(app_id) == memory.get(app_id)


async def test_agent_events_carry_the_model_and_prompt_version(application_url, maria, pg_store):
    _, app_id, token = maria
    async with connected(application_url, token) as session:
        await call(session, "check_application")
    event = pg_store.get(app_id).audit_log[-1]
    assert (event.actor, event.model, event.prompt_version) == ("agent", AGENT.model, AGENT.prompt_version)


# None: no token at all. The case ID: a caller that knows which case it wants.
@pytest.mark.parametrize("token", [None, "made-up-token", "02_maria_job_change"])
def test_a_missing_or_wrong_token_is_refused(application_url, maria, pg_store, token):
    _, app_id, _ = maria
    before = pg_store.get(app_id)
    assert status_without_valid_token(application_url, token) == 401
    assert pg_store.get(app_id) == before  # nothing read, nothing stored


def test_a_closed_session_is_refused(application_url, maria, pg_store, pg_conn):
    _, app_id, token = maria
    close_session(pg_conn, token)
    before = pg_store.get(app_id)
    assert status_without_valid_token(application_url, token) == 401
    assert pg_store.get(app_id) == before


async def test_the_client_cannot_even_connect_without_a_token(application_url, maria):
    with pytest.RaisesGroup(MCPError, flatten_subgroups=True):
        async with connected(application_url, "made-up-token"):
            pass


async def test_a_session_closed_during_a_connection_is_refused(application_url, maria, pg_store, pg_conn):
    _, app_id, token = maria
    async with connected(application_url, token) as session:
        assert not (await call(session, "check_application"))[1]
        close_session(pg_conn, token)
        before = pg_store.get(app_id)
        with pytest.raises(MCPError):
            await call(session, "check_application")
    assert pg_store.get(app_id) == before


async def test_a_token_reaches_only_its_own_case(application_url, maria, pg_store, pg_conn):
    _, maria_id, maria_token = maria
    daniel = load_case(CASES / "01_clean_application")
    daniel_id = submit_case(pg_store, daniel)
    open_session(pg_conn, daniel_id, "application", daniel.info.start_time, AGENT)
    daniel_before = pg_store.get(daniel_id)
    async with connected(application_url, maria_token) as session:
        text, _ = await call(session, "check_application")
        # Naming Daniel's case in the arguments changes nothing.
        await call(session, "check_application", {"application_id": daniel_id})
    assert "Maria" in text and "Daniel" not in text
    assert pg_store.get(daniel_id) == daniel_before
    assert len(pg_store.get(maria_id).audit_log) > 1


async def test_a_retried_message_is_sent_once(application_url, maria, pg_store):
    _, app_id, token = maria
    args = {"text": MESSAGE, "reason": "Pay stub too old."}
    async with connected(application_url, token) as session:
        first = await call(session, "send_message", args, "c1")
        retry = await call(session, "send_message", args, "c1")
    assert first == retry == ("message sent (message 1 in this case)", False)
    assert len(pg_store.get(app_id).messages) == 1


async def test_a_call_without_a_call_id_is_refused(application_url, maria, pg_store, pg_conn):
    case, app_id, token = maria
    args = {"text": MESSAGE, "reason": "Pay stub too old."}
    async with connected(application_url, token) as session:
        text, is_error = await call(session, "send_message", args)
    assert is_error and "no call ID" in text
    assert pg_store.get(app_id).messages == []
    assert SessionClock(pg_conn, token).now == case.info.start_time  # the clock did not move


def test_not_allowed_names_no_detail():
    # The refusal says nothing about why, so it cannot be used to probe tokens.
    assert NOT_ALLOWED == "not allowed: unknown or closed session"


async def test_the_tools_check_the_session_again(pg_conninfo, maria, pg_store, pg_conn):
    # The door checks every call's token first. The tools check it again, for
    # a session closed between the door and the tool; to test that second
    # lock on its own, this calls a tool without going through the door.
    _, app_id, token = maria
    close_session(pg_conn, token)
    before = pg_store.get(app_id)
    with ConnectionPool(pg_conninfo, kwargs={"autocommit": True}, min_size=1, max_size=1) as pool:
        server = create_application_server(pool)
        auth_context_var.set(AuthenticatedUser(AccessToken(token=token, client_id="x", scopes=[])))
        with pytest.raises(ToolError, match=NOT_ALLOWED):
            await server.call_tool("check_application", {})
    assert pg_store.get(app_id) == before


# The offer agent's endpoint: its own three tools, for offer sessions only.


@pytest.fixture
def offered(pg_store, pg_conn):
    """Maria's case with her offer made, and a session for the offer agent:
    (case, ID, token)."""
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case, "maria-offer")  # beside the maria fixture's case
    pg_store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    return case, app_id, open_session(pg_conn, app_id, "offer", case.info.start_time, AGENT)


async def test_the_offer_endpoint_offers_only_the_offer_agents_three_tools(offer_url, offered):
    _, _, token = offered
    async with connected(offer_url, token) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    assert sorted(tools) == ["request_approval", "retrieve_documents", "send_message"]
    assert tools["retrieve_documents"].description == offer_descriptions.RETRIEVE_DOCUMENTS
    assert tools["send_message"].description == offer_descriptions.SEND_MESSAGE  # the offer agent's own
    for tool in tools.values():
        params = set(tool.input_schema.get("properties", {}))
        assert not params & {"application_id", "call_id", "ctx", "token", "search"}, tool.name


async def test_the_offer_tools_through_mcp_match_the_direct_tools(offer_url, offered, pg_store):
    case, app_id, token = offered
    start = case.info.start_time

    # The direct tools, in memory, with the same clock and the same search.
    memory = ApplicationStore(agent=AGENT)
    submit_case(memory, case, app_id)
    memory.record_offer(app_id, offer_for(case.application), start)
    clock = SimulatedClock(start)
    direct = [
        retrieve_documents(memory, app_id, clock(), "o1", "prepayment penalty", lambda q: [SERVER_SECTION]),
        offer_send_message(memory, app_id, clock(), "o2", "There is no fee for paying early.", "Her question."),
        request_approval(memory, app_id, clock(), "o3", "Borrow $17,000.", "More than offered."),
    ]

    async with connected(offer_url, token) as session:
        texts = [
            await call(session, "retrieve_documents", {"query": "prepayment penalty"}, "o1"),
            await call(session, "send_message", {"text": "There is no fee for paying early.", "reason": "Her question."}, "o2"),
            await call(session, "request_approval", {"request": "Borrow $17,000.", "reason": "More than offered."}, "o3"),
        ]

    assert texts == [(text, False) for text in direct]
    assert pg_store.get(app_id) == memory.get(app_id)


def test_each_endpoint_refuses_the_other_agents_token(application_url, offer_url, maria, offered, pg_store):
    _, maria_id, application_token = maria
    _, offered_id, offer_token = offered
    before = (pg_store.get(maria_id), pg_store.get(offered_id))
    assert status_without_valid_token(offer_url, application_token) == 401
    assert status_without_valid_token(application_url, offer_token) == 401
    assert (pg_store.get(maria_id), pg_store.get(offered_id)) == before  # nothing read, nothing stored


async def test_the_offer_agent_cannot_call_an_application_tool(offer_url, offered, pg_store):
    _, app_id, token = offered
    before = pg_store.get(app_id)
    async with connected(offer_url, token) as session:
        text, is_error = await call(
            session, "finish", {"outcome": "complete", "reason": "all_checks_passed", "explanation": "x"}, "f1"
        )
    assert is_error and "finish" in text
    assert pg_store.get(app_id) == before


def test_the_offer_tools_check_the_sessions_agent_again(pg_conninfo, maria, pg_store):
    # The second lock on its own, without the door: the offer tools' way to
    # their case refuses an open session for the application agent.
    _, app_id, application_token = maria
    with ConnectionPool(pg_conninfo, kwargs={"autocommit": True}, min_size=1, max_size=1) as pool:
        auth_context_var.set(AuthenticatedUser(AccessToken(token=application_token, client_id="x", scopes=[])))
        with pytest.raises(ToolError, match=NOT_ALLOWED):
            with _case_opener(pool, "offer")():
                pass
        with _case_opener(pool, "application")() as c:  # its own agent's tools may
            assert c.application_id == app_id
