"""Tests for the bank's workflow steps on their own, without the simulation
(the full runs are tested in test_simulate.py)."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import AGENT
from loans.db.postgres_store import PostgresStore
from loans.evals.case import load_case, submit_case
from loans.evals.simulate import SimulatedClock
from loans.offers import offer_for
from loans.schemas.application import RawFile
from loans.store import ApplicationStore, CaseEnding
from loans.audit import AgentVersion
from loans.workflow import PROMPT_VERSIONS, after_application, open_case_for_agent, receive_reply

CASES = Path(__file__).parent.parent / "evals" / "cases"


def ended(outcome: str, reason: str) -> tuple[ApplicationStore, str, SimulatedClock]:
    """Maria's case, ended by the application agent this way."""
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    clock = SimulatedClock(case.info.start_time + timedelta(days=2))
    ending = CaseEnding(call_id="f", outcome=outcome, reason=reason, explanation="x", ended_at=clock.now)
    store.record_ending(app_id, ending)
    return store, app_id, clock


def test_a_complete_case_gets_the_credit_check_and_the_offer():
    store, app_id, clock = ended("complete", "all_checks_passed")
    after_application(store, app_id, clock)
    stored = store.get(app_id)
    assert stored.credit_check.passed
    assert [o.term_months for o in stored.offers] == [48]
    assert [e.event for e in stored.audit_log][-3:] == ["credit_checked", "offer_created", "offer_sent"]
    assert len(stored.offers_sent) == 1


@pytest.mark.parametrize(
    "outcome, reason",
    [("cancelled", "no_answer"), ("sent_to_human", "agent_unsure")],
)
def test_other_endings_go_no_further(outcome, reason):
    store, app_id, clock = ended(outcome, reason)
    after_application(store, app_id, clock)
    stored = store.get(app_id)
    assert stored.credit_check is None and stored.offers == []


def test_nothing_happens_before_the_case_has_ended():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    after_application(store, app_id, SimulatedClock(case.info.start_time))
    assert store.get(app_id).credit_check is None


def test_directly_the_application_agent_gets_its_six_tools_and_the_given_clock():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    clock = SimulatedClock(case.info.start_time)

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, None, clock, "application") as (tools, given):
            assert given is clock
            assert len(tools) == 6

    asyncio.run(check())


def test_directly_the_offer_agent_gets_its_three_tools_and_the_given_search():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    clock = SimulatedClock(case.info.start_time)
    queries = []

    def search(query):
        queries.append(query)
        return []

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, None, clock, "offer", search) as (tools, given):
            assert given is clock
            tools = {t.name: t for t in tools}
            assert set(tools) == {"retrieve_documents", "request_approval", "send_message"}
            call = {"type": "tool_call", "name": "retrieve_documents", "args": {"query": "late fee"}, "id": "c1"}
            tools["retrieve_documents"].invoke(call)
        assert queries == ["late fee"]

    asyncio.run(check())


def test_directly_the_offer_agent_needs_a_search():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, None, SimulatedClock(case.info.start_time), "offer"):
            pass

    with pytest.raises(ValueError, match="document search"):
        asyncio.run(check())


@pytest.mark.parametrize("agent", ["application", "offer"])
def test_directly_each_agents_events_carry_its_own_prompt_version(agent):
    store = ApplicationStore(agent=AgentVersion(model="m", prompt_version=PROMPT_VERSIONS["application"]))
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    clock = SimulatedClock(case.info.start_time)
    first_tool = {"application": ("check_application", {}), "offer": ("retrieve_documents", {"query": "late fee"})}

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, None, clock, agent, lambda q: []) as (tools, _):
            name, args = first_tool[agent]
            tool = next(t for t in tools if t.name == name)
            tool.invoke({"type": "tool_call", "name": name, "args": args, "id": "c1"})

    asyncio.run(check())
    event = store.get(app_id).audit_log[-1]
    assert (event.model, event.prompt_version) == ("m", PROMPT_VERSIONS[agent])
    assert store.agent.prompt_version == PROMPT_VERSIONS["application"]  # put back afterwards


def open_sessions(conn) -> int:
    return conn.execute("SELECT count(*) FROM case_sessions WHERE closed_at IS NULL").fetchone()[0]


def test_through_mcp_the_session_is_open_only_while_the_agent_works(mcp_url, pg_conn):
    store = PostgresStore(pg_conn, agent=AGENT)
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, mcp_url, SimulatedClock(case.info.start_time), "application") as (tools, clock):
            assert open_sessions(pg_conn) == 1
            assert {t.name for t in tools} >= {"check_application", "finish"}
            assert clock.now == case.info.start_time  # the session's clock
        assert open_sessions(pg_conn) == 0

    asyncio.run(check())


def test_through_mcp_the_session_is_closed_when_the_agent_fails(mcp_url, pg_conn):
    store = PostgresStore(pg_conn, agent=AGENT)
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)

    async def fail():
        async with open_case_for_agent(store, app_id, case.info.start_time, mcp_url, SimulatedClock(case.info.start_time), "application"):
            raise RuntimeError("the agent crashed")

    with pytest.raises(RuntimeError):
        asyncio.run(fail())
    assert open_sessions(pg_conn) == 0


def test_a_reply_is_recorded_with_its_uploads_and_corrections_together():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    now = case.info.start_time + timedelta(days=2)
    new_stub = case.documents["pay_stub_new.json"]
    file = RawFile(filename="stub.pdf", content_type="application/pdf")
    receive_reply(store, app_id, now, "Here is my new pay stub.", {"pay_stub": (new_stub, file)}, {"phone": "(312) 555-0100"})
    stored = store.get(app_id)
    reply = stored.replies[-1]
    assert (reply.received_at, reply.message, reply.uploads, reply.corrected_fields) == (
        now, "Here is my new pay stub.", ["pay_stub"], ["phone"]
    )
    assert stored.document_versions["pay_stub"] == 2 and stored.raw.phone == "(312) 555-0100"
    assert [e.event for e in stored.audit_log][-3:] == ["customer_replied", "document_uploaded", "form_corrected"]
    assert {e.at for e in stored.audit_log[-3:]} == {now}


def test_through_mcp_the_offer_agent_gets_its_tools_and_its_own_session(mcp_url, pg_conn):
    store = PostgresStore(pg_conn, agent=AGENT)
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, mcp_url, SimulatedClock(case.info.start_time), "offer") as (tools, _):
            assert {t.name for t in tools} == {"retrieve_documents", "request_approval", "send_message"}
            assert pg_conn.execute(
                "SELECT agent_name, prompt_version FROM case_sessions WHERE closed_at IS NULL"
            ).fetchone() == ("offer", PROMPT_VERSIONS["offer"])
            tool = next(t for t in tools if t.name == "retrieve_documents")
            await tool.ainvoke({"type": "tool_call", "name": "retrieve_documents", "args": {"query": "late fee"}, "id": "c1"})
        assert open_sessions(pg_conn) == 0

    asyncio.run(check())
    event = store.get(app_id).audit_log[-1]
    assert (event.event, event.prompt_version) == ("documents_retrieved", PROMPT_VERSIONS["offer"])
