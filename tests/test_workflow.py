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
from loans.store import ApplicationStore, CaseEnding
from loans.workflow import after_application, open_case_for_agent

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
    assert [e.event for e in stored.audit_log][-2:] == ["credit_checked", "offer_created"]


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


def test_directly_the_agent_gets_its_six_tools_and_the_given_clock():
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    clock = SimulatedClock(case.info.start_time)

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, None, clock) as (tools, given):
            assert given is clock
            assert len(tools) == 6

    asyncio.run(check())


def open_sessions(conn) -> int:
    return conn.execute("SELECT count(*) FROM case_sessions WHERE closed_at IS NULL").fetchone()[0]


def test_through_mcp_the_session_is_open_only_while_the_agent_works(mcp_url, pg_conn):
    store = PostgresStore(pg_conn, agent=AGENT)
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)

    async def check():
        async with open_case_for_agent(store, app_id, case.info.start_time, mcp_url, SimulatedClock(case.info.start_time)) as (tools, clock):
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
        async with open_case_for_agent(store, app_id, case.info.start_time, mcp_url, SimulatedClock(case.info.start_time)):
            raise RuntimeError("the agent crashed")

    with pytest.raises(RuntimeError):
        asyncio.run(fail())
    assert open_sessions(pg_conn) == 0
