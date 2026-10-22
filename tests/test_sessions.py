"""Tests for sessions: tokens that open one case, and the shared clock. They
need Postgres running (docker compose up -d) and are skipped without it."""

from datetime import timedelta
from pathlib import Path

import psycopg
import pytest

from loans.audit import AgentVersion
from loans.db.postgres_store import connect
from loans.evals.case import load_case, submit_case
from loans.mcp.sessions import (
    InvalidToken,
    SessionClock,
    close_session,
    find_session,
    open_session,
)
from loans.store import UnknownApplication

CASES = Path(__file__).parent.parent / "evals" / "cases"
AGENT = AgentVersion(model="ollama:qwen3:4b", prompt_version="v1")


@pytest.fixture
def maria(pg_store):
    """Maria's case, submitted: (its ID, its start time)."""
    case = load_case(CASES / "02_maria_job_change")
    return submit_case(pg_store, case), case.info.start_time


def test_a_token_finds_its_case_and_agent(pg_conn, maria):
    app_id, start = maria
    token = open_session(pg_conn, app_id, "application", start, AGENT)
    session = find_session(pg_conn, token)
    assert session.application_id == app_id
    assert session.agent == AGENT
    assert session.agent_name == "application"


@pytest.mark.parametrize("agent_name", ["application", "offer"])
def test_a_session_remembers_which_agent_it_is_for(pg_conn, maria, agent_name):
    app_id, start = maria
    token = open_session(pg_conn, app_id, agent_name, start)
    assert find_session(pg_conn, token).agent_name == agent_name


def test_a_session_for_an_unknown_agent_is_refused(pg_conn, maria):
    # The database itself refuses it, whatever the code passes.
    app_id, start = maria
    with pytest.raises(psycopg.errors.CheckViolation):
        open_session(pg_conn, app_id, "officer", start)


def test_each_token_opens_only_its_own_case(pg_conn, pg_store, maria):
    maria_id, start = maria
    daniel_id = submit_case(pg_store, load_case(CASES / "01_clean_application"))
    maria_token = open_session(pg_conn, maria_id, "application", start)
    daniel_token = open_session(pg_conn, daniel_id, "application", start)
    assert find_session(pg_conn, maria_token).application_id == maria_id
    assert find_session(pg_conn, daniel_token).application_id == daniel_id


@pytest.mark.parametrize("token", ["", "not-a-token", "02_maria_job_change"])
def test_an_unknown_token_finds_nothing(pg_conn, maria, token):
    app_id, start = maria
    open_session(pg_conn, app_id, "application", start)
    assert find_session(pg_conn, token) is None


def test_the_token_is_not_kept_in_the_database(pg_conn, maria):
    app_id, start = maria
    token = open_session(pg_conn, app_id, "application", start)
    rows = pg_conn.execute("SELECT * FROM case_sessions").fetchall()
    assert token not in str(rows)


def test_a_closed_session_refuses_its_token(pg_conn, maria):
    app_id, start = maria
    token = open_session(pg_conn, app_id, "application", start)
    close_session(pg_conn, token)
    assert find_session(pg_conn, token) is None
    clock = SessionClock(pg_conn, token)
    with pytest.raises(InvalidToken):
        clock()
    with pytest.raises(InvalidToken):
        clock.now
    with pytest.raises(InvalidToken):
        clock.jump_to(start + timedelta(days=1))


def test_one_open_session_per_case(pg_conn, maria):
    app_id, start = maria
    first = open_session(pg_conn, app_id, "application", start)
    with pytest.raises(ValueError, match="already has an open session"):
        open_session(pg_conn, app_id, "application", start)
    close_session(pg_conn, first)
    second = open_session(pg_conn, app_id, "application", start)  # a new run may start once it is closed
    assert find_session(pg_conn, second).application_id == app_id


def test_no_session_for_an_unknown_case(pg_conn, maria):
    _, start = maria
    with pytest.raises(UnknownApplication):
        open_session(pg_conn, "nope", "application", start)


def test_the_clock_moves_one_minute_per_step(pg_conn, maria):
    app_id, start = maria
    clock = SessionClock(pg_conn, open_session(pg_conn, app_id, "application", start))
    assert clock.now == start
    assert clock() == start
    assert clock() == start + timedelta(minutes=1)
    assert clock.now == start + timedelta(minutes=2)


def test_the_clock_jumps_forward_never_back(pg_conn, maria):
    app_id, start = maria
    clock = SessionClock(pg_conn, open_session(pg_conn, app_id, "application", start))
    later = start + timedelta(days=2)
    clock.jump_to(later)
    assert clock.now == later
    clock.jump_to(start)
    assert clock.now == later


def test_the_clock_is_shared_between_programs(pg_conn, pg_conninfo, maria):
    # The MCP server and the simulation each have their own connection.
    app_id, start = maria
    token = open_session(pg_conn, app_id, "application", start)
    SessionClock(pg_conn, token).jump_to(start + timedelta(days=2))
    with connect(pg_conninfo) as other:
        assert SessionClock(other, token)() == start + timedelta(days=2)
    assert SessionClock(pg_conn, token).now == start + timedelta(days=2, minutes=1)
