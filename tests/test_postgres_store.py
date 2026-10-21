"""Tests for the Postgres store. They need Postgres running (docker compose
up -d) and are skipped without it (see conftest.py)."""

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from loans.db.postgres_store import PostgresStore, connect, create_tables
from loans.evals.case import load_case, submit_case
from loans.schemas.application import RawApplication, RawFile
from loans.store import ApplicationStore, CustomerReply, UnknownApplication
from loans.tools.application.check_application import check_application
from loans.tools.application.finish import finish
from loans.tools.application.mark_income_verified import mark_income_verified
from loans.tools.application.read_document import read_document
from loans.tools.application.send_message import send_message
from loans.tools.application.verify_identity import verify_identity

CASES = Path(__file__).parent.parent / "evals" / "cases"
PDF = RawFile(filename="stub.pdf", content_type="application/pdf")
NOW = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)


def run_maria(store: ApplicationStore) -> str:
    """Maria's case from submission to the end, using every kind of change:
    a reply, an upload, a form correction and every tool."""
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    start = case.info.start_time
    later = start + timedelta(days=2)
    check_application(store, app_id, start)
    read_document(store, app_id, "pay_stub", start)
    send_message(store, app_id, start, "c1", "Please upload a newer pay stub.", "Pay stub too old.")
    store.record_reply(
        app_id, CustomerReply(received_at=later, message="Here it is.", uploads=["pay_stub"])
    )
    store.upload(app_id, "pay_stub", case.documents["pay_stub_new.json"], PDF, now=later)
    store.correct_fields(app_id, {"phone": "(312) 555-0100"}, now=later)
    read_document(store, app_id, "pay_stub", later)
    verify_identity(store, app_id, later)
    mark_income_verified(store, app_id, later, "c2", "pay_stub", "New stub at the new rate.")
    finish(store, app_id, later, "c3", "complete", "all_checks_passed", "All checks passed.")
    send_message(store, app_id, later, "c4", "One more thing.", "After the end.")  # blocked
    return app_id


def test_same_results_as_the_memory_store(pg_store):
    memory = ApplicationStore()
    app_id = run_maria(memory)
    assert run_maria(pg_store) == app_id
    assert pg_store.get(app_id) == memory.get(app_id)


def test_a_case_survives_a_restart(pg_store, pg_conninfo):
    app_id = run_maria(pg_store)
    before = pg_store.get(app_id)
    with connect(pg_conninfo) as conn:  # a new program, with its own connection
        assert PostgresStore(conn).get(app_id) == before


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE audit_events SET details = 'changed'",
        "DELETE FROM audit_events",
        "TRUNCATE audit_events",
    ],
)
def test_the_audit_log_cannot_be_changed(pg_store, pg_conn, sql):
    app_id = run_maria(pg_store)
    before = pg_store.get(app_id).audit_log
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        pg_conn.execute(sql)
    assert pg_store.get(app_id).audit_log == before


def test_a_change_and_its_event_are_saved_together(pg_store, monkeypatch):
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)

    def broken(event):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(pg_store, "_append_event", broken)
    reply = CustomerReply(received_at=case.info.start_time, message="Hi")
    with pytest.raises(RuntimeError):
        pg_store.record_reply(app_id, reply)
    assert pg_store.get(app_id).replies == []  # the reply was not saved without its event


def test_an_event_is_not_kept_when_its_change_fails(pg_store, monkeypatch):
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)

    def broken(stored):
        raise RuntimeError("the save failed")

    # The reply's event is logged first; then saving the case fails.
    monkeypatch.setattr("loans.db.postgres_store._as_json", broken)
    reply = CustomerReply(received_at=case.info.start_time, message="Hi")
    with pytest.raises(RuntimeError):
        pg_store.record_reply(app_id, reply)
    monkeypatch.undo()
    events = [e.event for e in pg_store.get(app_id).audit_log]
    assert events == ["application_submitted"]  # no event for a reply that was not saved


def test_unknown_application(pg_store):
    assert not pg_store.has("nope")
    with pytest.raises(UnknownApplication):
        pg_store.get("nope")
    with pytest.raises(UnknownApplication):
        pg_store.correct_fields("nope", {"phone": "1"}, now=NOW)
    with pytest.raises(UnknownApplication):
        pg_store.log_event("nope", NOW, "system", "form_corrected")


def test_duplicate_id_is_rejected(pg_store):
    case = load_case(CASES / "01_clean_application")
    submit_case(pg_store, case)
    with pytest.raises(ValueError, match="already exists"):
        pg_store.submit(RawApplication(), {}, case.info.start_time, application_id=case.case_id)
    assert len(pg_store.get(case.case_id).audit_log) == 1  # only the first submission


def test_create_tables_can_run_again(pg_store, pg_conn):
    app_id = run_maria(pg_store)
    create_tables(pg_conn)
    assert pg_store.get(app_id).ending is not None


def test_the_store_needs_autocommit(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn, pytest.raises(ValueError, match="autocommit"):
        PostgresStore(conn)


def test_two_programs_changing_one_case_do_not_lose_a_change(pg_store, pg_conninfo):
    # Two programs (say the MCP server and the simulation) change Maria's case
    # at the same moment, each with its own connection. The first holds its
    # change open; the second must wait for it rather than save over it.
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)
    first = CustomerReply(received_at=case.info.start_time, message="First")
    second = CustomerReply(received_at=case.info.start_time, message="Second")

    with connect(pg_conninfo) as other_conn:
        other = PostgresStore(other_conn)
        program_2 = threading.Thread(target=other.record_reply, args=(app_id, second))
        with pg_store._changing(app_id) as stored:  # program 1, in the middle of a change
            stored.replies.append(first)
            program_2.start()
            time.sleep(0.3)
            assert program_2.is_alive()  # waiting for program 1's lock
        program_2.join(5)

    assert [r.message for r in pg_store.get(app_id).replies] == ["First", "Second"]
