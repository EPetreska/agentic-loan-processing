"""Tests for the run script: its --show output, the files a run leaves, and
what it says when Postgres or the MCP server is not running."""

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import AGENT
from loans import run
from loans.agents.offer.prompt import PROMPT_VERSION as OFFER_PROMPT_VERSION
from loans.audit import read_audit_log
from loans.evals.case import load_case, submit_case
from loans.run import LivePrinter, conversation, format_message, write_conversation
from loans.store import ApplicationStore

CASES = Path(__file__).parent.parent / "evals" / "cases"


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


def test_without_text_only_the_tool_calls_are_shown():
    message = AIMessage(
        content="Let me check.",
        additional_kwargs={"reasoning_content": "Thinking..."},
        tool_calls=[{"name": "verify_identity", "args": {}, "id": "c1"}],
    )
    assert format_message(message, with_text=False) == "[agent] calls verify_identity()"


def test_live_printer_labels_each_stream_once(capsys):
    printer = LivePrinter()
    for word in ["The ", "form ", "is ", "fine."]:
        printer.token("thinking", word)
    printer.token("answer", "Checking now.")
    call = {"name": "verify_identity", "args": {}, "id": "c1"}
    printer.message(AIMessage(content="", tool_calls=[call]))
    out = capsys.readouterr().out
    assert out == (
        "[thinking] The form is fine.\n\n"
        "[agent] Checking now.\n\n"
        "[agent] calls verify_identity()\n\n"
    )


def maria_after_a_simulated_run():
    """Maria's case through the scripted customer, with a fake LLM."""
    from test_simulate import MARIA, FakeModel, run_simulation  # the same scripted run

    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    result = run_simulation(case, store, FakeModel(messages=iter(MARIA)))
    return case, store.get(result.application_id)


def test_conversation_in_order():
    case, stored = maria_after_a_simulated_run()
    entries = conversation(stored)
    assert [(e["from"], e["text"]) for e in entries] == [
        ("customer", case.customer.first_message),
        ("agent", "Please upload a recent pay stub."),
        ("customer", "Here is my latest pay stub with the new pay rate."),
    ]
    assert entries[0]["with_application"] is True
    assert entries[1]["reason"] == "Missing document."
    assert entries[2]["uploads"] == ["pay_stub"]
    assert entries[0]["at"] == "2026-09-14T15:00:00+00:00"


def test_conversation_file(tmp_path):
    case, stored = maria_after_a_simulated_run()
    path = tmp_path / "run" / "02_maria_job_change.conversation.json"
    write_conversation(path, stored)
    assert json.loads(path.read_text()) == conversation(stored)


def test_conversation_without_any_messages():
    store = ApplicationStore()
    case = load_case(CASES / "01_clean_application")
    stored = store.get(submit_case(store, case))
    assert conversation(stored) == []


def fake_keyring(monkeypatch, secrets):
    looked_up = []

    def lookup(service, account):
        looked_up.append(service)
        return secrets.get(service)

    monkeypatch.setattr(run, "keyring_lookup", lookup)
    return looked_up


def test_keys_come_from_the_keyring(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    fake_keyring(monkeypatch, {"langsmith": "ls-key"})
    run.load_keys("ollama:qwen3:4b")
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_only_the_needed_keys_are_looked_up(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    looked_up = fake_keyring(monkeypatch, {"anthropic": "a-key", "langsmith": "ls-key"})
    run.load_keys("ollama:qwen3:4b")
    assert looked_up == []  # no tracing, no Anthropic model: no keys at all
    run.load_keys("anthropic:claude-sonnet-5-5")
    assert looked_up == ["anthropic"]
    assert os.environ["ANTHROPIC_API_KEY"] == "a-key"


def test_without_a_langsmith_key_tracing_is_turned_off(monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    fake_keyring(monkeypatch, {})
    run.load_keys("ollama:qwen3:4b")
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "tracing is off" in capsys.readouterr().out


def test_a_key_already_set_is_kept(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "from-the-shell")
    looked_up = fake_keyring(monkeypatch, {"langsmith": "from-the-keyring"})
    run.load_keys("ollama:qwen3:4b")
    assert os.environ["LANGSMITH_API_KEY"] == "from-the-shell"
    assert looked_up == []


# The run script through MCP. These need Postgres running (docker compose
# up -d) and are skipped without it.


def test_without_postgres_the_run_says_how_to_start_it(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://loans@127.0.0.1:9/loans")  # nothing listens there
    with pytest.raises(SystemExit, match="docker compose up -d"):
        run.open_bank_store(AGENT)


def test_without_the_server_the_run_says_how_to_start_it(monkeypatch, pg_conninfo):
    monkeypatch.setenv("DATABASE_URL", pg_conninfo)
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:9/mcp")  # nothing listens there
    with pytest.raises(SystemExit, match="uv run python -m loans.mcp.server"):
        run.open_bank_store(AGENT)


def test_a_run_through_mcp_leaves_its_files(monkeypatch, tmp_path, mcp_url, pg_conninfo, capsys):
    from test_simulate import MARIA, FakeModel

    monkeypatch.setenv("DATABASE_URL", pg_conninfo)
    monkeypatch.setenv("MCP_URL", mcp_url)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(run, "make_model", lambda spec: FakeModel(messages=iter(MARIA)))
    run.main(["02_maria_job_change", "--model", "fake:model"])

    (case_dir,) = tmp_path.glob("*/02_maria_job_change@*")  # one folder, named by the run's ID
    events = read_audit_log(case_dir / "audit.jsonl")
    assert [e.event for e in events][-4:] == [
        "application_completed", "credit_checked", "offer_created", "offer_sent"
    ]
    assert {e.model for e in events if e.actor == "agent"} == {"fake:model"}
    assert json.loads((case_dir / "conversation.json").read_text())[1]["from"] == "agent"
    out = capsys.readouterr().out
    assert "ended: complete (all_checks_passed)" in out and f"MCP server at {mcp_url}" in out
    assert "credit check: passed" in out
    assert "offer: 48 months at 10.99% APR: $387.61 a month" in out



# The offer phase in a run: cases 05 to 08 go on after the application.


class ClosedOnce:
    """A stand-in for the search's database connection, which the run closes."""

    closed = False

    def close(self):
        self.closed = True


def run_directly(monkeypatch, tmp_path, case_name, answers):
    from test_simulate import SECTION, FakeModel

    conn = ClosedOnce()
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(run, "make_model", lambda spec: FakeModel(messages=iter(answers)))
    monkeypatch.setattr(run, "open_document_search", lambda: (conn, lambda query: [SECTION]))
    run.main([case_name, "--model", "fake:model", "--direct"])
    (case_dir,) = tmp_path.glob(f"*/{case_name}@*")
    return conn, case_dir


def test_a_direct_run_shows_the_offer_phase(monkeypatch, tmp_path, capsys):
    from test_simulate import MARIA, OFFER_AGENT

    conn, case_dir = run_directly(monkeypatch, tmp_path, "05_maria_asks_and_signs", MARIA + OFFER_AGENT)
    out = capsys.readouterr().out
    assert f"offer prompt: {OFFER_PROMPT_VERSION}" in out
    assert "offer 1 sent on 2026-09-16: $15,000, 48 months" in out
    assert "documents retrieved for 'prepayment penalty': bank-07" in out
    assert "signed: offer 1 on" in out and "offer phase closed: accepted on" in out
    assert conn.closed  # the search's connection is closed at the end
    entries = json.loads((case_dir / "conversation.json").read_text())
    assert [e.get("agent") for e in entries if e["from"] == "agent"] == ["application", "offer"]


def test_a_direct_run_shows_the_officers_decision(monkeypatch, tmp_path, capsys):
    from test_simulate import MARIA, asks_officer

    answers = MARIA + asks_officer("Borrow $12,000.", "A loan officer will decide.", "Approved.")
    run_directly(monkeypatch, tmp_path, "07_maria_asks_for_less", answers)
    out = capsys.readouterr().out
    assert "request 1 to a loan officer: approved by officer-17" in out
    assert "offer 2 sent on" in out and ": $12,000, 48 months" in out
    assert "signed: offer 2 on" in out


def test_a_case_without_an_offer_phase_needs_no_search(monkeypatch, tmp_path, capsys):
    from test_simulate import MARIA, FakeModel

    def no_search():
        raise AssertionError("the search was opened")

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(run, "make_model", lambda spec: FakeModel(messages=iter(MARIA)))
    monkeypatch.setattr(run, "open_document_search", no_search)
    run.main(["02_maria_job_change", "--model", "fake:model", "--direct"])
    out = capsys.readouterr().out
    assert "offer prompt:" not in out and "offer phase closed" not in out
    assert "offer 1 sent on" in out  # the offer is still sent by the bank


def test_a_direct_offer_phase_without_postgres_says_how_to_start_it(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://loans@127.0.0.1:9/loans")  # nothing listens there
    with pytest.raises(SystemExit, match="docker compose up -d"):
        run.open_document_search()
