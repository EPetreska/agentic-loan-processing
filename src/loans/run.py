"""Run one eval case with the agent.

    uv run python -m loans.run 01_clean_application
    uv run python -m loans.run 01_clean_application --model ollama:qwen3:8b

Submits the case's application, runs the agent on it, and prints the result
and the audit log as a table. Every audit event is written to
audit_logs/<run time>/<case>.jsonl as it happens.

Settings come from .env (see .env.example): LLM_MODEL chooses the model
(--model overrides it for one run, handy for comparing models), and the
LANGSMITH_* settings turn on tracing; LangGraph sends the traces by
itself. Time is simulated: the clock starts at the case's start time and moves
one minute per tool call.

Step 1: no customer replies yet, so cases that need an answer from the
customer stop after the agent's first message.

This is the development entry point. In a real deployment, a web service
takes its place: it starts the same agent when a customer submits an
application, again when the customer replies, and again when a daily timer
finds a case whose deadline has passed, using the real clock. The agent, the
tools, the store and the rules stay the same; only how cases come in and the
clock change.
"""

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from loans.agent.graph import build_graph, make_model
from loans.agent.prompt import PROMPT_VERSION, first_turn
from loans.agent.tools import make_tools
from loans.audit import AgentVersion, format_audit_log
from loans.evals.case import load_case, submit_case
from loans.store import ApplicationStore

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "cases"
LOGS = ROOT / "audit_logs"

# A limit on the number of loop steps, against an agent that never stops.
MAX_STEPS = 60


class SimulatedClock:
    """The case's time: starts at the case's start time and moves one minute
    each time it is read."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        now = self._now
        self._now += timedelta(minutes=1)
        return now


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run one eval case with the agent.")
    parser.add_argument("case", help="a case folder name, for example 01_clean_application")
    parser.add_argument(
        "--model", help="the model for this run, for example ollama:qwen3:8b (default: LLM_MODEL)"
    )
    args = parser.parse_args(argv)

    model_spec = args.model or os.environ.get("LLM_MODEL")
    if not model_spec:
        sys.exit("no model: use --model or set LLM_MODEL (see .env.example)")

    model = make_model(model_spec)  # first, so a wrong setting fails before anything is logged
    case = load_case(CASES / args.case)
    log_dir = LOGS / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    store = ApplicationStore(
        log_dir=log_dir, agent=AgentVersion(model=model_spec, prompt_version=PROMPT_VERSION)
    )
    application_id = submit_case(store, case)
    tools = make_tools(store, application_id, SimulatedClock(case.info.start_time))
    graph = build_graph(model, tools, store, application_id)

    result = graph.invoke(
        {"messages": [HumanMessage(first_turn(case.customer.first_message))]},
        config={
            "recursion_limit": MAX_STEPS,
            "run_name": f"case {case.case_id}",
            "tags": [case.case_id],
            "metadata": {"model": model_spec, "prompt_version": PROMPT_VERSION},
        },
    )

    stored = store.get(application_id)
    print(f"case: {case.case_id}  model: {model_spec}  prompt: {PROMPT_VERSION}")
    if stored.ending is not None:
        print(f"ended: {stored.ending.outcome} ({stored.ending.reason})")
    else:
        print("not ended; the agent's last words:")
        print(result["messages"][-1].content)
    print()
    print(format_audit_log(stored.audit_log))
    print()
    print(f"log file: {log_dir / f'{application_id}.jsonl'}")


if __name__ == "__main__":
    main()
