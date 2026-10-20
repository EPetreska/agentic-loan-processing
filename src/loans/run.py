"""Run one eval case with the agent.

    uv run python -m loans.run 01_clean_application
    uv run python -m loans.run 01_clean_application --model ollama:qwen3:8b
    uv run python -m loans.run 01_clean_application --show

With --show, every step is printed as it happens: the agent's thinking (for
models that think out loud), its tool calls, the tool results, and the
messages that start and resume the case. The thinking is the model's own
account, a useful view but not a guaranteed record of why it acted.

Submits the case's application, runs the agent on it, and prints the result
and the audit log as a table. Every audit event is written to
audit_logs/<run time>/<case>.jsonl as it happens.

Settings come from .env (see .env.example): LLM_MODEL chooses the model
(--model overrides it for one run, handy for comparing models), and the
LANGSMITH_* settings turn on tracing; LangGraph sends the traces by
itself. Time is simulated: the clock starts at the case's start time and moves
one minute per tool call, and jumps forward when the customer takes days to
answer (see loans.evals.simulate, which plays the scripted customer).

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
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from loans.agent.graph import make_model
from loans.agent.prompt import PROMPT_VERSION
from loans.audit import AgentVersion, format_audit_log
from loans.evals.case import load_case
from loans.evals.simulate import simulate_case
from loans.store import ApplicationStore

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "cases"
LOGS = ROOT / "audit_logs"

# A limit on the number of loop steps, against an agent that never stops.
MAX_STEPS = 60


def format_message(message: BaseMessage) -> str:
    """One message of the conversation, for --show."""
    if isinstance(message, AIMessage):
        lines = []
        thinking = message.additional_kwargs.get("reasoning_content")
        if thinking:
            lines.append(_block("thinking", thinking))
        if message.content:
            lines.append(_block("agent", str(message.content)))
        for call in message.tool_calls:
            args = ", ".join(f"{name}={value!r}" for name, value in call["args"].items())
            lines.append(f"[agent] calls {call['name']}({args})")
        return "\n".join(lines)
    if isinstance(message, ToolMessage):
        return _block(f"tool {message.name}", str(message.content))
    return _block("news", str(message.content))


def _block(label: str, text: str) -> str:
    """A labeled block; continuation lines are indented under the label."""
    first, *rest = text.strip().splitlines() or [""]
    return "\n".join([f"[{label}] {first}", *(f"    {line}" for line in rest)])


def _print_message(message: BaseMessage) -> None:
    print(format_message(message), flush=True)
    print(flush=True)


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run one eval case with the agent.")
    parser.add_argument("case", help="a case folder name, for example 01_clean_application")
    parser.add_argument(
        "--model", help="the model for this run, for example ollama:qwen3:8b (default: LLM_MODEL)"
    )
    parser.add_argument(
        "--show", action="store_true", help="print each step as it happens, with the thinking"
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
    result = simulate_case(
        case,
        store,
        model,
        config={
            "recursion_limit": MAX_STEPS,
            "run_name": f"case {case.case_id}",
            "tags": [case.case_id],
            "metadata": {"model": model_spec, "prompt_version": PROMPT_VERSION},
        },
        on_message=_print_message if args.show else None,
    )

    application_id = result.application_id
    stored = store.get(application_id)
    print(f"case: {case.case_id}  model: {model_spec}  prompt: {PROMPT_VERSION}")
    print(f"pauses to wait for the customer: {result.pauses}")
    if stored.ending is not None:
        print(f"ended: {stored.ending.outcome} ({stored.ending.reason})")
    else:
        if result.stopped_by_limit:
            print("not ended: the agent kept waiting, so the simulation stopped")
        print("the agent's last words:")
        print(result.messages[-1].content)
    print()
    print(format_audit_log(stored.audit_log))
    print()
    print(f"log file: {log_dir / f'{application_id}.jsonl'}")


if __name__ == "__main__":
    main()
