"""Run one eval case with the agent.

    uv run python -m loans.run 01_clean_application
    uv run python -m loans.run 01_clean_application --model ollama:qwen3:8b
    uv run python -m loans.run 01_clean_application --show

With --show, the run is printed as it happens: the agent's thinking (for
models that think out loud) and text word by word while the model writes
them, then its tool calls, the tool results, and the messages that start and
resume the case. The thinking is the model's own
account, a useful view but not a guaranteed record of why it acted.

Submits the case's application, runs the agent on it, and prints the result
and the audit log as a table. Each run gets a folder, with one folder per
case inside:

    runs/<run time>/<case>/
        audit.jsonl          the audit log, written event by event as it happens
        conversation.json    the messages between the agent and the customer,
                             written at the end (also if the run fails halfway)

The conversation is kept so it can be reviewed and checked by the evals; the
audit log itself keeps no message text.

Settings come from .env (see .env.example), API keys from the system keyring
(see KEYRING below): LLM_MODEL chooses the model
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
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from loans.agent.graph import make_model
from loans.agent.prompt import PROMPT_VERSION
from loans.audit import AgentVersion, format_audit_log
from loans.evals.case import EvalCase, load_case
from loans.evals.simulate import simulate_case
from loans.store import ApplicationStore, StoredApplication

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "cases"
RUNS = ROOT / "runs"

# API keys are kept in the system keyring (GNOME keyring, read with
# secret-tool), never in .env or any file in the project. Each run looks up
# only the keys it needs and puts them in the environment for that run.
# Store a key once with, for example:
#     secret-tool store --label="LangSmith API key" service langsmith account apikey
KEYRING = {
    "LANGSMITH_API_KEY": ("langsmith", "apikey"),
    "ANTHROPIC_API_KEY": ("anthropic", "apikey"),
}


def keyring_lookup(service: str, account: str) -> str | None:
    """A secret from the system keyring, or None if it is not there."""
    try:
        found = subprocess.run(
            ["secret-tool", "lookup", "service", service, "account", account],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return found.stdout.strip() or None


def load_keys(model_spec: str) -> None:
    """Put the keys this run needs into the environment, from the keyring
    unless already set: the LangSmith key if tracing is on, and the Anthropic
    key if the model is an Anthropic model. Without a LangSmith key, tracing
    is turned off for the run, with a notice."""
    needed = []
    if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
        needed.append("LANGSMITH_API_KEY")
    if model_spec.startswith("anthropic:"):
        needed.append("ANTHROPIC_API_KEY")
    for name in needed:
        if not os.environ.get(name):
            secret = keyring_lookup(*KEYRING[name])
            if secret:
                os.environ[name] = secret
    if "LANGSMITH_API_KEY" in needed and not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "false"
        print("note: no LangSmith key in the keyring, so tracing is off for this run")

# A limit on the number of loop steps, against an agent that never stops.
MAX_STEPS = 60


def conversation(case: EvalCase, stored: StoredApplication) -> list[dict]:
    """The messages between the agent and the customer, oldest first: the
    customer's message sent with the application, the agent's messages and
    the customer's replies."""
    entries = []
    if case.customer.first_message:
        entries.append(
            {
                "at": stored.metadata.submitted_at,
                "from": "customer",
                "text": case.customer.first_message,
                "with_application": True,
            }
        )
    entries += [
        {"at": m.sent_at, "from": "agent", "text": m.text, "reason": m.reason}
        for m in stored.messages
    ]
    entries += [
        {
            "at": r.received_at,
            "from": "customer",
            "text": r.message,
            "uploads": r.uploads,
            "corrected_fields": r.corrected_fields,
        }
        for r in stored.replies
    ]
    entries.sort(key=lambda entry: entry["at"])  # stable: equal times keep their order
    return [{**entry, "at": entry["at"].isoformat()} for entry in entries]


def write_conversation(path: Path, case: EvalCase, stored: StoredApplication) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conversation(case, stored), indent=2) + "\n")


def format_message(message: BaseMessage, with_text: bool = True) -> str:
    """One message of the conversation, for --show. with_text=False leaves
    out the agent's thinking and text, when they were already printed word by
    word as the model wrote them."""
    if isinstance(message, AIMessage):
        lines = []
        thinking = message.additional_kwargs.get("reasoning_content")
        if thinking and with_text:
            lines.append(_block("thinking", thinking))
        if message.content and with_text:
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


class LivePrinter:
    """Prints a run as it happens, for --show: the model's thinking and text
    word by word while it writes them, then each tool call, tool result and
    message that starts or resumes the case."""

    LABELS = {"thinking": "[thinking] ", "answer": "[agent] "}

    def __init__(self) -> None:
        self._streaming: str | None = None  # what is being printed word by word

    def token(self, kind: str, text: str) -> None:
        if kind != self._streaming:
            if self._streaming is not None:
                print("\n", flush=True)
            print(self.LABELS[kind], end="", flush=True)
            self._streaming = kind
        print(text, end="", flush=True)

    def message(self, message: BaseMessage) -> None:
        if self._streaming is not None:
            print("\n", flush=True)
            self._streaming = None
        text = format_message(message, with_text=not isinstance(message, AIMessage))
        if text:
            print(text, flush=True)
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

    load_keys(model_spec)
    model = make_model(model_spec)  # first, so a wrong setting fails before anything is logged
    case = load_case(CASES / args.case)
    run_dir = RUNS / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    store = ApplicationStore(
        run_dir=run_dir, agent=AgentVersion(model=model_spec, prompt_version=PROMPT_VERSION)
    )
    printer = LivePrinter()
    case_dir = run_dir / case.case_id
    conversation_file = case_dir / "conversation.json"
    try:
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
            **({"on_message": printer.message, "on_token": printer.token} if args.show else {}),
        )
    finally:
        # Also if the run fails halfway, the messages so far are kept.
        if store.has(case.case_id):
            write_conversation(conversation_file, case, store.get(case.case_id))

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
    print(f"files: {case_dir}/ (audit.jsonl, conversation.json)")


if __name__ == "__main__":
    main()
