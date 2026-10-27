"""The eval runner: runs the chosen cases, each a chosen number of times,
judges every run and prints one report.

    uv run python -m loans.evals.runner 09 13              # these cases (by number or name)
    uv run python -m loans.evals.runner --kind attack       # only the attack cases
    uv run python -m loans.evals.runner --all --times 3     # every case, 3 times each

With --direct and --model as for run.py (by default through the MCP server,
so Postgres and the server must be running). The cases run one after
another: on a laptop CPU, Ollama answers one request at a time, so running
them in parallel would not be faster.

Each run is the same as run.py's (loans.run.run_case: the files are saved
in its own folder and the automatic checks applied), and is then judged
against the case's expected result (case.json): the ending, whether the
income was verified, the documents asked for, the number of messages, and
how the offer phase closed; and the checks must pass. A run that crashes is
recorded as failed with its error, and the runner goes on with the next.

For attack cases the report also shows, per run, how many actions the code
refused (action_blocked in the audit log): a refused action means the LLM
tried something forbidden (agent level) but the code stopped it; a failed
run means something forbidden got through or the case ended wrongly
(system level).

Each run also records its LLM calls, input and output tokens, and cost in
USD (loans.evals.pricing; 0 for a local model, unknown for a model without a
price). Per case the report shows the median and slowest run time, the
average tokens and cost, and pass^k: whether all k runs passed, the strict
measure of reliability. In total it shows pass@1 (the share of all runs that
passed), pass^k (the share of cases whose every run passed), and the cost per
run and per passed run: a failed run costs money too, so the cost of a
correct result counts the failures as well.

The report (each case's passes out of its runs, the problems of each failed
run, the figures above, and the model and prompt versions) is printed and
saved as runs/runner/<run time>/report.json, apart from single runs of
run.py.
Each run's own files are in runs/runner/<run time>/<case>@<run time>-run<n>/.
"""

import argparse
import os
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from loans.agents.application.prompt import PROMPT_VERSION
from loans.agents.loop import make_model
from loans.agents.offer.prompt import PROMPT_VERSION as OFFER_PROMPT_VERSION
from loans.audit import AgentVersion
from loans.evals.case import CaseKind, EvalCase, load_case
from loans.evals.checks import CheckResult
from loans.evals.pricing import Usage, cost_usd, usage_of
from loans.run import CASES, ROOT, load_keys, open_store, run_case
from loans.store import StoredApplication

# The runner's own folder under runs/, one folder per start of the runner.
RUNS = ROOT / "runs" / "runner"

# The words that count as asking for each document in a message to the
# customer (lowercase), so a model is not failed for its wording.
SLOT_WORDS = {
    "pay_stub": ("pay stub", "paystub", "pay statement", "earnings statement", "pay slip", "payslip"),
    "w2": ("w-2", "w2", "wage and tax statement"),
    "tax_return": ("tax return", "schedule c"),
}


class RunVerdict(BaseModel):
    case_id: str
    kind: CaseKind
    run: int
    passed: bool
    problems: list[str] = []
    # Actions the code refused (action_blocked): the LLM tried, the code stopped it.
    blocked: int = 0
    seconds: float = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # None if unknown: a model without a price, or a run that crashed.
    cost_usd: float | None = None


class Report(BaseModel):
    started_at: datetime
    model: str
    prompt_version: str
    offer_prompt_version: str
    through: str
    times: int
    verdicts: list[RunVerdict]
    # The share of all runs that passed.
    pass_at_1: float = 0
    # The share of cases whose every run passed.
    pass_hat_k: float = 0
    # None if the cost of any run is unknown.
    cost_usd: float | None = None
    cost_per_passed_run_usd: float | None = None


# Choosing the cases.


def all_case_names() -> list[str]:
    return sorted(p.name for p in CASES.iterdir() if p.is_dir())


def pick_cases(names: list[str], kind: str | None = None, everything: bool = False) -> list[str]:
    """The case folders to run: those whose name starts with one of names
    (a number such as "09", or a whole name), those of a kind, or all."""
    available = all_case_names()
    if everything:
        return available
    chosen = []
    for name in names:
        matches = [case for case in available if case.startswith(name)]
        if not matches:
            raise ValueError(f"no case matches {name!r}")
        chosen += [case for case in matches if case not in chosen]
    if kind is not None:
        of_kind = [case for case in available if load_case(CASES / case).info.kind == kind]
        chosen = [case for case in chosen if case in of_kind] if names else of_kind
    if not chosen:
        raise ValueError("no cases chosen: name some, or use --kind or --all")
    return chosen


# Judging one run.


def judge(case: EvalCase, stored: StoredApplication, checks: list[CheckResult]) -> list[str]:
    """What went wrong in a run, compared with the case's expected result
    and the checks; empty if the run passed."""
    expected = case.info.expected
    problems = []
    ending = (stored.ending.outcome, stored.ending.reason) if stored.ending else None
    alternatives = [(e.outcome, e.reason) for e in expected.also_acceptable]
    if ending in alternatives:
        # Another correct ending: judged on the ending and the checks only.
        return _failed_checks(checks)
    if ending is None:
        problems.append("the case did not end")
    elif ending != (expected.outcome, expected.reason):
        also = "".join(f" or {o} ({r})" for o, r in alternatives)
        problems.append(f"ended {ending[0]} ({ending[1]}), expected {expected.outcome} ({expected.reason}){also}")
    if expected.income_verified is not None:
        verified = stored.income_verification is not None
        if verified != expected.income_verified:
            problems.append(f"income verified: {'yes' if verified else 'no'}, expected the opposite")
    first_offer = stored.offers_sent[0].sent_at if stored.offers_sent else None
    application_messages = [m for m in stored.messages if first_offer is None or m.sent_at < first_offer]
    texts = " ".join(m.text.lower() for m in application_messages)
    for slot in expected.must_request:
        if not any(word in texts for word in SLOT_WORDS.get(slot, (slot,))):
            problems.append(f"never asked for the {slot.replace('_', ' ')}")
    if expected.max_messages_to_customer is not None and len(application_messages) > expected.max_messages_to_customer:
        problems.append(
            f"{len(application_messages)} messages to the customer, at most {expected.max_messages_to_customer} expected"
        )
    if expected.offer_outcome is not None:
        closed = stored.offer_closing.outcome if stored.offer_closing else "not closed"
        if closed != expected.offer_outcome:
            problems.append(f"offer phase {closed}, expected {expected.offer_outcome}")
    return problems + _failed_checks(checks)


def _failed_checks(checks: list[CheckResult]) -> list[str]:
    return [f"check {c.check}: {problem}" for c in checks if not c.passed for problem in c.problems]


def blocked_actions(stored: StoredApplication) -> int:
    return sum(event.event == "action_blocked" for event in stored.audit_log)


# The figures.


def by_case(verdicts: list[RunVerdict]) -> dict[str, list[RunVerdict]]:
    cases: dict[str, list[RunVerdict]] = {}
    for verdict in verdicts:
        cases.setdefault(verdict.case_id, []).append(verdict)
    return cases


def pass_at_1(verdicts: list[RunVerdict]) -> float:
    """The share of all runs that passed."""
    return sum(v.passed for v in verdicts) / len(verdicts) if verdicts else 0


def pass_hat_k(verdicts: list[RunVerdict]) -> float:
    """The share of cases whose every run passed."""
    cases = by_case(verdicts)
    return sum(all(v.passed for v in runs) for runs in cases.values()) / len(cases) if cases else 0


def total_cost(verdicts: list[RunVerdict]) -> float | None:
    """The cost of all runs, or None if the cost of any run is unknown."""
    costs = [v.cost_usd for v in verdicts]
    return None if None in costs else round(sum(costs), 6)


def cost_per_passed_run(verdicts: list[RunVerdict]) -> float | None:
    """The cost of all runs, failed ones included, divided by the runs that
    passed: what one correct result costs. None if unknown or none passed."""
    cost, passed = total_cost(verdicts), sum(v.passed for v in verdicts)
    return None if cost is None or not passed else round(cost / passed, 6)


# The report.


def _usd(cost: float | None) -> str:
    return "unknown" if cost is None else f"${cost:.4f}"


def format_report(report: Report) -> str:
    lines = [
        f"model: {report.model}  prompts: {report.prompt_version} (application), "
        f"{report.offer_prompt_version} (offer)  tools: {report.through}",
        "",
    ]
    for case_id, verdicts in by_case(report.verdicts).items():
        passed = sum(v.passed for v in verdicts)
        line = f"{case_id}: passed {passed} of {len(verdicts)}"
        if len(verdicts) > 1:
            line += f"  pass^{len(verdicts)}: {'yes' if passed == len(verdicts) else 'no'}"
        if verdicts[0].kind == "attack":
            line += f"  (refused actions per run: {', '.join(str(v.blocked) for v in verdicts)})"
        lines.append(line)
        seconds = [v.seconds for v in verdicts]
        cost = total_cost(verdicts)
        lines.append(
            f"    time: median {statistics.median(seconds):.0f} s, slowest {max(seconds):.0f} s"
            f"  tokens per run: {sum(v.input_tokens for v in verdicts) // len(verdicts):,} in, "
            f"{sum(v.output_tokens for v in verdicts) // len(verdicts):,} out"
            f"  cost per run: {_usd(None if cost is None else cost / len(verdicts))}"
        )
        for v in verdicts:
            for problem in v.problems:
                lines.append(f"    run {v.run}: {problem}")
    total = len(report.verdicts)
    lines += [
        "",
        f"total: {sum(v.passed for v in report.verdicts)} of {total} runs passed",
        f"pass@1: {report.pass_at_1:.0%}  pass^{report.times}: {report.pass_hat_k:.0%} of cases",
        f"cost: {_usd(report.cost_usd)} in all, "
        f"{_usd(None if report.cost_usd is None or not total else report.cost_usd / total)} per run, "
        f"{_usd(report.cost_per_passed_run_usd)} per passed run",
    ]
    return "\n".join(lines)


def save_report(report: Report, run_dir: Path) -> Path:
    path = run_dir / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")
    return path


# Running.


def run_all(
    case_names: list[str],
    times: int,
    new_model: Callable[[], object],
    model_spec: str,
    direct: bool,
    show: bool = False,
) -> tuple[Report, Path]:
    """Run each case the given number of times, one after another, judging
    each run; print one line per run as it finishes.

    Each run gets a fresh model from new_model: every run has its own event
    loop (run_case uses asyncio.run), and a model's connection to Ollama
    stays tied to the loop of its first run, so a reused model fails in the
    next run with "Event loop is closed"."""
    started = datetime.now(UTC)
    run_dir = RUNS / started.strftime("%Y-%m-%dT%H-%M-%S")
    store, mcp_url = open_store(direct, run_dir, AgentVersion(model=model_spec, prompt_version=PROMPT_VERSION))
    verdicts = []
    for case_name in case_names:
        case = load_case(CASES / case_name)
        for n in range(1, times + 1):
            application_id = f"{case.case_id}@{run_dir.name}-run{n}"
            began = time.monotonic()
            try:
                model = new_model()
                run = run_case(case, store, model, model_spec, application_id, run_dir / application_id, mcp_url, show)
                problems = judge(case, run.stored, run.checks)
                blocked = blocked_actions(run.stored)
                usage = usage_of(run.result.messages + run.result.offer_messages)
                cost = cost_usd(model_spec, usage)
            except Exception as e:  # recorded, and the runner goes on
                problems, blocked = [f"error: {type(e).__name__}: {str(e) or 'no message'}"], 0
                # What a crashed run used is lost with it.
                usage, cost = Usage(), None
            verdict = RunVerdict(
                case_id=case.case_id, kind=case.info.kind, run=n, passed=not problems,
                problems=problems, blocked=blocked, seconds=round(time.monotonic() - began, 1),
                llm_calls=usage.llm_calls, input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens, cost_usd=cost,
            )  # fmt: skip
            verdicts.append(verdict)
            print(
                f"{case.case_id} run {n}/{times}: {'passed' if verdict.passed else 'FAILED'} ({verdict.seconds:.0f} s, "
                f"{verdict.llm_calls} LLM calls, {verdict.input_tokens:,} + {verdict.output_tokens:,} tokens, "
                f"cost {_usd(verdict.cost_usd)})",
                flush=True,
            )
    report = Report(
        started_at=started,
        model=model_spec,
        prompt_version=PROMPT_VERSION,
        offer_prompt_version=OFFER_PROMPT_VERSION,
        through="direct" if direct else f"MCP server at {mcp_url}",
        times=times,
        verdicts=verdicts,
        pass_at_1=pass_at_1(verdicts),
        pass_hat_k=pass_hat_k(verdicts),
        cost_usd=total_cost(verdicts),
        cost_per_passed_run_usd=cost_per_passed_run(verdicts),
    )
    return report, run_dir


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run chosen eval cases and judge them.")
    parser.add_argument("cases", nargs="*", help="cases by number or name, for example 09 13")
    parser.add_argument("--kind", help="only cases of this kind, for example attack")
    parser.add_argument("--all", action="store_true", help="every case")
    parser.add_argument("--times", type=int, default=1, help="runs per case (default 1)")
    parser.add_argument("--model", help="the model (default: LLM_MODEL)")
    parser.add_argument("--direct", action="store_true", help="without the MCP server, as run.py --direct")
    parser.add_argument("--show", action="store_true", help="print each step, as run.py --show")
    args = parser.parse_args(argv)

    try:
        case_names = pick_cases(args.cases, args.kind, args.all)
    except ValueError as e:
        sys.exit(str(e))
    if args.times < 1:
        sys.exit("--times must be at least 1")
    model_spec = args.model or os.environ.get("LLM_MODEL")
    if not model_spec:
        sys.exit("no model: use --model or set LLM_MODEL (see .env.example)")
    load_keys(model_spec)
    make_model(model_spec)  # first, so a wrong setting fails before any run

    print(f"running {len(case_names)} case(s), {args.times} time(s) each: {', '.join(case_names)}\n", flush=True)
    report, run_dir = run_all(case_names, args.times, lambda: make_model(model_spec), model_spec, args.direct, args.show)
    print()
    print(format_report(report))
    print(f"\nsaved: {save_report(report, run_dir)}")


if __name__ == "__main__":
    main()
