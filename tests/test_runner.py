"""Tests for the eval runner: choosing cases, judging runs and the report,
with the fake LLM and a fake document search (no Ollama; --direct, so no
Postgres)."""

import json

import pytest
from langchain_core.messages import AIMessage

from loans import run as run_module
from loans.evals import runner
from loans.evals.runner import format_report, pick_cases, run_all
from test_simulate import MARIA, OFFER_AGENT_ONLY, SECTION, FakeModel, ask, step, wait

TO_HUMAN = {"outcome": "sent_to_human", "reason": "agent_unsure", "explanation": "Not sure."}


@pytest.fixture
def direct(monkeypatch, tmp_path):
    """Runs go to a temporary folder, with a fake document search."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(run_module, "RUNS", tmp_path)
    monkeypatch.setattr(runner, "RUNS", tmp_path)

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(run_module, "open_document_search", lambda: (Conn(), lambda query: [SECTION]))
    return tmp_path


def runs(case_names, answers, times=1):
    return run_all(case_names, times, FakeModel(messages=iter(answers)), "fake:model", direct=True)


# Choosing the cases.


def test_cases_by_number_name_kind_or_all():
    assert pick_cases(["09"]) == ["09_offer_late_fee_question"]
    assert pick_cases(["02_maria_job_change", "09"]) == ["02_maria_job_change", "09_offer_late_fee_question"]
    attacks = pick_cases([], kind="attack")
    assert [name[:2] for name in attacks] == ["10", "11", "12", "13"]
    assert pick_cases(["1"], kind="attack") == attacks  # "1" matches 10 to 13, all attacks
    assert len(pick_cases([], everything=True)) == len(runner.all_case_names()) == 13


@pytest.mark.parametrize("names, kind", [(["99"], None), ([], None), (["02"], "attack")])
def test_a_choice_that_matches_nothing_is_refused(names, kind):
    with pytest.raises(ValueError):
        pick_cases(names, kind)


# Judging one run.


def test_a_good_run_passes(direct):
    report, _ = runs(["09_offer_late_fee_question"], OFFER_AGENT_ONLY)
    [verdict] = report.verdicts
    assert (verdict.passed, verdict.problems, verdict.kind) == (True, [], "common")


def test_an_unanswered_question_fails_the_run(direct):
    # As in the first real run of case 09: the answer written as plain text.
    answers = [step("retrieve_documents", {"query": "late fee"}, "o1"), AIMessage("A $25 late fee applies.")]
    report, _ = runs(["09_offer_late_fee_question"], answers)
    [verdict] = report.verdicts
    assert not verdict.passed
    assert verdict.problems == ["check every_message_answered: a customer message at 2026-09-15 15:03 got no reply"]


def test_a_wrong_ending_and_the_expectations_are_reported(direct):
    # Maria's case sent to a human without asking for anything: wrong
    # ending, no income verified, and the pay stub never asked for.
    report, _ = runs(["02_maria_job_change"], [step("finish", TO_HUMAN)])
    [verdict] = report.verdicts
    assert verdict.problems == [
        "ended sent_to_human (agent_unsure), expected complete (all_checks_passed)",
        "income verified: no, expected the opposite",
        "never asked for the pay stub",
    ]


def test_too_many_messages_fail_the_run(direct):
    answers = [ask(call_id="m1"), ask("And your W-2, please.", call_id="m2"), wait(), *MARIA[3:]]
    report, _ = runs(["02_maria_job_change"], answers)
    assert "2 messages to the customer, at most 1 expected" in report.verdicts[0].problems


# Running and the report.


def test_each_case_runs_the_chosen_number_of_times(direct):
    report, run_dir = runs(["09_offer_late_fee_question"], OFFER_AGENT_ONLY * 2, times=2)
    assert [(v.case_id, v.run, v.passed) for v in report.verdicts] == [
        ("09_offer_late_fee_question", 1, True), ("09_offer_late_fee_question", 2, True)
    ]  # fmt: skip
    folders = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    assert [f.rsplit("-", 1)[1] for f in folders] == ["run1", "run2"]  # each run in its own folder
    assert (run_dir / folders[0] / "transcript.json").exists()


def test_a_crash_is_recorded_and_the_runner_goes_on(direct):
    # The fake LLM runs out of answers in the first case.
    report, _ = runs(["02_maria_job_change", "09_offer_late_fee_question"], [step("check_application")])
    first, second = report.verdicts
    assert not first.passed and first.problems[0].startswith("error: RuntimeError: ")
    assert first.problems[0] != "error: RuntimeError: "  # says something, even for an empty error
    assert second.case_id == "09_offer_late_fee_question"  # still run


def test_the_report_counts_passes_and_refused_actions(direct, capsys):
    # The attack case 13 with the agent answering properly.
    answers = [step("send_message", {"text": "Your offer stands at 9.99% APR.", "reason": "Her claim."}, "o1"), wait()]
    report, run_dir = runs(["13_attack_offer_false_approval"], answers)
    text = format_report(report)
    assert "13_attack_offer_false_approval: passed 1 of 1  (refused actions per run: 0)" in text
    assert "total: 1 of 1 runs passed" in text
    saved = json.loads(runner.save_report(report, run_dir).read_text())
    assert saved["verdicts"][0]["case_id"] == "13_attack_offer_false_approval"

