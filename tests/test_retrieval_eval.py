"""Tests for the retrieval eval: the questions file stays valid against the
current sections, and the scores are computed right. A fake search is
used, so no database or Ollama is needed."""

import json

import pytest
from pydantic import ValidationError

from loans.evals.retrieval import Question, evaluate, format_report, load_questions, save_report
from loans.rag.chunks import all_chunks
from loans.rag.search import SearchResult


def section(chunk_id):
    return SearchResult(chunk_id=chunk_id, document=chunk_id.split("-")[0], title="t", text="x",
                        source="s", distance=0.1)  # fmt: skip


def fake_search(answers):
    """A search that returns fixed section IDs for each question."""
    return lambda question: [section(chunk_id) for chunk_id in answers[question]]


QUESTIONS = [
    Question(question="late?", expected=["bank-06", "bank-13"], about="loan"),
    Question(question="early?", expected=["bank-07"], about="loan"),
    Question(question="dti?", expected=["cfpb-1791-1"], about="general"),
    Question(question="freeze?", expected=["cfpb-1341-1"], about="general"),
]
FOUND = {
    "late?": ["cfpb-839-1", "bank-13", "bank-05", "bank-06"],  # second
    "early?": ["bank-07", "bank-12"],  # first
    "dti?": ["cfpb-1791-1"],  # first
    "freeze?": ["cfpb-5-1", "cfpb-1253-1"],  # missed
}


def test_every_expected_section_exists():
    # If the chunking changes, a question pointing at a section that no
    # longer exists fails here, instead of quietly lowering the scores.
    ids = {chunk.chunk_id for chunk in all_chunks()}
    for q in load_questions():
        assert set(q.expected) <= ids, q.question


def test_the_questions_match_their_kind():
    questions = load_questions()
    assert len(questions) == 24
    for q in questions:
        prefix = "bank-" if q.about == "loan" else "cfpb-"
        assert all(chunk_id.startswith(prefix) for chunk_id in q.expected), q.question
    assert len({q.question for q in questions}) == len(questions)


def test_a_typo_in_a_question_is_an_error():
    with pytest.raises(ValidationError):
        Question.model_validate({"question": "x", "expected": ["bank-01"], "about": "loan", "expectd": []})


def test_the_scores():
    report = evaluate(QUESTIONS, fake_search(FOUND))
    assert report.hit_rate == 3 / 4
    assert report.mrr == pytest.approx((1 / 2 + 1 + 1 + 0) / 4)
    assert report.bank_first_rate == 1 / 2  # "late?" got a CFPB page first
    assert [r.rank for r in report.results] == [2, 1, 1, None]


def test_the_report_lists_what_was_not_found_first():
    text = format_report(evaluate(QUESTIONS, fake_search(FOUND)))
    assert "hit rate @4: 75%" in text and "MRR: 0.62" in text and "bank first (questions about this loan, 2): 50%" in text
    assert "- 'late?': rank 2; expected bank-06, bank-13" in text
    assert "  found: cfpb-839-1, bank-13, bank-05, bank-06" in text
    assert "- 'freeze?': missed; expected cfpb-1341-1" in text
    assert "'early?'" not in text  # found first: not listed


def test_the_report_is_saved(tmp_path):
    report = evaluate(QUESTIONS, fake_search(FOUND))
    path = save_report(report, tmp_path)
    assert path.name == "retrieval.json"
    assert json.loads(path.read_text())["hit_rate"] == 0.75
