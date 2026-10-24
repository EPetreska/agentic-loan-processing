"""The retrieval eval: how well the RAG search finds the right sections.

    uv run python -m loans.evals.retrieval

It measures the search alone (loans.rag.search), with no agent and no LLM:
each customer question in evals/retrieval/questions.json is searched as the
customer wrote it, and the sections found are compared with the sections
that answer it. Needs Postgres with the index built (loans.rag.index) and
Ollama for the questions' embeddings; takes seconds.

The questions are written as customers ask them, not copying the section
headings, so the search has to match meaning rather than words. A question
lists every section that answers it (two sections can say the same thing).
Some are traps: a section about another product (a CFPB answer about car
loans or mortgages) competes with the bank's own.

Three scores, the standard ones for retrieval:

- hit rate @k: how often a right section is among the k the agent gets
  (k = loans.rag.search.TOP_K);
- MRR (mean reciprocal rank): how high the first right section ranks, 1 for
  first, 1/2 for second and so on, 0 if not found, averaged;
- bank first: for questions about this loan, how often the top section is
  from the bank's terms rather than a CFPB answer.

The report prints the scores and every miss with what was found instead,
and is saved as JSON in runs/retrieval/<run time>/retrieval.json.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from loans.rag.search import TOP_K, SearchResult

ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "evals" / "retrieval" / "questions.json"
RUNS = ROOT / "runs" / "retrieval"


class Question(BaseModel):
    """One question and the sections that answer it."""

    model_config = ConfigDict(extra="forbid")  # written by hand: a typo is an error

    question: str
    expected: list[str] = Field(min_length=1)
    # "loan": about this bank's loan, answered by the bank's terms;
    # "general": answered by a CFPB answer.
    about: Literal["loan", "general"]
    note: str = ""


class QuestionsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    about: str
    questions: list[Question]


class QuestionResult(BaseModel):
    question: str
    about: Literal["loan", "general"]
    expected: list[str]
    found: list[str]
    # The rank of the first right section (1 for first), None if not found.
    rank: int | None
    # For questions about this loan: whether the top section is the bank's.
    bank_first: bool | None


class Report(BaseModel):
    k: int
    hit_rate: float
    mrr: float
    bank_first_rate: float
    results: list[QuestionResult]


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return QuestionsFile.model_validate_json(path.read_text()).questions


def evaluate(questions: list[Question], search: Callable[[str], list[SearchResult]]) -> Report:
    """Search every question and score the sections found."""
    results = []
    for q in questions:
        found = [r.chunk_id for r in search(q.question)]
        rank = next((n for n, chunk_id in enumerate(found, start=1) if chunk_id in q.expected), None)
        bank_first = (bool(found) and found[0].startswith("bank-")) if q.about == "loan" else None
        results.append(
            QuestionResult(
                question=q.question, about=q.about, expected=q.expected,
                found=found, rank=rank, bank_first=bank_first,
            )  # fmt: skip
        )
    loan = [r for r in results if r.about == "loan"]
    return Report(
        k=TOP_K,
        hit_rate=sum(r.rank is not None for r in results) / len(results),
        mrr=sum(1 / r.rank for r in results if r.rank is not None) / len(results),
        bank_first_rate=sum(bool(r.bank_first) for r in loan) / len(loan) if loan else 0.0,
        results=results,
    )


def format_report(report: Report) -> str:
    """The scores, then each question that did not get a right section
    first, with what was found instead."""
    loan = sum(r.about == "loan" for r in report.results)
    lines = [
        f"questions: {len(report.results)}",
        f"hit rate @{report.k}: {report.hit_rate:.0%}",
        f"MRR: {report.mrr:.2f}",
        f"bank first (questions about this loan, {loan}): {report.bank_first_rate:.0%}",
    ]
    not_first = [r for r in report.results if r.rank != 1]
    if not_first:
        lines += ["", "not found first:"]
        for r in not_first:
            where = "missed" if r.rank is None else f"rank {r.rank}"
            lines.append(f"- {r.question!r}: {where}; expected {', '.join(r.expected)}")
            lines.append(f"  found: {', '.join(r.found) or 'nothing'}")
    return "\n".join(lines)


def save_report(report: Report, runs: Path = RUNS) -> Path:
    path = runs / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S") / "retrieval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2) + "\n")
    return path


def main() -> None:
    from dotenv import load_dotenv

    from loans.db.postgres_store import connect, database_conninfo
    from loans.rag.search import search

    load_dotenv(ROOT / ".env")
    with connect(database_conninfo(connect_timeout=3)) as conn:
        report = evaluate(load_questions(), lambda question: search(conn, question))
    print(format_report(report))
    print(f"\nsaved: {save_report(report)}")


if __name__ == "__main__":
    main()
