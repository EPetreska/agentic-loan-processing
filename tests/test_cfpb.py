"""Tests for the CFPB consumer guides: cleaning a downloaded page, and the
committed document. No test downloads anything."""

from datetime import date
from pathlib import Path

from loans.rag import cfpb

PAGE = (Path(__file__).parent / "fixtures" / "cfpb_answer_2114.html").read_text()
FURNITURE = ["Don't see what you're looking for?", "Browse related questions", "Learn more about", "Search"]


def test_the_question_and_answer_are_kept():
    question, paragraphs = cfpb.clean(PAGE)
    assert question == "What is a personal installment loan?"
    assert paragraphs[0].startswith("A personal installment loan is a type of loan")
    assert "### What happens if I miss a payment or fail to pay back my personal installment loan?" in paragraphs
    assert "- Income" in paragraphs  # list items stay list items


def test_the_page_furniture_is_dropped():
    _, paragraphs = cfpb.clean(PAGE)
    text = "\n".join(paragraphs)
    assert not any(line.startswith(f) for line in paragraphs for f in FURNITURE)
    assert " ." not in text and " ," not in text  # no gaps left by links


def test_each_answer_has_its_heading_and_source():
    question, paragraphs = cfpb.clean(PAGE)
    address = "https://www.consumerfinance.gov/ask-cfpb/what-is-a-personal-installment-loan-en-2114/"
    text = cfpb.render([(2114, address, question, paragraphs)], date(2026, 10, 21))
    assert "## What is a personal installment loan?" in text
    assert f"Source: {address} (Ask CFPB answer 2114)" in text
    assert "downloaded on 2026-10-21" in text


def test_the_committed_document_has_every_chosen_answer():
    text = cfpb.PATH.read_text()
    for number in cfpb.PAGES:
        assert f"(Ask CFPB answer {number})" in text
    assert not any(line.startswith(f) for line in text.splitlines() for f in FURNITURE)


def test_an_answer_is_found_by_its_number():
    # Any words before the number reach the answer.
    assert cfpb.page_url(2114) == "https://www.consumerfinance.gov/ask-cfpb/answer-en-2114/"
