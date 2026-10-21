"""Tests for splitting the RAG documents into chunks."""

import re

import pytest

from loans.rag import bank_terms, cfpb
from loans.rag.chunks import MAX_CHUNK_CHARS, all_chunks, bank_chunks, cfpb_chunks


@pytest.fixture(scope="module")
def chunks():
    return all_chunks()


def test_each_faq_question_is_its_own_chunk():
    questions = [line[4:] for line in bank_terms.render().splitlines() if line.startswith("### ")]
    titles = [c.title for c in bank_chunks(bank_terms.render())]
    assert all(titles.count(q) == 1 for q in questions)


def test_each_topic_of_the_bank_terms_is_its_own_chunk():
    titles = [c.title for c in bank_chunks(bank_terms.render())]
    for topic in ["Late payments", "Paying early", "Interest rates and APR", "Financial hardship"]:
        assert topic in titles


def test_every_cfpb_answer_has_its_chunks_and_source(chunks):
    for number in cfpb.PAGES:
        mine = [c for c in chunks if c.chunk_id.startswith(f"cfpb-{number}-")]
        assert mine, number
        assert all(c.source.endswith(f"(Ask CFPB answer {number})") for c in mine)
        assert all(c.source.startswith("https://www.consumerfinance.gov/ask-cfpb/") for c in mine)


def test_a_long_answer_is_cut_at_its_subheadings(chunks):
    # Answer 1449 is about 1,600 words.
    pieces = [c for c in chunks if c.chunk_id.startswith("cfpb-1449-")]
    assert len(pieces) > 2
    # Each piece after the first starts at one of the answer's subheadings,
    # so it is one sub-topic, not a cut in the middle of a list or a thought.
    assert all(p.text.startswith("### ") for p in pieces[1:])
    assert len({p.title for p in pieces}) == 1  # every piece keeps the question


def test_no_chunk_is_too_long(chunks):
    assert max(len(c.text) for c in chunks) <= MAX_CHUNK_CHARS


def _lines_kept(markdown: str, chunks, skip_before_first_answer: bool, skip_questions: bool):
    joined = " ".join(re.sub(r"\s+", " ", c.text) for c in chunks)
    started = not skip_before_first_answer
    for line in markdown.splitlines():
        if line.startswith("## "):
            started = True
        text = re.sub(r"\s+", " ", line).strip()
        if not started or not text or line.startswith(("# ", "## ", "Source: ")):
            continue
        if skip_questions and line.startswith("### "):
            continue  # the bank's questions are the chunks' titles
        assert text in joined, text


def test_no_text_of_the_bank_terms_is_lost():
    # Every line, except headings, is in a chunk.
    markdown = bank_terms.render()
    _lines_kept(markdown, bank_chunks(markdown), skip_before_first_answer=False, skip_questions=True)


def test_no_text_of_the_cfpb_answers_is_lost():
    # Every line after the introduction, except the answers' headings and
    # source lines (kept as the chunks' titles and sources), is in a chunk.
    markdown = cfpb.PATH.read_text()
    _lines_kept(markdown, cfpb_chunks(markdown), skip_before_first_answer=True, skip_questions=False)


def test_chunk_ids_are_unique_and_stable(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in all_chunks()]


def test_the_embedded_text_starts_with_the_title(chunks):
    tiny = next(c for c in chunks if c.title == "Can my rate change during the loan?")
    assert tiny.text_to_embed.startswith("Can my rate change during the loan?\n\n")
    assert "fixed" in tiny.text_to_embed
