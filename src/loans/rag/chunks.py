"""Splitting the RAG documents into chunks: the sections the search returns.

Two stages, the usual practice for documents with clear headings:

1. By structure (LangChain's Markdown header splitter): one chunk per topic
   and per FAQ question of the bank's terms, and one per CFPB answer, so a
   chunk is one complete topic or answer.
2. By size, only for sections that are still too long (some CFPB answers
   reach about 1,600 words): LangChain's recursive splitter cuts them at the
   answer's own subheadings first, then between paragraphs, then between
   lines, with a small overlap so a thought cut in two keeps some context.

Each chunk keeps its title (the FAQ question, the topic or the CFPB question)
and its source (the bank's document, or the CFPB answer's address and
number), so the agent can name where an answer came from. The text that is
embedded starts with the title, so the search knows what a chunk is about
even when it is a piece cut from the middle of a long answer.
"""

import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel

from loans.rag import bank_terms, cfpb

# A section longer than this is cut by size. About 300 words: long enough for
# a complete answer, short enough to return a few at once.
MAX_CHUNK_CHARS = 1800
OVERLAP_CHARS = 150

BANK_SOURCE = "Personal Loan Terms and Frequently Asked Questions (the bank's own document)"

_SOURCE_LINE = re.compile(r"^Source: (\S+) \(Ask CFPB answer (\d+)\)$", re.MULTILINE)


class Chunk(BaseModel):
    # Stable and readable, for example "bank-07" or "cfpb-2114-2".
    chunk_id: str
    # "bank" (the bank's terms) or "cfpb".
    document: str
    title: str
    text: str
    # Where the chunk comes from, for the agent to name.
    source: str

    @property
    def text_to_embed(self) -> str:
        return f"{self.title}\n\n{self.text}"


_by_size = RecursiveCharacterTextSplitter(
    # The answer's subheadings first, then lines (in the CFPB document each
    # paragraph and list item is one line), then words.
    separators=["\n### ", "\n", " "],
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=OVERLAP_CHARS,
)


def _section_text(page_content: str) -> str:
    """The header splitter joins a section's lines with Markdown line breaks
    (two spaces and a newline) and drops blank lines; this turns them back
    into plain newlines."""
    return page_content.replace("  \n", "\n").strip()


def _pieces(text: str) -> list[str]:
    """The section as it is, or cut by size if it is too long."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    return _by_size.split_text(text)


def bank_chunks(markdown: str) -> list[Chunk]:
    """The bank's terms: one chunk per topic and per FAQ question."""
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "document"), ("##", "topic"), ("###", "question")]
    ).split_text(markdown)
    chunks = []
    for section in sections:
        meta = section.metadata
        title = meta.get("question") or meta.get("topic") or meta["document"]
        for piece in _pieces(_section_text(section.page_content)):
            chunks.append(
                Chunk(
                    chunk_id=f"bank-{len(chunks) + 1:02d}",
                    document="bank",
                    title=title,
                    text=piece,
                    source=BANK_SOURCE,
                )
            )
    return chunks


def cfpb_chunks(markdown: str) -> list[Chunk]:
    """The CFPB answers: one chunk per answer, long answers cut by size. The
    document's introduction (before the first answer) is not a chunk."""
    answers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "question")], strip_headers=True
    ).split_text(markdown)
    chunks = []
    for answer in answers:
        question = answer.metadata.get("question")
        if question is None:
            continue
        content = _section_text(answer.page_content)
        match = _SOURCE_LINE.search(content)
        address, number = match.group(1), match.group(2)
        text = _SOURCE_LINE.sub("", content).strip()
        for i, piece in enumerate(_pieces(text), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"cfpb-{number}-{i}",
                    document="cfpb",
                    title=question,
                    text=piece.strip(),
                    source=f"{address} (Ask CFPB answer {number})",
                )
            )
    return chunks


def all_chunks(
    bank_path: Path = bank_terms.PATH, cfpb_path: Path = cfpb.PATH
) -> list[Chunk]:
    """Every chunk of both documents: the bank's first."""
    return bank_chunks(bank_path.read_text()) + cfpb_chunks(cfpb_path.read_text())
