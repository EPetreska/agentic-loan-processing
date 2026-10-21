"""The CFPB consumer guides for the offer agent's RAG.

A few general questions and answers from the Consumer Financial Protection
Bureau's "Ask CFPB" (https://www.consumerfinance.gov/ask-cfpb/), the US
consumer finance regulator's plain-language answers for consumers. They
answer "explain this to me" questions (what an installment loan is, APR,
credit inquiries and reports), while the bank's own terms answer questions
about this bank's loan (loans.rag.bank_terms). CFPB pages are works of the US
government, so they are public domain and kept word for word.

The answers are chosen for a customer with a personal installment loan offer:
the loan itself, APR, credit reports and scores, paying late, debt
consolidation and debt collection, discrimination, credit freezes and
identity theft. Some are about other products on purpose (car loans,
mortgages, student loans, payday loans), as a real bank's collection would
be, and several are close cousins of real questions (late fees on a car
loan, "can't pay my mortgage", a mortgage's APR, a mortgage's prepayment
penalty): the agent must answer a question about this loan from the bank's
terms, not from a page about another product. About 40 answers make the
search a real test: it has to find the right answer among many similar ones.

The script downloads each page, keeps only the question and the answer
(dropping menus, links and the search box) and writes one document, each
answer under its own heading with its source address and the download date.
An answer is identified by its number (the "en-733" in its address): the
same answer can be reached under several addresses.

To download the pages again:

    uv run python -m loans.rag.cfpb
"""

import re
import time
import urllib.request
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag

PATH = Path(__file__).resolve().parents[3] / "documents" / "cfpb" / "cfpb_loan_basics.md"

# The answers kept, by their Ask CFPB number.
PAGES = [
    # The loan
    2114,  # What is a personal installment loan?
    2120,  # Do personal installment loans have fees?
    733,  # What is the difference between a loan interest rate and the APR?
    45,  # What is the difference between a fixed APR and a variable APR?
    1791,  # What is a debt-to-income ratio?
    2021,  # How do automatic payments from a bank account work?
    # Credit reports and scores
    2069,  # What is the difference between a credit report and a credit score?
    1883,  # What is a FICO score?
    1317,  # What is a credit inquiry?
    1229,  # Does requesting my credit report hurt my credit score?
    323,  # How long does information stay on my credit report?
    1249,  # Is it possible to remove accurate but negative information from my credit report?
    5,  # How do I get a free copy of my credit reports?
    1251,  # What is a credit reporting company?
    1305,  # Who can request to see my credit report?
    1253,  # What can I do if my credit application was denied because of my credit report?
    1341,  # What is a credit freeze or security freeze on my credit report?
    1243,  # What is identity theft?
    # Fair lending
    1171,  # What do I do if I think a lender discriminated against me?
    1181,  # Is a lender allowed to consider my age or where my income comes from ...?
    1785,  # What is covered under the Military Lending Act?
    # Debt consolidation, debt relief and debt collection
    1861,  # What do I need to know about consolidating my credit card debt?
    1859,  # Advertisements for companies that consolidate credit card debt: legitimate?
    1449,  # Credit counseling vs debt settlement, debt consolidation or credit repair
    1457,  # What is a debt relief program and how do I know if I should use one?
    1343,  # How can I tell a credit repair scam from a reputable credit counselor?
    1387,  # What is an original creditor, and how is it different from a debt collector?
    1427,  # What may happen if I ignore or avoid a debt collector?
    1411,  # How do I get a debt collector to stop calling or contacting me?
    1423,  # Can debt collectors collect a debt that's several years old?
    # Other products, on purpose (several are close cousins of real questions)
    1957,  # What is a prepayment penalty? (mortgages)
    839,  # When are late fees charged on a car loan?
    849,  # What should I do if I can't make my car payments?
    2131,  # What should I know before I finalize a car or auto loan?
    268,  # If I can't pay my mortgage loan, what are my options?
    135,  # What is the difference between a mortgage interest rate and an APR?
    100,  # Fixed-rate vs adjustable-rate mortgage (ARM)
    104,  # What is a balloon payment? When is one allowed?
    639,  # What should I do if I can't afford my student loan payment?
    1567,  # What is a payday loan?
    1625,  # What is an APR, and why is it higher than the interest rate for my payday loan?
    1293,  # Will paying off my credit card balance every month improve my credit score?
]

ASK_CFPB = "https://www.consumerfinance.gov/ask-cfpb/"

USER_AGENT = "Mozilla/5.0 (compatible; loans-rag/1.0; downloads public CFPB answers)"

# Where the page furniture starts, after the answer.
END_OF_ANSWER = "Don't see what you're looking for?"
# A link to more pages on the site, not part of the answer.
LINK_LINE = "Learn more about"


def page_url(number: int) -> str:
    # Any words before the number reach the answer; the site redirects to
    # its real address.
    return f"{ASK_CFPB}answer-en-{number}/"


def download(number: int) -> tuple[str, str]:
    """The answer's page (HTML) and its real address."""
    # The site refuses requests that do not look like a browser's: it wants
    # the usual Accept headers and a browser-compatible program name, which
    # still says what the program is.
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    request = urllib.request.Request(page_url(number), headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8"), response.url


def _text(element: Tag) -> str:
    text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
    # A link inside a sentence leaves a space before the punctuation after it.
    return re.sub(r" ([.,;:!?)])", r"\1", text)


def clean(html: str) -> tuple[str, list[str]]:
    """The question and the answer's paragraphs, without the page furniture.
    The answer's own subheadings start with "### " and list items with "- "."""
    soup = BeautifulSoup(html, "html.parser")
    question = _text(soup.find("h1"))
    paragraphs = []
    lead = soup.select_one("div.lead-paragraph")
    if lead:
        paragraphs.append(_text(lead))
    answer = soup.select_one("div.answer-text")
    if answer:
        for element in answer.find_all(["p", "li", "h2", "h3", "h4"]):
            if element.find_parent("li") or (element.name == "p" and element.find_parent(["li"])):
                continue  # already part of a list item
            text = _text(element)
            if text.startswith(END_OF_ANSWER):
                break
            if not text or text.startswith(LINK_LINE):
                continue
            if element.name == "li":
                text = f"- {text}"
            elif element.name in ("h2", "h3", "h4"):
                text = f"### {text}"
            paragraphs.append(text)
    return question, paragraphs


def render(answers: list[tuple[int, str, str, list[str]]], downloaded: date) -> str:
    """The document: each (number, address, question, paragraphs) under its
    own heading."""
    lines = [
        "# Consumer guides from the CFPB",
        "",
        "Questions and answers from the Consumer Financial Protection Bureau's",
        f'"Ask CFPB" ({ASK_CFPB}), downloaded on {downloaded.isoformat()}.',
        "Works of the US government, in the public domain, kept word for word.",
    ]
    for number, address, question, paragraphs in answers:
        lines += ["", f"## {question}", "", f"Source: {address} (Ask CFPB answer {number})"]
        for paragraph in paragraphs:
            lines += ["", paragraph]
    return "\n".join(lines) + "\n"


def write(path: Path = PATH) -> None:
    answers = []
    for number in PAGES:
        time.sleep(1)  # one page a second, to be polite to the site
        html, address = download(number)
        question, paragraphs = clean(html)
        answers.append((number, address, question, paragraphs))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(answers, date.today()))


if __name__ == "__main__":
    write()
    print(f"written: {PATH}")
