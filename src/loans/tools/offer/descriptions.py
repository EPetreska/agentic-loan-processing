"""What the LLM reads about each of the offer agent's tools and their
parameters.

Shared by both ways the offer agent can reach its tools: directly
(loans.agents.offer.tools) and, later, through the MCP server, so the two
cannot drift apart. They say how each tool behaves; what the agent should do
is in its prompt. Numbers come from the code, so a description cannot drift
from the real limit. The message texts are the application agent's, since
both agents write to the customer under the same rules.
"""

from loans.rag.search import TOP_K
from loans.tools.application.descriptions import MESSAGE_REASON, MESSAGE_TEXT
from loans.tools.messages import MAX_MESSAGE_LENGTH
from loans.tools.offer.request_approval import MAX_REQUEST_LENGTH
from loans.tools.offer.retrieve_documents import MAX_QUERY_LENGTH

__all__ = [
    "MESSAGE_REASON",
    "MESSAGE_TEXT",
    "QUERY",
    "REQUEST",
    "REQUEST_APPROVAL",
    "REQUEST_REASON",
    "RETRIEVE_DOCUMENTS",
    "SEND_MESSAGE",
]

RETRIEVE_DOCUMENTS = (
    "Search the bank's loan terms and FAQ and the CFPB's consumer answers. "
    f"Returns the {TOP_K} closest sections, each with its source, marked as data. "
    "Write the query in your own words, about one topic. Refused if the query "
    f"is empty or longer than {MAX_QUERY_LENGTH} characters, before the case "
    "has an offer, or after the offer phase has closed."
)

REQUEST_APPROVAL = (
    "Ask a loan officer to decide a change the customer wants to their offer. "
    "Describe what the customer wants and why a loan officer must decide. The "
    "case then waits for the officer's decision. Refused while another request "
    "is waiting, if the request or the reason is empty, if the request is longer "
    f"than {MAX_REQUEST_LENGTH} characters, before the case has an offer, or "
    "after the offer phase has closed."
)

SEND_MESSAGE = (
    "Send a message to the customer. Refused if the customer has not given "
    "e-sign consent, if the text is empty or longer than "
    f"{MAX_MESSAGE_LENGTH} characters, without a reason, before the case has "
    "an offer, or after the offer phase has closed."
)

# Parameters
QUERY = "What to look up, in a few words."
REQUEST = "What the customer wants, in your words, with any amounts."
REQUEST_REASON = "Why a loan officer must decide this."
