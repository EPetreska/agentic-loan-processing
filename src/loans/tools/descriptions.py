"""What the LLM reads about each tool and its parameters.

Shared by both ways the agent can reach the tools: directly
(loans.agent.tools) and through the MCP server (loans.mcp.server), so the two
cannot drift apart. The plain tools in this package keep their own
docstrings, for developers. Numbers come from the code (here the message
limit), so a description cannot drift from the real rule.
"""

from loans.tools.send_message import MAX_MESSAGE_LENGTH

CHECK_APPLICATION = (
    "Check that the application form is filled in correctly. Returns one line "
    "per field: ok, error or warning. It checks the form only: it does not "
    "compare the ID with the form or read the income documents. For fields such "
    "as the name, employer, income and loan, it also shows the customer's value: "
    "the checked value when ok (free text in quotes), and exactly what was typed, "
    "in quotes, when there is an error. Sensitive fields such as the SSN and the "
    "date of birth show only the result."
)

READ_DOCUMENT = (
    'Read an uploaded income document. Returns two parts. First, "extraction '
    'findings:", what the extraction service read, one "name: value" line each '
    "for: what the file really is (document_type) and whether it matches the "
    "slot it was uploaded in (matches_slot); whether it could be read "
    "(readable); the fields on the document, with text from the document in "
    'quotes; values calculated in code, marked "(calculated)", such as the '
    "yearly income and the pay stub's age; and any other text found on the "
    "document, in quotes. Then the bank's own check: \"bank income rules: met\", "
    'or "not met" with one line per rule that is not met.'
)

VERIFY_IDENTITY = (
    "Compare the uploaded ID with the form. Returns one line per finding: yes, "
    'no, or "cannot check" with the reason. The findings are facts, not a '
    'decision. A "no" never says how the values differ, because personal data '
    "is never shown. After a new ID upload or a form correction, verify again: "
    "earlier findings are then out of date."
)

MARK_INCOME_VERIFIED = (
    "Record that the document in this slot verifies the claimed income. Refused "
    "if a bank income rule is not met. After a new upload of this document or a "
    "form correction, mark it again: the earlier verification is then out of "
    "date."
)

SEND_MESSAGE = (
    "Send a message to the customer. Refused if the customer has not given "
    "e-sign consent, if the text is empty or longer than "
    f"{MAX_MESSAGE_LENGTH} characters, or without a reason."
)

FINISH = (
    "End the case with an outcome, a reason that fits it, and an explanation. "
    "Refused if the bank's rules do not allow this ending, listing every "
    "problem. A case ends only once."
)

# Parameters
INCOME_REASON = "Why this document verifies the claimed income."
MESSAGE_TEXT = "The message to the customer."
MESSAGE_REASON = "Why you are sending it, in a short sentence."
EXPLANATION = "A short explanation for whoever takes the case next."
