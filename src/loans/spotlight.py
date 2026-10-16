"""Spotlighting: marking untrusted text before it reaches the LLM.

Text that comes from the customer or from a document (a name, an employer, a
typed value, hidden text on a pay stub) is always put in quotes, kept on one
line and cut to a maximum length. It can then not pretend to be another line
of a tool result (for example a fake "complete: yes"). Used by every tool that
shows such text to the agent.
"""

import json

# Untrusted text shown to the LLM is cut to this length, to limit how much of
# it (for example an injection attempt) reaches the prompt. Long enough for
# real long legal names, so the agent can still compare an employer name with
# the documents.
MAX_QUOTED_LENGTH = 200


def quote(value) -> str:
    """The value quoted, cut and on one line, safe to show to the LLM."""
    text = str(value)
    if len(text) > MAX_QUOTED_LENGTH:
        text = text[:MAX_QUOTED_LENGTH] + "..."
    # json.dumps adds quotes and escapes newlines, so the value cannot
    # pretend to be another line of the result.
    return json.dumps(text)
