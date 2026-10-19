"""check_application: the agent's tool for checking the application form.

The agent asks "is this application filled in correctly?" and gets back the
safe text result of check_fields: one line per field, ok, error or warning.

The store, application_id and today are injected by the system and are not
part of the tool's parameters that the LLM sees; the tool has no parameters
the LLM chooses. Text in a prompt, including an ID injected by an attacker,
therefore cannot change which application is checked: the tool only ever works
on the current application (protection against cross-tenant access). The LLM
is not given the application ID at all; it has no use for it. The date is
supplied by the system because time is simulated in the evals.

The raw application never leaves the tool; errors on system fields are left
out of the text (see check_fields).
"""

from datetime import date

from loans.checks.fields import check_fields
from loans.store import ApplicationStore


def check_application(store: ApplicationStore, application_id: str, today: date) -> str:
    """Check the current application and return the result as text for the agent."""
    stored = store.get(application_id)
    return check_fields(stored.raw, stored.metadata, today).to_llm_text()
