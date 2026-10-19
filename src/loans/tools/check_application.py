"""check_application: the agent's tool for checking the application form.

The agent asks "is this application filled in correctly?" and gets back the
safe text result of check_fields: one line per field, ok, error or warning.

The store, application_id and now are injected by the system and are not
part of the tool's parameters that the LLM sees; the tool has no parameters
the LLM chooses. Text in a prompt, including an ID injected by an attacker,
therefore cannot change which application is checked: the tool only ever works
on the current application (protection against cross-tenant access). The LLM
is not given the application ID at all; it has no use for it. The time is
supplied by the system because time is simulated in the evals. Each check is
recorded in the audit log (the counts of errors and warnings, not the values).

The raw application never leaves the tool; errors on system fields are left
out of the text (see check_fields).
"""

from datetime import datetime

from loans.checks.fields import check_fields
from loans.store import ApplicationStore


def check_application(store: ApplicationStore, application_id: str, now: datetime) -> str:
    """Check the current application and return the result as text for the agent."""
    stored = store.get(application_id)
    text = check_fields(stored.raw, stored.metadata, now.date()).to_llm_text()
    summary = text.splitlines()[0]  # for example "complete: no (1 error, 0 warnings)"
    store.log_event(application_id, now, "agent", "form_checked", summary)
    return text
