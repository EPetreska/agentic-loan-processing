"""Application intake: what the bank's system does when an application arrives.

Today the tests call these functions directly. Later, the FastAPI endpoint
calls the same functions; the server only adds receiving and saving.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from loans.schemas.application import RawApplication, SubmissionMetadata

_BOOL = TypeAdapter(bool)


def _is_yes(value) -> bool:
    """True only if a checkbox value clearly means yes ("yes", "true", True, ...)."""
    if value is None:
        return False
    try:
        return _BOOL.validate_python(value)
    except ValidationError:
        return False


def create_metadata(
    raw: RawApplication,
    now: datetime,
    application_id: str | None = None,
) -> SubmissionMetadata:
    """Create the values only the system sets: the application ID and the times.

    The time is passed in instead of read from the clock, so frozen evals can
    use a fixed time and test deadlines (for example the 26 days) without
    waiting. The server passes the current time.

    application_id is normally generated; evals may pass a fixed one.
    """
    if now.tzinfo is None:
        raise ValueError("now must include a time zone, for example UTC")

    return SubmissionMetadata(
        application_id=application_id or str(uuid4()),
        submitted_at=now,
        credit_check_consented_at=now if _is_yes(raw.credit_check_consent) else None,
        esign_consented_at=now if _is_yes(raw.esign_consent) else None,
    )
