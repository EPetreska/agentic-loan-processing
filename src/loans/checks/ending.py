"""ending_problems: the bank's rules for when a case may end the way the
agent chose.

The agent decides how a case ends; these rules decide whether that ending is
allowed (defense in depth: the LLM decides what to do, the code decides
whether the outcome is allowed). They return every problem, so the agent can
fix them or choose another ending. Used by the finish tool.

- complete: the form has no errors, the applicant is 18 or older, e-sign
  consent was given, the identity findings on file are current and all as
  required (ID of the declared type, readable, not expired, name and date of
  birth match the form), and an income verification on file is current. The
  recorded findings and verification are checked; they are not redone.
- cancelled because the customer did not answer or did not send the
  documents (no_answer, documents_not_provided): only after at least 26 days
  since the last contact, never earlier (policy.py). The last contact is the
  agent's last message or the customer's last reply, whichever is later, so a
  customer who answered gets the full 26 days again.
- cancelled because the customer withdrew or refused the credit check: the
  agent's judgment from the customer's messages, stated in its explanation.
- sent_to_human: always allowed. Handing a case to a person is the safe
  direction.
"""

from datetime import date, datetime, timedelta

from pydantic import TypeAdapter, ValidationError

from loans import policy
from loans.checks.fields import check_fields
from loans.schemas.application import CleanApplication
from loans.schemas.outcome import Outcome
from loans.store import StoredApplication

# The reasons that cancel a case for waiting too long on the customer.
WAITING_REASONS = {"no_answer", "documents_not_provided"}

_EMPLOYMENT_STATUS = TypeAdapter(
    CleanApplication.model_fields["employment_status"].rebuild_annotation()
)


def _employment_status(stored: StoredApplication) -> str | None:
    try:
        return _EMPLOYMENT_STATUS.validate_python(stored.raw.employment_status)
    except ValidationError:
        return None  # reported as a form error


_DATE_OF_BIRTH = TypeAdapter(
    CleanApplication.model_fields["date_of_birth"].rebuild_annotation()
)


def _is_under_18(stored: StoredApplication, today: date) -> bool:
    try:
        born = _DATE_OF_BIRTH.validate_python(stored.raw.date_of_birth)
    except ValidationError:
        return False  # reported as a form error
    before_birthday = (today.month, today.day) < (born.month, born.day)
    return today.year - born.year - before_birthday < 18


def _identity_problems(stored: StoredApplication) -> list[str]:
    if not stored.identity_findings_are_current():
        return ["the identity findings are missing or out of date"]
    findings = stored.identity_findings
    required = {
        "document_type_matches_declared": True,
        "readable": True,
        "expired": False,
        "name_matches_form": True,
        "date_of_birth_matches_form": True,
    }
    return [
        f"identity: {field} is not as required"
        for field, value in required.items()
        if getattr(findings, field) is not value
    ]


def _complete_problems(stored: StoredApplication, today: date) -> list[str]:
    problems = []
    if not check_fields(stored.raw, stored.metadata, today).complete:
        problems.append("the form still has errors")
    if _is_under_18(stored, today):
        problems.append("the applicant is under 18")
    if stored.metadata.esign_consented_at is None:
        problems.append("the customer has not given e-sign consent")
    problems += _identity_problems(stored)
    if stored.income_verification is None:
        if _employment_status(stored) == "other":
            problems.append(
                'the income cannot be verified: no document counts for the employment status "other"'
            )
        else:
            problems.append("the income has not been verified")
    elif not stored.income_verification_is_current():
        problems.append(
            "the income verification is out of date (the document or the form changed)"
        )
    return problems


def last_contact(stored: StoredApplication) -> datetime | None:
    """The time of the agent's last message or the customer's last reply,
    whichever is later; None if neither happened."""
    times = [m.sent_at for m in stored.messages] + [r.received_at for r in stored.replies]
    return max(times, default=None)


def _waiting_problems(stored: StoredApplication, now: datetime) -> list[str]:
    if not stored.messages:
        return ["no message has been sent to the customer"]
    waited = now - last_contact(stored)
    if waited < timedelta(days=policy.NO_ANSWER_CANCEL_DAYS):
        return [
            f"too early: {waited.days} days since the last message or reply "
            f"(the minimum is {policy.NO_ANSWER_CANCEL_DAYS} days)"
        ]
    return []


def ending_problems(stored: StoredApplication, ending: Outcome, now: datetime) -> list[str]:
    """Every reason the case may not end this way. Empty if it may."""
    if ending.outcome == "complete":
        return _complete_problems(stored, now.date())
    if ending.reason in WAITING_REASONS:
        return _waiting_problems(stored, now)
    return []
