"""check_fields: check the raw application and report each field to the agent.

This is plain checking logic, not an agent tool. The agent reaches it only
through the tool check_application, which takes the application ID, loads the
application from the store and calls check_fields.

For each field the result is ok, error or warning, in the exact formats from
the README. The checks are
plain code: errors come from CleanApplication, warnings are added here.

What reaches the LLM is limited on purpose, so private information does not
leak. Everything in the result is put into the prompt, which is sent in an API
call to the model provider (when an external model is used) and stored in the
traces. A model that sees private data could also repeat it, for example to
an attacker who tricks it with a prompt injection. So:
- Pydantic's raw errors are never passed on. They carry the customer's value,
  and library messages (the email check) are not under our control. Each
  error is translated into a fixed message instead.
- The customer's typed value is added only for fields the LLM may see (for
  example `annual_income: not a number (customer entered "78k")`), never for
  sensitive ones (SSN, date of birth, address, phone, email, ID document).
- Free text typed by the customer (in errors, and the name and employer in ok
  results) is always quoted, kept on one line and cut, so it cannot pretend
  to be another line of the result.
- Errors on system fields (application ID, times) are kept out of the LLM's
  view: the customer cannot fix them.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from loans.schemas.application import (
    CleanApplication,
    RawApplication,
    SubmissionMetadata,
)
from loans.spotlight import quote

# Fields whose value the LLM may see, so an error on them can show what the
# customer typed. Everything else is sensitive and gets only the result.
LLM_VISIBLE_FIELDS = {
    "full_name",
    "employment_status",
    "employer_name",
    "annual_income",
    "monthly_housing_payment",
    "loan_amount",
    "term_months",
    "credit_check_consent",
    "esign_consent",
}

# Set by the system; errors here are system problems, not customer mistakes.
SYSTEM_FIELDS = {
    "application_id",
    "submitted_at",
    "credit_check_consented_at",
    "esign_consented_at",
}


# Results are listed in the order of the form.
FIELD_ORDER = [
    "full_name", "date_of_birth", "ssn", "address", "phone", "email",
    "id_document", "employment_status", "employer_name", "annual_income",
    "income_documents", "monthly_housing_payment", "loan_amount",
    "term_months", "credit_check_consent", "esign_consent",
]  # fmt: skip

# Field-specific messages for values that cannot be read at all. Our own
# rules in application.py (for example "required to process the application")
# keep their message.
FIELD_MESSAGES = {
    "email": "not a valid email address",
    "term_months": "must be 12, 24, 36, 48 or 60 months",
    "employment_status": "must be employed, self_employed or other",
    "id_document.type": "not a supported ID type",
    "credit_check_consent": "must be yes or no",
    "esign_consent": "must be yes or no",
}

# Messages by Pydantic error type, for errors not covered above.
TYPE_MESSAGES = {
    "decimal_parsing": "not a number",
    "decimal_type": "not a number",
    "int_parsing": "not a whole number",
    "date_from_datetime_parsing": "not a valid date",
    "date_parsing": "not a valid date",
    "date_type": "not a valid date",
    "greater_than": "must be more than 0",
    "greater_than_equal": "cannot be negative",
    "bool_parsing": "must be yes or no",
}


# ---------------------------------------------------------------------------
# The result returned to the agent
# ---------------------------------------------------------------------------


class FieldResult(BaseModel):
    field: str
    status: Literal["ok", "error", "warning"]
    message: str


class FieldCheckResult(BaseModel):
    """The result of check_fields. A typed model with fixed fields, so no extra data can be attached."""

    # True only if CleanApplication passes, so the application can move on.
    complete: bool
    results: list[FieldResult]
    # Problems with system fields. Kept for logging, never shown to the LLM.
    system_errors: list[str] = Field(default_factory=list, exclude=True)

    def to_llm_text(self) -> str:
        """The text the agent sees, one line per field."""
        errors = sum(r.status == "error" for r in self.results)
        warnings = sum(r.status == "warning" for r in self.results)
        header = (
            f"form complete: {'yes' if self.complete else 'no'} "
            f"({_count(errors, 'error')}, {_count(warnings, 'warning')})"
        )
        lines = [f"{r.status}: {r.field}: {r.message}" for r in self.results]
        return "\n".join([header, *lines])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _order(result: FieldResult) -> int:
    top = result.field.split(".")[0]
    return FIELD_ORDER.index(top) if top in FIELD_ORDER else len(FIELD_ORDER)


def _path(loc: tuple) -> str:
    return ".".join(str(part) for part in loc)


def _raw_value(raw: dict, loc: tuple):
    """The value the customer sent at this location, or None."""
    value = raw
    for part in loc:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())




def _parse(field: str, value):
    """Parse one raw value with the same type as in CleanApplication.

    Used for the warnings, which need the clean value of a single valid field
    even when other fields have errors. Returns None if it cannot be parsed.
    """
    annotation = CleanApplication.model_fields[field].rebuild_annotation()
    try:
        return TypeAdapter(annotation).validate_python(value)
    except ValidationError:
        return None


def _age(born: date, today: date) -> int:
    before_birthday = (today.month, today.day) < (born.month, born.day)
    return today.year - born.year - before_birthday


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _error_message(error: dict, path: str, raw: dict) -> tuple[str, str]:
    """Translate one Pydantic error into (field, fixed message)."""
    loc = error["loc"]

    # Rules on the whole application raise "field: message" themselves.
    if not loc:
        field, _, message = str(error["ctx"]["error"]).partition(": ")
        return field, message

    # A file slot: report the slot, not its internal parts.
    if loc[-1] in ("filename", "content_type"):
        slot = _path(loc[:-1])
        if _is_missing(_raw_value(raw, loc)):
            return slot, "missing"
        if loc[-1] == "content_type":
            return slot, "not an image or PDF"
        return slot, "missing"

    if _is_missing(_raw_value(raw, loc)):
        return path, "missing"
    if error["type"] == "value_error" and path != "email":
        # Our own validators in application.py, written without the value.
        # (The email check is a library message, so it is replaced below.)
        return path, str(error["ctx"]["error"])
    if path in FIELD_MESSAGES:
        return path, FIELD_MESSAGES[path]
    return path, TYPE_MESSAGES.get(error["type"], "not valid")


def _errors(
    exc: ValidationError, raw: dict
) -> tuple[list[FieldResult], list[str]]:
    results: list[FieldResult] = []
    system_errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    for error in exc.errors():
        path = _path(error["loc"])
        field, message = _error_message(error, path, raw)

        if field in SYSTEM_FIELDS:
            system_errors.append(f"{field}: {message}")
            continue
        if (field, message) in seen:
            continue
        seen.add((field, message))

        top = field.split(".")[0]
        typed = _raw_value(raw, error["loc"])
        if top in LLM_VISIBLE_FIELDS and not _is_missing(typed):
            message = f"{message} (customer entered {quote(typed)})"
        results.append(FieldResult(field=field, status="error", message=message))

    return results, system_errors


# ---------------------------------------------------------------------------
# Warnings and ok results
# ---------------------------------------------------------------------------


def _warning(field: str, message: str) -> FieldResult:
    return FieldResult(field=field, status="warning", message=message)


def _ok(field: str, message: str) -> FieldResult:
    return FieldResult(field=field, status="ok", message=message)


def _valid_fields(
    raw: dict, error_fields: set[str], today: date
) -> list[FieldResult]:
    """An ok or warning result for each field that has no error."""
    results: list[FieldResult] = []

    def valid(field: str) -> bool:
        return not any(
            e == field or e.startswith(field + ".") for e in error_fields
        )

    income = _parse("annual_income", raw["annual_income"])
    income_valid = valid("annual_income") and income is not None

    if valid("full_name"):
        # Free text from the customer: quoted, like the values in errors.
        results.append(_ok("full_name", quote(str(raw["full_name"]).strip())))

    if valid("date_of_birth"):
        born = _parse("date_of_birth", raw["date_of_birth"])
        age = _age(born, today) if born else None
        if age is not None and age < 18:
            results.append(
                _warning("date_of_birth", "applicant appears to be under 18")
            )
        elif age is not None and age > 120:
            results.append(
                _warning(
                    "date_of_birth",
                    "applicant appears to be over 120, likely a typo",
                )
            )
        else:
            results.append(_ok("date_of_birth", "ok"))

    for field in ("ssn", "address", "phone", "email"):
        if valid(field):
            results.append(_ok(field, "ok"))

    # The ID and income lines say what they cover, so "ok" is not read as
    # "the ID is verified" or "the income is proven": this check sees only the
    # form; the documents themselves are checked by other tools.
    if valid("id_document"):
        id_document = _parse("id_document", raw["id_document"])
        sides = "front and back" if id_document.back else "front"
        results.append(
            _ok(
                "id_document",
                f"{id_document.type}, {sides} uploaded (the form only; the ID itself "
                "is compared with verify_identity)",
            )
        )


    if valid("employment_status"):
        results.append(
            _ok("employment_status", str(raw["employment_status"]).strip())
        )

    employer = raw["employer_name"]
    if valid("employer_name") and not _is_missing(employer):
        results.append(_ok("employer_name", quote(str(employer).strip())))

    if income_valid:
        results.append(_ok("annual_income", str(income)))

    if valid("income_documents"):
        slots = raw["income_documents"]
        uploaded = [slot for slot, file in slots.items() if file is not None]
        message = "uploaded: " + ", ".join(uploaded) if uploaded else "none uploaded"
        message += " (the form only; the documents themselves are read with read_document)"
        results.append(_ok("income_documents", message))

    # The housing payment is not proven by any document, so this warning is
    # the only chance to catch yearly rent typed as monthly.
    if valid("monthly_housing_payment"):
        housing = _parse("monthly_housing_payment", raw["monthly_housing_payment"])
        if income_valid and housing is not None and housing > income / 12 / 2:
            results.append(
                _warning(
                    "monthly_housing_payment",
                    f"very high compared to income ({housing})",
                )
            )
        else:
            results.append(_ok("monthly_housing_payment", str(housing)))

    if valid("loan_amount"):
        results.append(_ok("loan_amount", str(_parse("loan_amount", raw["loan_amount"]))))

    if valid("term_months"):
        results.append(_ok("term_months", str(_parse("term_months", raw["term_months"]))))

    if valid("credit_check_consent"):
        results.append(_ok("credit_check_consent", "given"))

    if valid("esign_consent"):
        if _parse("esign_consent", raw["esign_consent"]):
            results.append(_ok("esign_consent", "given"))
        else:
            results.append(_ok("esign_consent", "declined, paper process needed"))

    return results


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def check_fields(
    raw: RawApplication,
    metadata: SubmissionMetadata,
    today: date | None = None,
) -> FieldCheckResult:
    """Check every field of the application.

    today is passed in so evals can use a fixed date for the age checks; by
    default it is the real date.
    """
    today = today or date.today()
    raw_dict = raw.model_dump()

    error_results: list[FieldResult] = []
    system_errors: list[str] = []
    try:
        CleanApplication.model_validate(
            raw_dict | metadata.model_dump(), context={"today": today}
        )
        complete = True
    except ValidationError as exc:
        error_results, system_errors = _errors(exc, raw_dict)
        complete = False

    error_fields = {r.field for r in error_results}
    results = error_results + _valid_fields(raw_dict, error_fields, today)
    results.sort(key=_order)  # stable: keeps several errors of one field in order
    return FieldCheckResult(
        complete=complete, results=results, system_errors=system_errors
    )
