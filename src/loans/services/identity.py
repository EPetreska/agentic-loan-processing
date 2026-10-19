"""check_identity: the mock identity verification vendor.

It compares the ID document with the form and reports only its findings: is
the document of the declared type, readable and not expired, and do the name
and date of birth match the form. It gives no verdict (no pass, review or
fail): what to do with the findings is the bank's policy, which lives in the
frame (the agent's instructions and the checks in finish), not in the vendor.

Names are compared ignoring letter case, extra spaces and punctuation, so
"DANIEL KIM" matches "Daniel Kim". Anything else counts as a difference,
including a middle name or initial; the agent deals with it.

Document authenticity and fraud are not checked (see the notes).
"""

import re
from datetime import date

from pydantic import BaseModel, TypeAdapter, ValidationError

from loans.schemas.application import CleanApplication, RawApplication
from loans.schemas.documents import IdDocumentContent, UploadedDocument

_DATE_OF_BIRTH = TypeAdapter(
    CleanApplication.model_fields["date_of_birth"].rebuild_annotation()
)


class IdentityFindings(BaseModel):
    """The vendor's findings. None means the comparison could not be made;
    cannot_check then says why, per field."""

    document_type_matches_declared: bool
    readable: bool
    expired: bool | None = None
    name_matches_form: bool | None = None
    date_of_birth_matches_form: bool | None = None
    cannot_check: dict[str, str] = {}
    # Which upload of the ID front and which version of the form were
    # compared, and when, so finish can tell whether the findings are still
    # current.
    id_front_version: int
    form_version: int
    checked_on: date

    def to_llm_text(self) -> str:
        """The text the agent sees. No name, date of birth or ID number."""

        def show(field: str) -> str:
            value = getattr(self, field)
            if value is None:
                return f"{field}: cannot check ({self.cannot_check[field]})"
            return f"{field}: {'yes' if value else 'no'}"

        return "\n".join(
            show(field)
            for field in [
                "document_type_matches_declared",
                "readable",
                "expired",
                "name_matches_form",
                "date_of_birth_matches_form",
            ]
        )


def normalize_name(name: str) -> str:
    """Lowercase, punctuation removed (hyphens become spaces), single spaces."""
    name = name.lower().replace("-", " ")
    name = re.sub(r"[^\w\s]", "", name)
    return " ".join(name.split())


def _form_date_of_birth(raw: RawApplication) -> date | None:
    try:
        return _DATE_OF_BIRTH.validate_python(raw.date_of_birth)
    except ValidationError:
        return None


def check_identity(
    document: UploadedDocument,
    raw: RawApplication,
    id_front_version: int,
    form_version: int,
    today: date,
) -> IdentityFindings:
    """Compare the ID document with the form, as an identity vendor would."""
    findings = IdentityFindings(
        document_type_matches_declared=document.actual_type == raw.id_document.type,
        readable=document.readable,
        id_front_version=id_front_version,
        form_version=form_version,
        checked_on=today,
    )
    content = document.parsed()
    if not isinstance(content, IdDocumentContent):
        reason = (
            "the document could not be read" if not document.readable
            else "the document is not an ID"
        )
        findings.cannot_check = dict.fromkeys(
            ["expired", "name_matches_form", "date_of_birth_matches_form"], reason
        )
        return findings

    findings.expired = content.expiry_date < today

    form_name = raw.full_name
    if form_name is None or not str(form_name).strip():
        findings.cannot_check["name_matches_form"] = "the name on the form is missing"
    else:
        findings.name_matches_form = normalize_name(str(form_name)) == normalize_name(
            content.full_name
        )

    form_dob = _form_date_of_birth(raw)
    if form_dob is None:
        findings.cannot_check["date_of_birth_matches_form"] = (
            "the date of birth on the form is missing or not valid"
        )
    else:
        findings.date_of_birth_matches_form = form_dob == content.date_of_birth

    return findings
