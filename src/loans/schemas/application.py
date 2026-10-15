"""The loan application, in two versions.

- RawApplication: what the customer sent. Every value may be missing or wrong,
  so any input from the web form is accepted without crashing.
- CleanApplication: what the application must look like before it moves on.
  Values have strict types, and error rules are enforced here.

Warnings (for example "applicant appears to be under 18") are not part of these
classes: the format is valid, so they are produced by check_fields instead.

The error messages written in this file never contain the customer's value,
because they are returned to the LLM and some values are sensitive (SSN, date
of birth). Pydantic's errors do carry the original value (printing a
ValidationError shows it as input_value), and library messages such as the
email check are not under our control. check_fields must therefore never pass
a raw error, its value or a library message to the LLM.
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Raw application: what the customer sent
# ---------------------------------------------------------------------------

# A single value from the web form. Depending on the form element it may
# arrive as text, a number or a checkbox value, or not at all.
RawValue = str | int | float | bool | None


class RawFile(BaseModel):
    """A file in an upload slot. Set by the upload system, not typed by the customer."""

    filename: RawValue = None
    content_type: RawValue = None


class RawAddress(BaseModel):
    street: RawValue = None
    city: RawValue = None
    state: RawValue = None
    zip: RawValue = None


class RawIdDocument(BaseModel):
    type: RawValue = None
    front: RawFile | None = None
    back: RawFile | None = None


class RawIncomeDocuments(BaseModel):
    """The file in each income document upload slot. An empty slot is None."""

    pay_stub: RawFile | None = None
    w2: RawFile | None = None
    tax_return: RawFile | None = None


class RawApplication(BaseModel):
    """What the customer sent, exactly as received."""

    model_config = ConfigDict(extra="ignore")

    full_name: RawValue = None
    date_of_birth: RawValue = None
    ssn: RawValue = None
    address: RawAddress = Field(default_factory=RawAddress)
    phone: RawValue = None
    email: RawValue = None
    id_document: RawIdDocument = Field(default_factory=RawIdDocument)

    employment_status: RawValue = None
    employer_name: RawValue = None
    annual_income: RawValue = None
    income_documents: RawIncomeDocuments = Field(default_factory=RawIncomeDocuments)

    monthly_housing_payment: RawValue = None
    loan_amount: RawValue = None
    term_months: RawValue = None

    credit_check_consent: RawValue = None
    esign_consent: RawValue = None


class SubmissionMetadata(BaseModel):
    """Values set by the bank's system, never by the customer or the LLM."""

    application_id: str
    submitted_at: datetime
    credit_check_consented_at: datetime | None = None
    esign_consented_at: datetime | None = None


# ---------------------------------------------------------------------------
# Small helpers: formatting cleanup before the strict checks
# ---------------------------------------------------------------------------


def _empty_to_none(value):
    """Treat an empty or whitespace-only text as missing."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _strip_money_formatting(value):
    """Remove "$", commas and spaces, so "$5,000" is read as 5000."""
    value = _empty_to_none(value)
    if isinstance(value, str):
        return value.replace("$", "").replace(",", "").replace(" ", "")
    return value


NonEmptyText = Annotated[
    str,
    BeforeValidator(_empty_to_none),
    StringConstraints(strip_whitespace=True, min_length=1),
]
Money = Annotated[Decimal, BeforeValidator(_strip_money_formatting)]

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY", "PR",
}  # fmt: skip

PO_BOX = re.compile(r"\b(p\.?\s*o\.?\s*box|post\s+office\s+box)\b", re.IGNORECASE)

ALLOWED_TERMS = {12, 24, 36, 48, 60}
MIN_LOAN_AMOUNT = Decimal(1_000)
MAX_LOAN_AMOUNT = Decimal(50_000)

# ---------------------------------------------------------------------------
# Clean application: what the application must look like before it moves on
# ---------------------------------------------------------------------------


class UploadedFile(BaseModel):
    filename: NonEmptyText
    content_type: Literal["image/jpeg", "image/png", "application/pdf"]


class Address(BaseModel):
    street: NonEmptyText
    city: NonEmptyText
    state: NonEmptyText
    zip: NonEmptyText

    @field_validator("street")
    @classmethod
    def not_a_po_box(cls, value: str) -> str:
        # The Customer Identification Program requires a street address.
        if PO_BOX.search(value):
            raise ValueError("must be a street address, not a P.O. box")
        return value

    @field_validator("state")
    @classmethod
    def valid_state(cls, value: str) -> str:
        value = value.upper()
        if value not in US_STATES:
            raise ValueError("not a valid US state")
        return value

    @field_validator("zip")
    @classmethod
    def valid_zip(cls, value: str) -> str:
        if not re.fullmatch(r"\d{5}(-\d{4})?", value):
            raise ValueError("not a valid ZIP code")
        return value


class IdDocument(BaseModel):
    type: Literal["drivers_license", "state_id", "passport"]
    front: UploadedFile
    # Required for cards; checked below. validate_default makes the check run
    # even when no back side was uploaded at all.
    back: UploadedFile | None = Field(default=None, validate_default=True)

    @field_validator("back")
    @classmethod
    def back_required_for_cards(
        cls, value: UploadedFile | None, info: ValidationInfo
    ) -> UploadedFile | None:
        is_card = info.data.get("type") in ("drivers_license", "state_id")
        if is_card and value is None:
            raise ValueError("missing")
        return value


class IncomeDocuments(BaseModel):
    """One upload slot per document. Which slots must be filled depends on the
    employment status; that rule is in CleanApplication."""

    pay_stub: UploadedFile | None = None
    w2: UploadedFile | None = None
    tax_return: UploadedFile | None = None


class CleanApplication(BaseModel):
    """The application after all error rules pass.

    Only a CleanApplication can be marked complete (the validation gate).
    """

    # Set by the system.
    application_id: str
    submitted_at: datetime
    credit_check_consented_at: datetime
    esign_consented_at: datetime | None = None

    # Applicant.
    full_name: NonEmptyText
    date_of_birth: Annotated[date, BeforeValidator(_empty_to_none)]
    ssn: Annotated[str, BeforeValidator(_empty_to_none)]
    address: Address
    phone: Annotated[str, BeforeValidator(_empty_to_none)]
    email: Annotated[EmailStr, BeforeValidator(_empty_to_none)]
    id_document: IdDocument

    # Employment and income. employment_status must come before employer_name,
    # because the employer check reads it.
    employment_status: Literal["employed", "self_employed", "other"]
    employer_name: str | None = Field(default=None, validate_default=True)
    annual_income: Annotated[Money, Field(gt=0)]
    income_documents: IncomeDocuments

    # Obligations and loan request.
    monthly_housing_payment: Annotated[Money, Field(ge=0)]
    loan_amount: Money
    term_months: int

    # Consents (the customer's answers; the timestamps are above).
    credit_check_consent: bool
    esign_consent: bool

    @field_validator("date_of_birth")
    @classmethod
    def not_in_future(cls, value: date, info: ValidationInfo) -> date:
        # Evals pass a fixed "today" through the validation context.
        today = (info.context or {}).get("today") or date.today()
        if value > today:
            raise ValueError("in the future")
        return value

    @field_validator("ssn")
    @classmethod
    def valid_ssn(cls, value: str) -> str:
        digits = value.replace("-", "").replace(" ", "")
        if not digits.isdigit() or len(digits) != 9:
            raise ValueError("must have exactly 9 digits")
        area, group, serial = digits[:3], digits[3:5], digits[5:]
        if (
            area in ("000", "666")
            or area.startswith("9")
            or group == "00"
            or serial == "0000"
        ):
            raise ValueError("not a valid SSN")
        return digits

    @field_validator("phone")
    @classmethod
    def valid_phone(cls, value: str) -> str:
        digits = re.sub(r"[\s\-().]", "", value).removeprefix("+")
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if not digits.isdigit() or len(digits) != 10:
            raise ValueError("must have 10 digits")
        if digits[0] in "01":
            raise ValueError("not a valid US phone number")
        return digits

    @field_validator("employer_name")
    @classmethod
    def employer_required_if_employed(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        value = _empty_to_none(value)
        if info.data.get("employment_status") == "employed" and value is None:
            raise ValueError("missing")
        return value.strip() if value else None

    @field_validator("loan_amount")
    @classmethod
    def loan_amount_in_range(cls, value: Decimal) -> Decimal:
        if not MIN_LOAN_AMOUNT <= value <= MAX_LOAN_AMOUNT:
            raise ValueError("must be between $1,000 and $50,000")
        return value

    @field_validator("term_months")
    @classmethod
    def allowed_term(cls, value: int) -> int:
        if value not in ALLOWED_TERMS:
            raise ValueError("must be 12, 24, 36, 48 or 60 months")
        return value

    @field_validator("credit_check_consent")
    @classmethod
    def credit_check_consent_given(cls, value: bool) -> bool:
        if not value:
            raise ValueError("required to process the application")
        return value

    @model_validator(mode="after")
    def required_income_documents(self) -> "CleanApplication":
        # Needs both employment_status and income_documents, so it runs only
        # after every single field is valid.
        docs = self.income_documents
        if self.employment_status == "employed" and not (docs.pay_stub or docs.w2):
            raise ValueError("income_documents: a pay stub or W-2 is required")
        if self.employment_status == "self_employed" and not docs.tax_return:
            raise ValueError("income_documents: a tax return is required")
        return self

    @model_validator(mode="after")
    def consent_timestamps_present(self) -> "CleanApplication":
        # The system sets these when a box is checked. A missing one is a
        # system error, not a customer mistake.
        if self.esign_consent and self.esign_consented_at is None:
            raise ValueError("esign_consented_at: missing")
        return self
