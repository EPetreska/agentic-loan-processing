"""The format of one eval case, and loading it from its folder.

A case is a folder:

    evals/cases/<case_id>/
      case.json         what the case is about, and the correct result
      application.json  the form, as RawApplication reads it
      documents/        one UploadedDocument per upload, named after its slot
                        (pay_stub.json, id_front.json, ...), plus documents the
                        customer uploads later (for example pay_stub_new.json)
      customer.json     how the scripted customer reacts, during the
                        application and, if the case goes on, after the offer
      officer.json      (optional) how the scripted loan officer decides the
                        offer agent's requests, in order

Case files are written by hand, so they are checked strictly: an unknown or
misspelled key is an error, not silently ignored. A typo such as
"max_messages_to_custmer" would otherwise quietly weaken the test.
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from loans.schemas.application import RawApplication, RawValue
from loans.schemas.documents import UploadedDocument, UploadSlot
from loans.schemas.outcome import Outcome
from loans.store import ApplicationStore, filled_slots
from loans.workflow import receive_application

# What a case mainly tests (one kind per case):
#   common      a normal, everyday application
#   rule        one specific rule (an expired ID, a missing pay stub)
#   reasoning   real thinking across several pieces of information
#   disruption  an uncooperative or unpredictable customer over time
#   attack      someone tries to trick the agent
#   edge        a borderline value (turning 18 tomorrow)
CaseKind = Literal["common", "rule", "reasoning", "disruption", "attack", "edge"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expected(Strict, Outcome):
    """The correct result. Only outcome and reason are required; the other
    checks are used when they matter for the case."""

    # Whether the agent should end with the income verified (None: not checked).
    income_verified: bool | None = None
    # Documents the agent must ask the customer for.
    must_request: list[UploadSlot] = Field(default_factory=list)
    # The most messages the agent may send the customer (None: no limit).
    max_messages_to_customer: int | None = None
    # How the offer phase should close, for a case that goes on after the
    # application (None: not checked).
    offer_outcome: Literal["accepted", "expired"] | None = None


class CaseInfo(Strict):
    """case.json"""

    title: str
    kind: CaseKind
    description: str = ""
    # The moment the application is submitted. Fixed, so results repeat.
    start_time: datetime
    # Where the case starts: with the application (the application agent
    # runs), or at the offer phase (the application is set up as completed,
    # without the application agent, so only the offer agent is evaluated).
    starts_at: Literal["application", "offer"] = "application"
    expected: Expected

    @field_validator("start_time")
    @classmethod
    def has_time_zone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("start_time must include a time zone, for example Z")
        return value


class Reply(Strict):
    """The customer's reaction to one message from the agent."""

    # How long the customer takes to react.
    after_days: float = Field(default=1, ge=0)
    # A text reply, if any.
    message: str | None = None
    # New uploads: slot -> file name in documents/.
    uploads: dict[UploadSlot, str] = Field(default_factory=dict)
    # Corrections to form fields, as the customer types them.
    changes: dict[str, RawValue | dict] = Field(default_factory=dict)

    @field_validator("changes")
    @classmethod
    def known_fields(cls, value: dict) -> dict:
        unknown = set(value) - set(RawApplication.model_fields)
        if unknown:
            raise ValueError(f"unknown form fields: {sorted(unknown)}")
        return value


class OfferMessage(Strict):
    """A message from the customer after the offer was sent."""

    # Days after the offer was sent (for the first message) or after the
    # agent's last message to the customer.
    after_days: float = Field(default=1, ge=0)
    message: str


class OfferPhaseScript(Strict):
    """The customer's part after the offer was sent: messages to the bank,
    one at a time, then the signature, or none."""

    messages: list[OfferMessage] = Field(default_factory=list)
    # Days after the agent's last message (or the offer, if the customer
    # never wrote) when the customer signs the latest offer. None: the
    # customer never signs, and the offer expires.
    signs_after_days: float | None = Field(default=None, ge=0)


class CustomerScript(Strict):
    """customer.json. When the replies run out, the customer stops answering.
    A case that goes on after the application has an offer phase."""

    # A message sent together with the application.
    first_message: str | None = None
    replies: list[Reply] = Field(default_factory=list)
    offer_phase: OfferPhaseScript | None = None


class OfficerDecision(Strict):
    """The scripted loan officer's decision on one request."""

    # Days after the offer agent made the request.
    after_days: float = Field(default=1, ge=0)
    decision: Literal["approve", "decline"]
    # For an approval: the amount and term of the revised offer.
    amount: Decimal | None = None
    term_months: int | None = None
    note: str = ""

    @model_validator(mode="after")
    def approval_has_amount_and_term(self) -> "OfficerDecision":
        if self.decision == "approve" and (self.amount is None or self.term_months is None):
            raise ValueError("an approval needs an amount and a term_months")
        return self


class OfficerScript(Strict):
    """officer.json: the loan officer's decisions, one per request, in order.
    When they run out, a request stays waiting."""

    officer_id: str
    decisions: list[OfficerDecision] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One loaded case, ready for the runner."""

    case_id: str
    info: CaseInfo
    application: RawApplication
    documents: dict[str, UploadedDocument]  # file name -> document
    customer: CustomerScript
    officer: OfficerScript | None = None


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_case(folder: Path) -> EvalCase:
    """Load a case folder and check that its files fit together."""
    folder = Path(folder)
    documents = {
        path.name: UploadedDocument.model_validate(_read_json(path))
        for path in sorted((folder / "documents").glob("*.json"))
    }
    case = EvalCase(
        case_id=folder.name,
        info=CaseInfo.model_validate(_read_json(folder / "case.json")),
        application=RawApplication.model_validate(
            _read_json(folder / "application.json")
        ),
        documents=documents,
        customer=CustomerScript.model_validate(_read_json(folder / "customer.json")),
        officer=(
            OfficerScript.model_validate(_read_json(folder / "officer.json"))
            if (folder / "officer.json").exists()
            else None
        ),
    )

    # Every upload in the form has its document, named after the slot.
    for slot in filled_slots(case.application):
        if f"{slot}.json" not in documents:
            raise ValueError(f"{case.case_id}: missing documents/{slot}.json")
    # Every later upload by the customer points to an existing document.
    for reply in case.customer.replies:
        for slot, name in reply.uploads.items():
            if name not in documents:
                raise ValueError(f"{case.case_id}: reply uploads missing documents/{name}")

    return case


def submit_case(
    store: ApplicationStore, case: EvalCase, application_id: str | None = None
) -> str:
    """Submit a case's application through the bank's intake
    (loans.workflow.receive_application), at the case's fixed start time.
    The application ID is the case ID unless another is given (run.py gives
    each run's case its own ID, since cases stay in the database). The
    customer's first message is submitted with it."""
    documents = {
        slot: case.documents[f"{slot}.json"] for slot in filled_slots(case.application)
    }
    return receive_application(
        store,
        case.application,
        documents,
        case.info.start_time,
        application_id=application_id or case.case_id,
        first_message=case.customer.first_message,
    )
