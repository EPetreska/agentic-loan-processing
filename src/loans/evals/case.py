"""The format of one eval case, and loading it from its folder.

A case is a folder:

    evals/cases/<case_id>/
      case.json         what the case is about, and the correct result
      application.json  the form, as RawApplication reads it
      documents/        one UploadedDocument per upload, named after its slot
                        (pay_stub.json, id_front.json, ...), plus documents the
                        customer uploads later (for example pay_stub_new.json)
      customer.json     how the scripted customer reacts

Case files are written by hand, so they are checked strictly: an unknown or
misspelled key is an error, not silently ignored. A typo such as
"max_messages_to_custmer" would otherwise quietly weaken the test.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from loans.schemas.application import RawApplication, RawValue
from loans.schemas.documents import UploadedDocument, UploadSlot
from loans.schemas.outcome import Outcome
from loans.store import ApplicationStore, filled_slots

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


class CaseInfo(Strict):
    """case.json"""

    title: str
    kind: CaseKind
    description: str = ""
    # The moment the application is submitted. Fixed, so results repeat.
    start_time: datetime
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


class CustomerScript(Strict):
    """customer.json. When the replies run out, the customer stops answering."""

    # A message sent together with the application.
    first_message: str | None = None
    replies: list[Reply] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One loaded case, ready for the runner."""

    case_id: str
    info: CaseInfo
    application: RawApplication
    documents: dict[str, UploadedDocument]  # file name -> document
    customer: CustomerScript


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
    """Submit a case's application to the store, as at intake, at the case's
    fixed start time. The application ID is the case ID unless another is
    given (run.py gives each run's case its own ID, since cases stay in the
    database)."""
    documents = {
        slot: case.documents[f"{slot}.json"] for slot in filled_slots(case.application)
    }
    return store.submit(
        case.application, documents, case.info.start_time, application_id=application_id or case.case_id
    )
