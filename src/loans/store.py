"""The application store: every application, its metadata and its documents,
kept by application ID.

The agent never reads the store. Its tools do: each tool receives only the
application ID, looks up what it needs here, and returns a safe result
(reference passing).

ApplicationStore keeps the cases in memory, for the tests and the direct
runs. PostgresStore (loans.db.postgres_store) keeps them in Postgres, so a
separate service can share them and they survive a restart. Both follow the
same rules, written once here: only where the cases are kept differs. Every
change to a case goes through one of the methods below, inside _changing, so
the Postgres store can lock the case, save it and log the change in one
transaction.

If the store is given a run folder, every audit event is also written to the
case's log file, <run folder>/<application ID>/audit.jsonl (one JSON line per
event), the moment it happens, so a person can review the log at any time and
a crash loses nothing. The tests use no folder, so nothing is written to
disk.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from loans.audit import Actor, AgentVersion, AuditEvent, EventName
from loans.intake import create_metadata, is_yes
from loans.schemas.application import RawApplication, RawFile, SubmissionMetadata
from loans.schemas.documents import IncomeSlot, UploadedDocument, UploadSlot
from loans.schemas.outcome import OutcomeKind, Reason
from loans.offers import Offer
from loans.services.credit_bureau import CreditCheckResult
from loans.services.identity import IdentityFindings

# Each consent on the form, and the metadata field holding when it was given.
CONSENT_TIMES: dict[str, str] = {
    "credit_check_consent": "credit_check_consented_at",
    "esign_consent": "esign_consented_at",
}

# Where each upload slot sits in the form.
SLOT_PATHS: dict[str, tuple[str, str]] = {
    "id_front": ("id_document", "front"),
    "id_back": ("id_document", "back"),
    "pay_stub": ("income_documents", "pay_stub"),
    "w2": ("income_documents", "w2"),
    "tax_return": ("income_documents", "tax_return"),
}


def filled_slots(application: RawApplication) -> list[str]:
    """The upload slots that have a file in the application."""
    return [
        slot
        for slot, (group, name) in SLOT_PATHS.items()
        if getattr(getattr(application, group), name) is not None
    ]


class UnknownApplication(LookupError):
    """No application with this ID."""


class SentMessage(BaseModel):
    """A message the agent sent to the customer."""

    # The ID of the tool call that sent it, so a retried call is not sent twice.
    message_id: str
    sent_at: datetime
    text: str
    # Why the agent sent it, in its own words.
    reason: str


class CustomerReply(BaseModel):
    """A reply from the customer, kept on file for auditing."""

    received_at: datetime
    # The customer's own words, if they wrote any.
    message: str | None = None
    # The slots the customer uploaded a file into with this reply.
    uploads: list[UploadSlot] = []
    # The form fields the customer corrected with this reply (names only).
    corrected_fields: list[str] = []


class IncomeVerification(BaseModel):
    """The agent's decision that a document verifies the claimed income,
    recorded only after the bank's income rules passed."""

    # The ID of the tool call that recorded it, so a retried call is not
    # recorded twice.
    call_id: str
    slot: IncomeSlot
    # Why the agent accepts the document, in its own words (for example why
    # the employer on the pay stub matches the form). Shown to the human
    # approver.
    reason: str
    # Which upload of the document and which version of the form were
    # checked, and when, so finish can tell whether the verification is still
    # current.
    document_version: int
    form_version: int
    verified_on: date


class CaseEnding(BaseModel):
    """How the agent ended the case, recorded only if the ending was allowed."""

    # The ID of the tool call that ended the case, so a retried call returns
    # the same confirmation.
    call_id: str
    outcome: OutcomeKind
    reason: Reason
    # The agent's explanation in its own words, for the next step (the
    # credit check or the person who takes over the case).
    explanation: str
    ended_at: datetime


class StoredApplication(BaseModel):
    application_id: str
    raw: RawApplication
    metadata: SubmissionMetadata
    # The current document in each filled slot.
    documents: dict[UploadSlot, UploadedDocument]
    # How many times each slot has been uploaded (1 at submission). Lets finish
    # tell whether saved findings refer to the current upload.
    document_versions: dict[UploadSlot, int] = {}
    # How many times the form has been changed by the customer (1 at
    # submission), for the same reason.
    form_version: int = 1
    # The identity vendor's latest findings, kept on file as a bank would.
    identity_findings: IdentityFindings | None = None
    # The agent's latest income verification.
    income_verification: IncomeVerification | None = None
    # The message the customer wrote with the application, if any (sent at
    # submission, metadata.submitted_at). Kept apart from the replies: it
    # answers nothing, and the waiting rules count from the agent's messages
    # and the customer's replies.
    first_message: str | None = None
    # Messages sent to the customer, oldest first.
    messages: list[SentMessage] = []
    # Replies from the customer, oldest first.
    replies: list[CustomerReply] = []
    # Set once the agent has ended the case.
    ending: CaseEnding | None = None
    # The credit check, run by the bank's workflow after a complete ending.
    credit_check: CreditCheckResult | None = None
    # The offers created after a passed credit check, shortest term first:
    # the whole allowed space for the offer agent.
    offers: list[Offer] = []
    # Everything that happened in the case, oldest first. Only ever added to.
    audit_log: list[AuditEvent] = []

    def identity_findings_are_current(self) -> bool:
        """True if identity findings are on file and refer to the current ID
        upload and the current version of the form."""
        findings = self.identity_findings
        return (
            findings is not None
            and findings.id_front_version == self.document_versions.get("id_front")
            and findings.form_version == self.form_version
        )

    def income_verification_is_current(self) -> bool:
        """True if an income verification is on file and refers to the
        current upload of its document and the current version of the form."""
        verification = self.income_verification
        return (
            verification is not None
            and verification.document_version == self.document_versions.get(verification.slot)
            and verification.form_version == self.form_version
        )


class ApplicationStore:
    def __init__(
        self, run_dir: Path | None = None, agent: AgentVersion | None = None
    ) -> None:
        self._applications: dict[str, StoredApplication] = {}
        self._run_dir = run_dir
        # Added to agent and guardrail events.
        self._agent = agent

    @property
    def agent(self) -> AgentVersion | None:
        """The model and prompt version added to agent and guardrail events."""
        return self._agent

    def submit(
        self,
        raw: RawApplication,
        documents: dict[str, UploadedDocument],
        now: datetime,
        application_id: str | None = None,
        first_message: str | None = None,
    ) -> str:
        """Store a new application (intake) and return its ID.

        documents holds the document behind each filled upload slot;
        first_message is what the customer wrote with the application, if
        anything. The log says that a message came with it, never the text.
        """
        if set(documents) != set(filled_slots(raw)):
            raise ValueError(
                f"documents {sorted(documents)} do not match the filled "
                f"upload slots {sorted(filled_slots(raw))}"
            )
        metadata = create_metadata(raw, now, application_id)
        if self.has(metadata.application_id):
            raise ValueError(f"application {metadata.application_id} already exists")
        stored = StoredApplication(
            application_id=metadata.application_id,
            raw=raw,
            metadata=metadata,
            documents=documents,
            document_versions=dict.fromkeys(documents, 1),
            first_message=first_message,
        )
        details = "uploaded: " + (", ".join(sorted(documents)) or "nothing")
        if first_message:
            details += "; with a message"
        with self._adding(stored):
            self.log_event(
                metadata.application_id, now, "system", "application_submitted", details
            )
        return metadata.application_id

    def has(self, application_id: str) -> bool:
        return application_id in self._applications

    def get(self, application_id: str) -> StoredApplication:
        try:
            return self._applications[application_id]
        except KeyError:
            raise UnknownApplication(application_id) from None

    # Where the cases are kept. PostgresStore replaces these four methods and
    # _append_event; the rules in the other methods stay the same.

    @contextmanager
    def _adding(self, stored: StoredApplication) -> Generator[None]:
        """Add a new case, then run the body (which logs the submission)."""
        self._applications[stored.application_id] = stored
        yield

    @contextmanager
    def _changing(self, application_id: str) -> Generator[StoredApplication]:
        """The case to change. In memory it is changed in place."""
        yield self.get(application_id)

    def _append_event(self, event: AuditEvent) -> None:
        self.get(event.application_id).audit_log.append(event)

    def upload(
        self,
        application_id: str,
        slot: UploadSlot,
        document: UploadedDocument,
        file: RawFile,
        now: datetime,
    ) -> None:
        """The customer uploads a file into a slot, replacing any earlier one.

        Both the document and the form's slot are updated, so the field check
        and the tools always see the same state.
        """
        group, name = SLOT_PATHS[slot]
        with self._changing(application_id) as stored:
            form = stored.raw.model_dump()
            form[group][name] = file.model_dump()
            stored.raw = RawApplication.model_validate(form)
            stored.documents[slot] = document
            stored.document_versions[slot] = stored.document_versions.get(slot, 0) + 1
            self.log_event(application_id, now, "customer", "document_uploaded", slot)

    def record_message(self, application_id: str, message: SentMessage) -> None:
        """Keep a sent message on file."""
        with self._changing(application_id) as stored:
            stored.messages.append(message)

    def record_reply(self, application_id: str, reply: CustomerReply) -> None:
        """Keep a customer reply on file and log it. The log names only the
        uploaded slots and the corrected fields, never the message text or
        the values; the text is kept here, with the reply. The uploads and
        corrections themselves are applied with upload and correct_fields."""
        parts = []
        if reply.message:
            parts.append("message")
        if reply.uploads:
            parts.append("uploaded: " + ", ".join(reply.uploads))
        if reply.corrected_fields:
            parts.append("corrected: " + ", ".join(sorted(reply.corrected_fields)))
        with self._changing(application_id) as stored:
            stored.replies.append(reply)
            self.log_event(
                application_id,
                reply.received_at,
                "customer",
                "customer_replied",
                "; ".join(parts) or "empty reply",
            )

    def record_identity_findings(
        self, application_id: str, findings: IdentityFindings
    ) -> None:
        """Keep the identity vendor's latest findings on file."""
        with self._changing(application_id) as stored:
            stored.identity_findings = findings

    def record_income_verification(
        self, application_id: str, verification: IncomeVerification
    ) -> None:
        """Keep the agent's income verification on file, replacing any earlier one."""
        with self._changing(application_id) as stored:
            stored.income_verification = verification

    def record_ending(self, application_id: str, ending: CaseEnding) -> None:
        """Keep the ending of the case on file."""
        with self._changing(application_id) as stored:
            stored.ending = ending

    def record_credit_check(self, application_id: str, result: CreditCheckResult) -> None:
        """Keep the credit check's result on file and log it, in one step."""
        with self._changing(application_id) as stored:
            stored.credit_check = result
            self.log_event(
                application_id,
                result.checked_at,
                "system",
                "credit_checked",
                "passed" if result.passed else "not passed",
            )

    def record_offers(self, application_id: str, offers: list[Offer], now: datetime) -> None:
        """Keep the created offers on file and log them, in one step. The log
        names the terms only."""
        with self._changing(application_id) as stored:
            stored.offers = offers
            terms = ", ".join(str(offer.term_months) for offer in offers)
            self.log_event(application_id, now, "system", "offers_created", f"{terms} months")

    def correct_fields(self, application_id: str, changes: dict, now: datetime) -> None:
        """The customer corrects form fields, as typed (for example
        {"annual_income": "78000"}). Only the field names are logged, never
        the values.

        A consent given in a correction is timestamped with the time of the
        correction, as intake does at submission; a consent taken back loses
        its timestamp. A consent that was already given keeps its first time.
        """
        unknown = set(changes) - set(RawApplication.model_fields)
        if unknown:
            raise ValueError(f"unknown form fields: {sorted(unknown)}")
        with self._changing(application_id) as stored:
            stored.raw = RawApplication.model_validate(stored.raw.model_dump() | changes)
            stored.form_version += 1
            metadata = stored.metadata
            for field, time_field in CONSENT_TIMES.items():
                if field not in changes:
                    continue
                if not is_yes(changes[field]):
                    setattr(metadata, time_field, None)
                elif getattr(metadata, time_field) is None:
                    setattr(metadata, time_field, now)
            self.log_event(
                application_id, now, "customer", "form_corrected", ", ".join(sorted(changes))
            )

    def log_event(
        self,
        application_id: str,
        at: datetime,
        actor: Actor,
        event: EventName,
        details: str = "",
        reason: str | None = None,
    ) -> None:
        """Add one event to the case's audit log, and to its log file if the
        store has a run folder."""
        agent = self._agent if actor in ("agent", "guardrail") else None
        audit_event = AuditEvent(
            application_id=application_id,
            at=at,
            actor=actor,
            event=event,
            details=details,
            reason=reason,
            model=agent.model if agent else None,
            prompt_version=agent.prompt_version if agent else None,
        )
        self._append_event(audit_event)
        if self._run_dir is not None:
            case_dir = self._run_dir / application_id
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "audit.jsonl").open("a") as f:
                f.write(audit_event.model_dump_json() + "\n")

    def refuse_if_ended(self, application_id: str, now: datetime, tool: str) -> str | None:
        """For tools: once the case has ended, every action is refused and
        logged as blocked. Returns the refusal, or None if the case is open."""
        ending = self.get(application_id).ending
        if ending is None:
            return None
        self.log_event(
            application_id, now, "guardrail", "action_blocked", f"{tool}: the case has already ended"
        )
        return f"not done: the case has already ended ({ending.outcome}, {ending.reason})"
