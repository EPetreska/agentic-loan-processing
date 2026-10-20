"""The application store: every application, its metadata and its documents,
kept by application ID.

The agent never reads the store. Its tools do: each tool receives only the
application ID, looks up what it needs here, and returns a safe result
(reference passing). For now the store lives in memory and is filled from the
eval cases; later it moves to Postgres without changing the tools.

If the store is given a log folder, every audit event is also written to the
case's log file (one JSON line per event) the moment it happens, so a person
can review the log at any time and a crash loses nothing. The tests use no
folder, so nothing is written to disk.
"""

from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from loans.audit import Actor, AgentVersion, AuditEvent, EventName
from loans.intake import create_metadata, is_yes
from loans.schemas.application import RawApplication, RawFile, SubmissionMetadata
from loans.schemas.documents import IncomeSlot, UploadedDocument, UploadSlot
from loans.schemas.outcome import OutcomeKind, Reason
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
    # Messages sent to the customer, oldest first.
    messages: list[SentMessage] = []
    # Set once the agent has ended the case.
    ending: CaseEnding | None = None
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
        self, log_dir: Path | None = None, agent: AgentVersion | None = None
    ) -> None:
        self._applications: dict[str, StoredApplication] = {}
        self._log_dir = log_dir
        # Added to agent and guardrail events.
        self._agent = agent

    def submit(
        self,
        raw: RawApplication,
        documents: dict[str, UploadedDocument],
        now: datetime,
        application_id: str | None = None,
    ) -> str:
        """Store a new application (intake) and return its ID.

        documents holds the document behind each filled upload slot.
        """
        if set(documents) != set(filled_slots(raw)):
            raise ValueError(
                f"documents {sorted(documents)} do not match the filled "
                f"upload slots {sorted(filled_slots(raw))}"
            )
        metadata = create_metadata(raw, now, application_id)
        if metadata.application_id in self._applications:
            raise ValueError(f"application {metadata.application_id} already exists")
        self._applications[metadata.application_id] = StoredApplication(
            application_id=metadata.application_id,
            raw=raw,
            metadata=metadata,
            documents=documents,
            document_versions=dict.fromkeys(documents, 1),
        )
        self.log_event(
            metadata.application_id,
            now,
            "system",
            "application_submitted",
            "uploaded: " + (", ".join(sorted(documents)) or "nothing"),
        )
        return metadata.application_id

    def get(self, application_id: str) -> StoredApplication:
        try:
            return self._applications[application_id]
        except KeyError:
            raise UnknownApplication(application_id) from None

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
        stored = self.get(application_id)
        group, name = SLOT_PATHS[slot]
        form = stored.raw.model_dump()
        form[group][name] = file.model_dump()
        stored.raw = RawApplication.model_validate(form)
        stored.documents[slot] = document
        stored.document_versions[slot] = stored.document_versions.get(slot, 0) + 1
        self.log_event(application_id, now, "customer", "document_uploaded", slot)

    def record_message(self, application_id: str, message: SentMessage) -> None:
        """Keep a sent message on file."""
        self.get(application_id).messages.append(message)

    def record_identity_findings(
        self, application_id: str, findings: IdentityFindings
    ) -> None:
        """Keep the identity vendor's latest findings on file."""
        self.get(application_id).identity_findings = findings

    def record_income_verification(
        self, application_id: str, verification: IncomeVerification
    ) -> None:
        """Keep the agent's income verification on file, replacing any earlier one."""
        self.get(application_id).income_verification = verification

    def record_ending(self, application_id: str, ending: CaseEnding) -> None:
        """Keep the ending of the case on file."""
        self.get(application_id).ending = ending

    def correct_fields(self, application_id: str, changes: dict, now: datetime) -> None:
        """The customer corrects form fields, as typed (for example
        {"annual_income": "78000"}). Only the field names are logged, never
        the values.

        A consent given in a correction is timestamped with the time of the
        correction, as intake does at submission; a consent taken back loses
        its timestamp. A consent that was already given keeps its first time.
        """
        stored = self.get(application_id)
        unknown = set(changes) - set(RawApplication.model_fields)
        if unknown:
            raise ValueError(f"unknown form fields: {sorted(unknown)}")
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
        store has a log folder."""
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
        self.get(application_id).audit_log.append(audit_event)
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            with (self._log_dir / f"{application_id}.jsonl").open("a") as f:
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
