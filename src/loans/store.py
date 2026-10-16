"""The application store: every application, its metadata and its documents,
kept by application ID.

The agent never reads the store. Its tools do: each tool receives only the
application ID, looks up what it needs here, and returns a safe result
(reference passing). For now the store lives in memory and is filled from the
eval cases; later it moves to Postgres without changing the tools.
"""

from datetime import datetime

from pydantic import BaseModel

from loans.intake import create_metadata
from loans.schemas.application import RawApplication, RawFile, SubmissionMetadata
from loans.schemas.documents import UploadedDocument, UploadSlot

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


class StoredApplication(BaseModel):
    application_id: str
    raw: RawApplication
    metadata: SubmissionMetadata
    # The current document in each filled slot.
    documents: dict[UploadSlot, UploadedDocument]


class ApplicationStore:
    def __init__(self) -> None:
        self._applications: dict[str, StoredApplication] = {}

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

    def correct_fields(self, application_id: str, changes: dict) -> None:
        """The customer corrects form fields, as typed (for example
        {"annual_income": "78000"})."""
        stored = self.get(application_id)
        unknown = set(changes) - set(RawApplication.model_fields)
        if unknown:
            raise ValueError(f"unknown form fields: {sorted(unknown)}")
        stored.raw = RawApplication.model_validate(stored.raw.model_dump() | changes)
