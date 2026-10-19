"""extract_document: the mock document extraction service.

A real system would read the uploaded PDF or image with a service such as AWS
Textract or a vision LLM. In this project each upload is mocked as an
UploadedDocument whose content is already known, so extraction never misreads
a value. What is simulated: what the file really is (document_type), whether
it is in the right slot (matches_slot), and unreadable scans.

The result holds only what the agent may see: the allowlisted fields of the
document (identifiers removed), values calculated in code so the agent does
not do arithmetic (annualized pay, the age of a pay stub), and any hidden
text, which the agent must treat as data, not instructions.
"""

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from loans.income import annualize
from loans.schemas.documents import PayStub, UploadedDocument
from loans.spotlight import quote


class ExtractionResult(BaseModel):
    slot: str
    document_type: str
    matches_slot: bool
    readable: bool
    # The allowlisted fields of the document, as JSON-friendly values.
    fields: dict[str, Any] = {}
    # Which of these fields are free text, to be quoted.
    text_fields: set[str] = set()
    # Calculated in code for pay stubs.
    annualized_gross: Decimal | None = None
    days_since_pay_date: int | None = None
    # Text found on the document outside its normal fields.
    hidden_text: str | None = None

    def to_llm_text(self) -> str:
        """The text the agent sees. Every piece of text from the document is
        quoted (spotlighting); numbers and dates are shown as they are."""
        lines = [
            f"slot: {self.slot}",
            f"document_type: {self.document_type}",
            f"matches_slot: {'yes' if self.matches_slot else 'no'}",
            f"readable: {'yes' if self.readable else 'no'}",
        ]
        for name, value in self.fields.items():
            shown = quote(value) if name in self.text_fields else value
            lines.append(f"{name}: {shown}")
        if self.annualized_gross is not None:
            lines.append(f"annualized_gross: {self.annualized_gross} (calculated)")
        if self.days_since_pay_date is not None:
            lines.append(f"days_since_pay_date: {self.days_since_pay_date} (calculated)")
        if self.hidden_text is not None:
            lines.append(f"other text found on the document: {quote(self.hidden_text)}")
        return "\n".join(lines)


def _text_fields(model: type[BaseModel]) -> set[str]:
    """The fields typed as free text (str). Dates, numbers and fixed choices
    such as the pay frequency are validated, so they cannot carry injected
    text and are shown without quotes."""
    return {
        name
        for name, info in model.model_fields.items()
        if info.annotation in (str, str | None)
    }


def extract_document(document: UploadedDocument, slot: str, today: date) -> ExtractionResult:
    """Read one uploaded document, as a mock extraction service."""
    result = ExtractionResult(
        slot=slot,
        document_type=document.actual_type,
        matches_slot=document.actual_type == slot,
        readable=document.readable,
    )
    content = document.parsed()
    if content is None:
        return result

    result.fields = content.redacted()
    result.text_fields = _text_fields(type(content))
    result.hidden_text = document.hidden_text
    if isinstance(content, PayStub):
        result.annualized_gross = annualize(content.gross_pay, content.pay_frequency)
        result.days_since_pay_date = (today - content.pay_date).days
    return result
