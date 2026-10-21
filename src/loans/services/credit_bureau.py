"""check_credit: the mock credit bureau.

After the application agent completes a case, the bank's workflow (not the
agent) runs the credit check. Only complete cases are checked: cancelled
cases and cases sent to a human go no further in this project.

For now the check always passes, so a case can go on to the next steps (the
offer agent). A realistic report (credit score, debts) comes with the
decision model.

Frame rule, enforced here: no credit pull without the customer's credit
check consent on file. A complete case always has it (the form check
requires it), so this is a second lock.
"""

from datetime import datetime

from pydantic import BaseModel

from loans.schemas.application import SubmissionMetadata


class NoCreditConsent(Exception):
    """The credit check was asked for without the customer's consent."""


class CreditCheckResult(BaseModel):
    passed: bool
    checked_at: datetime


def check_credit(metadata: SubmissionMetadata, now: datetime) -> CreditCheckResult:
    """Pull the customer's credit; refused without their consent on file."""
    if metadata.credit_check_consented_at is None:
        raise NoCreditConsent(f"application {metadata.application_id}: no credit check consent")
    return CreditCheckResult(passed=True, checked_at=now)
