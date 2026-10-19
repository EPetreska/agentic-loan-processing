"""How a case ends: the outcome and its reason.

Reasons come from a fixed list per outcome, so an eval can compare them
exactly. The agent's final tool uses the same list, so the agent can only end
a case with a valid outcome and reason.
"""

from typing import Literal

from pydantic import BaseModel, model_validator

OutcomeKind = Literal["complete", "cancelled", "sent_to_human"]

Reason = Literal[
    # complete: ready for the credit check
    "all_checks_passed",
    # cancelled
    "no_answer",  # the deadline passed without any reply
    "documents_not_provided",  # replied, but never sent the required documents
    "customer_withdrew",  # no longer wants the loan
    "credit_consent_refused",  # keeps refusing the credit check
    # sent_to_human
    "other_employment",  # neither employed nor self-employed (Regulation B)
    "esign_declined",  # a paper process is needed
    "identity_review",  # a name or date of birth difference on the ID that could not be resolved
    "identity_failed",  # possible fraud: a person decides (fraud screening is not simulated in Phase 1)
    "under_18",
    "suspected_manipulation",  # for example an attack attempt was detected
    "customer_requested_human",
    "agent_unsure",  # cannot resolve the case with confidence
]

REASONS_BY_OUTCOME: dict[str, set[str]] = {
    "complete": {"all_checks_passed"},
    "cancelled": {
        "no_answer",
        "documents_not_provided",
        "customer_withdrew",
        "credit_consent_refused",
    },
    "sent_to_human": {
        "other_employment",
        "esign_declined",
        "identity_review",
        "identity_failed",
        "under_18",
        "suspected_manipulation",
        "customer_requested_human",
        "agent_unsure",
    },
}


class Outcome(BaseModel):
    outcome: OutcomeKind
    reason: Reason

    @model_validator(mode="after")
    def reason_fits_outcome(self) -> "Outcome":
        if self.reason not in REASONS_BY_OUTCOME[self.outcome]:
            raise ValueError(
                f"reason {self.reason!r} does not belong to outcome {self.outcome!r}"
            )
        return self
