"""check_income_document: the bank's rules for when a document can verify
the claimed income.

These are the bank's rules, not the vendor's: the extraction service only
reports what is on a document, and these rules decide whether the document
fits the employment status, is current, and shows a yearly income close
enough to the claimed income. The values come from policy.py.

The same code runs in two places: read_document shows the results to the
agent, so the agent never compares numbers itself, and mark_income_verified
refuses to record a verification that breaks a rule, even if the agent asks
for it by mistake or because of a manipulation.

What stays the agent's judgment, stated in its reason: whether the document
is the applicant's (the name) and from the applicant's current job (the
employer), and whether the customer's explanations make sense.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, TypeAdapter, ValidationError

from loans import policy
from loans.income import annualize
from loans.schemas.application import CleanApplication, RawApplication
from loans.schemas.documents import PayStub, TaxReturn, UploadedDocument, W2

# Which documents can verify income, by employment status.
DOCUMENTS_BY_STATUS: dict[str, set[str]] = {
    "employed": {"pay_stub", "w2"},
    "self_employed": {"tax_return"},
}

MAX_YEARS_OLD: dict[str, int] = {
    "w2": policy.W2_MAX_YEARS_OLD,
    "tax_return": policy.TAX_RETURN_MAX_YEARS_OLD,
}


class IncomeRuleResult(BaseModel):
    slot: str
    # One plain-language line per rule that is not met. Empty if all are met.
    not_met: list[str] = []

    @property
    def met(self) -> bool:
        return not self.not_met

    def to_llm_text(self) -> str:
        """The text the agent sees. Contains no customer-written text, only
        numbers, dates and fixed wording."""
        if self.met:
            return "bank income rules: met"
        return "\n".join(
            ["bank income rules: not met", *(f"- {line}" for line in self.not_met)]
        )


def _parse(field: str, value):
    """Parse one form value with the same type as in CleanApplication, or None."""
    annotation = CleanApplication.model_fields[field].rebuild_annotation()
    try:
        return TypeAdapter(annotation).validate_python(value)
    except ValidationError:
        return None


def _status_problem(status: str | None, slot: str) -> str | None:
    if status is None:
        return "the employment status on the form is missing or not valid"
    if status not in DOCUMENTS_BY_STATUS:
        return "income is verified with documents only for employed or self-employed applicants"
    if slot in DOCUMENTS_BY_STATUS[status]:
        return None
    if status == "employed":
        return "for an employed applicant, income is verified with a pay stub or a W-2"
    return "for a self-employed applicant, income is verified with a tax return"


def _yearly_income(content: PayStub | W2 | TaxReturn) -> Decimal:
    if isinstance(content, PayStub):
        return annualize(content.gross_pay, content.pay_frequency)
    if isinstance(content, W2):
        return content.wages
    return content.net_profit  # not gross receipts, which are before expenses


def _age_problem(content: PayStub | W2 | TaxReturn, slot: str, today: date) -> str | None:
    if isinstance(content, PayStub):
        days = (today - content.pay_date).days
        if days > policy.PAY_STUB_MAX_AGE_DAYS:
            return (
                f"too old: the pay date was {days} days ago "
                f"(the limit is {policy.PAY_STUB_MAX_AGE_DAYS} days)"
            )
        return None
    oldest_year = today.year - MAX_YEARS_OLD[slot]
    if content.tax_year < oldest_year:
        return (
            f"too old: tax year {content.tax_year} "
            f"(tax year {oldest_year} or later is required)"
        )
    return None


def _amount_problem(yearly: Decimal, claimed: Decimal) -> str | None:
    difference = (yearly - claimed) / claimed
    if abs(difference) <= policy.INCOME_TOLERANCE:
        return None
    percent = abs(difference) * 100
    direction = "below" if difference < 0 else "above"
    limit = policy.INCOME_TOLERANCE * 100
    return (
        f"the yearly income on the document ({yearly}) is {percent:.1f}% {direction} "
        f"the claimed income ({claimed}) (the limit is {limit:.0f}%)"
    )


def check_income_document(
    document: UploadedDocument, slot: str, raw: RawApplication, today: date
) -> IncomeRuleResult:
    """Apply the bank's income rules to the document in one income slot."""
    result = IncomeRuleResult(slot=slot)

    status_problem = _status_problem(_parse("employment_status", raw.employment_status), slot)
    if status_problem:
        result.not_met.append(status_problem)

    if document.actual_type != slot:
        result.not_met.append(f"the document in this slot is not a {slot}")
        return result
    content = document.parsed()
    if content is None:
        result.not_met.append("the document could not be read")
        return result

    age_problem = _age_problem(content, slot, today)
    if age_problem:
        result.not_met.append(age_problem)

    claimed = _parse("annual_income", raw.annual_income)
    if claimed is None:
        result.not_met.append("the claimed income on the form is missing or not valid")
    else:
        amount_problem = _amount_problem(_yearly_income(content), claimed)
        if amount_problem:
            result.not_met.append(amount_problem)
    return result
