"""Income calculations.

Arithmetic is done in code, not by the agent: LLMs make small arithmetic
mistakes, and "biweekly" (26 times a year) is easily confused with
"semimonthly" (24 times a year). The agent reasons about the result; the code
computes it, exactly and the same way every time.
"""

from decimal import Decimal
from typing import Literal

PayFrequency = Literal["weekly", "biweekly", "semimonthly", "monthly"]

PERIODS_PER_YEAR: dict[str, int] = {
    "weekly": 52,
    "biweekly": 26,
    "semimonthly": 24,
    "monthly": 12,
}


def annualize(pay: Decimal, frequency: PayFrequency) -> Decimal:
    """Pay for a full year at this rate, for example 2,850 biweekly is 74,100."""
    return pay * PERIODS_PER_YEAR[frequency]
