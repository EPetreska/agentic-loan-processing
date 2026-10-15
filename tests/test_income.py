"""Tests for the income calculations."""

from decimal import Decimal

import pytest

from loans.income import annualize


@pytest.mark.parametrize(
    "frequency, pay, expected",
    [
        ("biweekly", Decimal("2850.00"), Decimal("74100.00")),  # the Maria example
        ("weekly", Decimal("1000"), Decimal("52000")),
        ("semimonthly", Decimal("3000"), Decimal("72000")),
        ("monthly", Decimal("5000"), Decimal("60000")),
    ],
)
def test_annualize(frequency, pay, expected):
    assert annualize(pay, frequency) == expected


def test_biweekly_and_semimonthly_differ():
    # The classic trap: 26 versus 24 pay periods a year.
    assert annualize(Decimal("2850"), "biweekly") - annualize(Decimal("2850"), "semimonthly") == Decimal("5700")
