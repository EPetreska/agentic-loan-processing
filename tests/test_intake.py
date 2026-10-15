"""Tests for create_metadata: the values only the system sets."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from loans.intake import create_metadata
from loans.schemas.application import RawApplication

NOW = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def test_generates_a_random_application_id():
    first = create_metadata(RawApplication(), NOW)
    second = create_metadata(RawApplication(), NOW)
    assert UUID(first.application_id).version == 4
    assert first.application_id != second.application_id


def test_keeps_a_fixed_application_id_for_evals():
    metadata = create_metadata(RawApplication(), NOW, application_id="EVAL-001")
    assert metadata.application_id == "EVAL-001"


def test_submission_time_is_the_time_passed_in():
    assert create_metadata(RawApplication(), NOW).submitted_at == NOW


@pytest.mark.parametrize("answer", ["yes", "true", "on", True])
def test_consent_yes_records_the_time(answer):
    raw = RawApplication(credit_check_consent=answer, esign_consent=answer)
    metadata = create_metadata(raw, NOW)
    assert metadata.credit_check_consented_at == NOW
    assert metadata.esign_consented_at == NOW


@pytest.mark.parametrize("answer", ["no", "maybe", "", None, False])
def test_unclear_or_no_consent_records_no_time(answer):
    raw = RawApplication(credit_check_consent=answer, esign_consent=answer)
    metadata = create_metadata(raw, NOW)
    assert metadata.credit_check_consented_at is None
    assert metadata.esign_consented_at is None


def test_time_without_time_zone_is_rejected():
    with pytest.raises(ValueError, match="time zone"):
        create_metadata(RawApplication(), datetime(2026, 3, 1, 10, 0))


def test_customer_cannot_send_system_values():
    # A tampered form that tries to set its own ID and time is ignored.
    raw = RawApplication.model_validate(
        {"application_id": "HACK", "submitted_at": "2020-01-01T00:00:00Z"}
    )
    metadata = create_metadata(raw, NOW)
    assert metadata.application_id != "HACK"
    assert metadata.submitted_at == NOW
