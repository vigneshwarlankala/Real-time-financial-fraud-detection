import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "analyst_cases.py"
spec = importlib.util.spec_from_file_location("analyst_cases", MODULE_PATH)
analyst_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyst_cases)


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    monkeypatch.setattr(analyst_cases, "DB_PATH", tmp_path / "test_cases.db")


def test_create_case_starts_open(isolated_db):
    case, created = analyst_cases.create_case("TX-1001", 0.98)
    assert created is True
    assert case["transaction_id"] == "TX-1001"
    assert case["status"] == "OPEN"
    assert case["fraud_probability"] == pytest.approx(0.98)


def test_update_case_records_analyst_decision(isolated_db):
    case, _ = analyst_cases.create_case("TX-1002", 0.91)
    updated = analyst_cases.update_case(
        case["case_id"], "CONFIRMED_FRAUD", "Verified by analyst"
    )
    assert updated["status"] == "CONFIRMED_FRAUD"
    assert updated["analyst_notes"] == "Verified by analyst"


def test_duplicate_transaction_does_not_create_duplicate_case(isolated_db):
    first, created_first = analyst_cases.create_case("TX-1003", 0.85)
    second, created_second = analyst_cases.create_case("TX-1003", 0.85)
    assert created_first is True
    assert created_second is False
    assert first["case_id"] == second["case_id"]
    assert len(analyst_cases.list_cases()) == 1


def test_invalid_status_is_rejected(isolated_db):
    case, _ = analyst_cases.create_case("TX-1004", 0.80)
    with pytest.raises(ValueError):
        analyst_cases.update_case(case["case_id"], "MAYBE_FRAUD")


def test_invalid_probability_is_rejected(isolated_db):
    with pytest.raises(ValueError):
        analyst_cases.create_case("TX-1005", 1.5)
