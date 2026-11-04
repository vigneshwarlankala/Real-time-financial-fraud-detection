from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_REPO = PROJECT_ROOT / "feature_store" / "feature_repo"
sys.path.insert(0, str(FEATURE_REPO))

import feature_definitions as fd


def test_feast_parquet_source_exists():
    assert fd.OUTPUT_FILE.exists(), f"Missing Feast Parquet source: {fd.OUTPUT_FILE}"


def test_feast_feature_views_are_defined():
    assert fd.origin_transaction_features is not None
    assert fd.destination_transaction_features is not None


def test_feast_source_uses_event_timestamp():
    assert fd.transactions_source.timestamp_field == "event_timestamp"
