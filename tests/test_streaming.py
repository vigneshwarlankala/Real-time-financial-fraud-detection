
from unittest.mock import Mock, patch

import pytest
import requests

from src.streaming.kafka_consumer import (
    prepare_payload,
    predict_transaction,
)


@pytest.fixture
def sample_transaction():
    return {
        "transaction_id": "TXN-TEST-001",
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 181.0,
        "timestamp": "2026-10-12T09:00:00+00:00",
    }


def test_transaction_payload_contains_expected_fields(sample_transaction):
    payload = prepare_payload(sample_transaction)

    expected_fields = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    }

    assert set(payload.keys()) == expected_fields
    assert "transaction_id" not in payload
    assert "timestamp" not in payload


def test_transaction_payload_preserves_values(sample_transaction):
    payload = prepare_payload(sample_transaction)

    assert payload["type"] == "TRANSFER"
    assert payload["amount"] == 181.0
    assert payload["oldbalanceOrg"] == 181.0
    assert payload["newbalanceOrig"] == 0.0


@patch("src.streaming.kafka_consumer.requests.post")
def test_predict_transaction_success(mock_post, sample_transaction):
    mock_response = Mock()
    mock_response.json.return_value = {
        "fraud_probability": 0.95,
        "prediction": 1,
        "risk_level": "HIGH",
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = predict_transaction(
        sample_transaction,
        api_url="https://example.com/predict",
    )

    assert result["fraud_probability"] == 0.95
    assert result["prediction"] == 1
    assert result["risk_level"] == "HIGH"

    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["amount"] == 181.0
    assert mock_post.call_args.kwargs["timeout"] == 60


@patch("src.streaming.kafka_consumer.requests.post")
def test_predict_transaction_api_failure(mock_post, sample_transaction):
    mock_post.side_effect = requests.ConnectionError("Connection failed")

    with pytest.raises(requests.ConnectionError):
        predict_transaction(sample_transaction)

    mock_post.assert_called_once()
