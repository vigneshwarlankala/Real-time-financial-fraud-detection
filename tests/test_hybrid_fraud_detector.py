from src.hybrid_fraud_detector import score_transaction


SAMPLE_TRANSACTION = {
    "step": 1,
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "type": "TRANSFER",
}


class FakeGNN:
    def get_account_risk(self, account_id):
        if account_id == "known-account":
            return 0.7
        return None


def test_hybrid_scoring_known_account(monkeypatch):
    class FakeXGBoost:
        feature_names_in_ = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
            "origin_balance_change", "destination_balance_change",
            "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
            "type_PAYMENT", "type_TRANSFER",
        ]

        def predict_proba(self, features):
            return [[0.1, 0.9]]

    monkeypatch.setattr(
        "src.hybrid_fraud_detector.load_xgboost_model",
        lambda: FakeXGBoost(),
    )

    result = score_transaction(
        SAMPLE_TRANSACTION, "known-account", FakeGNN()
    )

    assert result["gnn_account_found"] is True
    assert result["xgboost_transaction_probability"] == 0.9
    assert result["gnn_account_risk"] == 0.7
    assert result["experimental_blend"]["risk_level"] == "HIGH"


def test_hybrid_scoring_unknown_account(monkeypatch):
    class FakeXGBoost:
        feature_names_in_ = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
            "origin_balance_change", "destination_balance_change",
            "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
            "type_PAYMENT", "type_TRANSFER",
        ]

        def predict_proba(self, features):
            return [[0.8, 0.2]]

    monkeypatch.setattr(
        "src.hybrid_fraud_detector.load_xgboost_model",
        lambda: FakeXGBoost(),
    )

    result = score_transaction(
        SAMPLE_TRANSACTION, "unknown-account", FakeGNN()
    )

    assert result["gnn_account_found"] is False
    assert result["gnn_account_risk"] is None
    assert result["experimental_blend"] is None


def test_hybrid_scoring_rejects_invalid_transaction_type(monkeypatch):
    invalid_transaction = dict(SAMPLE_TRANSACTION)
    invalid_transaction["type"] = "INVALID"

    class UnusedGNN:
        def get_account_risk(self, account_id):
            return None

    try:
        score_transaction(invalid_transaction, "account", UnusedGNN())
        assert False, "Expected ValueError for invalid transaction type"
    except ValueError as exc:
        assert "Unsupported transaction type" in str(exc)