
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    transaction = {
        "step": 1,
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "type": "TRANSFER",
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "prediction" in data
    assert "risk_level" in data

    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["HIGH", "LOW"]


def test_invalid_transaction_type():
    transaction = {
        "step": 1,
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "type": "INVALID_TYPE",
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422
