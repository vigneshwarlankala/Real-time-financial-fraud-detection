from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_xgboost.joblib"

app = FastAPI(
    title="Real-Time Financial Fraud Detection API",
    description=(
        "Educational fraud-risk prediction API. "
        "Predictions are not proof of fraudulent activity."
    ),
    version="1.0.0",
)

VALID_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    type: str

    @field_validator("type")
    @classmethod
    def validate_transaction_type(cls, value):
        if value not in VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Unsupported transaction type: {value}"
            )
        return value


@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.model_dump()
    transaction_type = data.pop("type")

    data["origin_balance_change"] = (
        data["oldbalanceOrg"] - data["newbalanceOrig"]
    )
    data["destination_balance_change"] = (
        data["newbalanceDest"] - data["oldbalanceDest"]
    )

    for tx_type in VALID_TRANSACTION_TYPES:
        data[f"type_{tx_type}"] = int(
            transaction_type == tx_type
        )

    features = pd.DataFrame([data])
    features = features.reindex(
        columns=model.feature_names_in_,
        fill_value=0,
    )

    probability = float(model.predict_proba(features)[0][1])
    prediction = int(probability >= 0.5)

    return {
        "fraud_probability": round(probability, 6),
        "prediction": prediction,
        "risk_level": "HIGH" if prediction == 1 else "LOW",
        "threshold": 0.5,
        "disclaimer": (
            "Educational prediction only; a high-risk result "
            "does not establish that a transaction is fraudulent."
        ),
    }