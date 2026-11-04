
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.analyst_cases import create_case, list_cases, update_case

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_xgboost.joblib"

app = FastAPI(
    title="Real-Time Financial Fraud Detection API",
    description=(
        "Educational fraud-risk prediction API with analyst case management. "
        "Predictions are not proof of fraudulent activity."
    ),
    version="1.1.0",
)

VALID_TRANSACTION_TYPES = {
    "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER",
}

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

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
            raise ValueError(f"Unsupported transaction type: {value}")
        return value


class CaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=200)
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: str = "HIGH"

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, value):
        value = value.upper().strip()
        if value not in VALID_RISK_LEVELS:
            raise ValueError("risk_level must be LOW, MEDIUM, or HIGH")
        return value


class CaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    analyst_notes: str = Field(default="", max_length=5000)


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
        data[f"type_{tx_type}"] = int(transaction_type == tx_type)

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


@app.post("/cases", status_code=201)
def create_analyst_case(payload: CaseCreate):
    try:
        case, created = create_case(
            transaction_id=payload.transaction_id,
            fraud_probability=payload.fraud_probability,
            risk_level=payload.risk_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not created:
        raise HTTPException(
            status_code=409,
            detail="A case already exists for this transaction_id",
        )

    return {"case": case}


@app.get("/cases")
def get_analyst_cases(
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        return {"cases": list_cases(status=status, limit=limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/cases/{case_id}")
def review_analyst_case(case_id: int, payload: CaseUpdate):
    try:
        case = update_case(
            case_id=case_id,
            status=payload.status,
            analyst_notes=payload.analyst_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    return {"case": case}
