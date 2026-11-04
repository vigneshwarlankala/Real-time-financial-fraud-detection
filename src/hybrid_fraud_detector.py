from pathlib import Path
import json
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
XGB_PATH = ROOT / "models" / "fraud_xgboost.joblib"
GNN_PATH = ROOT / "models" / "fraud_gnn.pt"
GNN_METRICS_PATH = ROOT / "reports" / "gnn_metrics.json"
HYBRID_METRICS_PATH = ROOT / "reports" / "hybrid_metrics.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gnn_inference import GNNAccountRiskModel


def load_xgboost_model():
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {XGB_PATH}")
    return joblib.load(XGB_PATH)


def blend_risk_scores(
    xgboost_probability: float,
    gnn_probability: float,
    xgboost_weight: float = 0.8,
) -> dict:
    """Experimental blend of transaction and account-level risk scores."""
    if not 0 <= xgboost_probability <= 1:
        raise ValueError("XGBoost probability must be between 0 and 1.")
    if not 0 <= gnn_probability <= 1:
        raise ValueError("GNN probability must be between 0 and 1.")
    if not 0 <= xgboost_weight <= 1:
        raise ValueError("XGBoost weight must be between 0 and 1.")

    gnn_weight = 1.0 - xgboost_weight
    combined = (
        xgboost_weight * xgboost_probability
        + gnn_weight * gnn_probability
    )

    risk_level = (
        "HIGH" if combined >= 0.80
        else "MEDIUM" if combined >= 0.40
        else "LOW"
    )

    return {
        "xgboost_probability": round(float(xgboost_probability), 6),
        "gnn_account_risk": round(float(gnn_probability), 6),
        "xgboost_weight": xgboost_weight,
        "gnn_weight": round(gnn_weight, 6),
        "combined_risk_score": round(float(combined), 6),
        "risk_level": risk_level,
        "note": (
            "Experimental score only. XGBoost scores transactions while "
            "GraphSAGE scores accounts. The blended score is not calibrated "
            "for production decisions."
        ),
    }


def save_hybrid_metrics(metrics: dict) -> None:
    HYBRID_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYBRID_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def main():
    xgb_model = load_xgboost_model()
    gnn_metrics = (
        json.loads(GNN_METRICS_PATH.read_text(encoding="utf-8"))
        if GNN_METRICS_PATH.exists()
        else {}
    )

    print("Loading trained GraphSAGE inference...")
    gnn = GNNAccountRiskModel()

    sample_account = next(iter(gnn.account_ids))
    account_score = gnn.get_account_risk(sample_account)

    result = {
        "xgboost_model_loaded": xgb_model is not None,
        "gnn_checkpoint_loaded": GNN_PATH.exists(),
        "gnn_inference_ready": account_score is not None,
        "gnn_model_type": gnn_metrics.get("model", "GraphSAGE"),
        "gnn_target": gnn_metrics.get("target", "account-level risk"),
        "sample_account_id": sample_account,
        "sample_account_risk": round(account_score, 6),
        "integration_status": (
            "XGBoost and real GNN account-risk inference loaded successfully."
        ),
        "limitations": [
            "The sample GNN covers a subset of PaySim transactions.",
            "An account absent from the sampled graph has no GNN score.",
            "The blended score requires validation and calibration.",
            "This module does not yet score a transaction end-to-end."
        ],
    }

    save_hybrid_metrics(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

def score_transaction(transaction: dict, sender_account_id: str, gnn_detector):
    """Score a transaction with XGBoost and look up its sender's GNN risk."""
    import pandas as pd

    xgb_model = load_xgboost_model()
    transaction = dict(transaction)
    transaction_type = transaction.pop("type")

    valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    if transaction_type not in valid_types:
        raise ValueError(f"Unsupported transaction type: {transaction_type}")

    transaction["origin_balance_change"] = (
        transaction["oldbalanceOrg"] - transaction["newbalanceOrig"]
    )
    transaction["destination_balance_change"] = (
        transaction["newbalanceDest"] - transaction["oldbalanceDest"]
    )

    for tx_type in valid_types:
        transaction[f"type_{tx_type}"] = int(transaction_type == tx_type)

    features = pd.DataFrame([transaction]).reindex(
        columns=xgb_model.feature_names_in_,
        fill_value=0,
    )
    xgb_probability = float(xgb_model.predict_proba(features)[0][1])
    gnn_probability = gnn_detector.get_account_risk(sender_account_id)

    result = {
        "sender_account_id": sender_account_id,
        "xgboost_transaction_probability": round(xgb_probability, 6),
        "gnn_account_risk": (
            round(gnn_probability, 6)
            if gnn_probability is not None else None
        ),
        "gnn_account_found": gnn_probability is not None,
        "note": (
            "Experimental research output, not a production fraud decision."
        ),
    }

    if gnn_probability is not None:
        result["experimental_blend"] = blend_risk_scores(
            xgb_probability, gnn_probability
        )
    else:
        result["experimental_blend"] = None
        result["note"] += (
            " Sender account is absent from the sampled GNN graph; "
            "no blended score was produced."
        )

    return result
