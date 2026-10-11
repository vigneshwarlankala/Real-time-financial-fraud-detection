
import json
from pathlib import Path

import joblib
import mlflow
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "transactions_processed.csv"
MODEL_PATH = ROOT / "models" / "fraud_xgboost.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
MLFLOW_DB_PATH = ROOT / "mlflow.db"

TARGET = "isFraud"

# --------------------------------------------------
# 2. Configure MLflow using SQLite
# --------------------------------------------------
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

mlflow.set_experiment("Financial-Fraud-Detection")

# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------
print("Loading processed transaction dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("Total transactions:", len(df))
print("Fraudulent transactions:", int(y.sum()))
print("Legitimate transactions:", int((y == 0).sum()))

# --------------------------------------------------
# 4. Split dataset
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

fraud_count = int(y_train.sum())
legitimate_count = int((y_train == 0).sum())

scale_weight = legitimate_count / fraud_count

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 5. Define model parameters
# --------------------------------------------------
params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_weight,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": 42,
}

# --------------------------------------------------
# 6. Train model and track experiment
# --------------------------------------------------
print("\nStarting model training and MLflow tracking...")

with mlflow.start_run(run_name="xgboost-fraud-detection") as run:

    mlflow.log_params(params)
    mlflow.log_param("dataset", "PaySim")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("decision_threshold", 0.5)

    mlflow.log_metric("training_rows", len(X_train))
    mlflow.log_metric("test_rows", len(X_test))
    mlflow.log_metric("fraud_training_rows", fraud_count)

    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 7. Evaluate model
    # --------------------------------------------------
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    average_precision = average_precision_score(
        y_test, probabilities
    )

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    matrix = confusion_matrix(
        y_test, predictions
    ).tolist()

    metrics = {
        "average_precision": float(average_precision),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": 0.5,
        "confusion_matrix": matrix,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    mlflow.log_metric("average_precision", average_precision)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # --------------------------------------------------
    # 8. Save model and metrics
    # --------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save output files as MLflow artifacts
    mlflow.log_artifact(str(METRICS_PATH))
    mlflow.log_artifact(str(MODEL_PATH))

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )

    print("Average Precision (PR-AUC):", round(average_precision, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("Confusion Matrix:", matrix)
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("MLflow database:", MLFLOW_DB_PATH)
    print("MLflow run ID:", run.info.run_id)

print("\nTraining and MLflow logging completed successfully.")
