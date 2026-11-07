# Real-Time Financial Fraud Detection Platform

A production-grade AI/ML system that reads live transactions, scores each one for fraud risk in milliseconds, links suspicious accounts into fraud rings, and provides analysts with an interactive dashboard to review and action cases.

## Live Demo
| Service | URL |
|---------|-----|
| Dashboard | https://aiml-01-fraud-dashboard.onrender.com |
| API | https://aiml-01-fraud-detection.onrender.com |
| API Docs (Swagger) | https://aiml-01-fraud-detection.onrender.com/docs |

---

## Architecture

```
Kafka Producer
     │
     ▼
fraud-transactions topic (Kafka)
     │
     ▼
Kafka Consumer ──► Feast Feature Store (online) ──► XGBoost + GNN Hybrid Model
     │                                                        │
     ▼                                                        ▼
prediction_results.csv                              Fraud Score + Risk Level
                                                             │
                                             ┌───────────────┼───────────────┐
                                             ▼               ▼               ▼
                                        FastAPI          Streamlit       MLflow
                                        Service         Dashboard       Tracking
                                             │
                                             ▼
                                      Kubernetes (Kind)
                                             │
                                             ▼
                                    Evidently AI Drift Monitor
                                             │
                                             ▼
                                    Auto-Retraining Trigger
```

---

## Dataset

**PaySim** synthetic mobile-money transactions (Kaggle)
- Total transactions: **6,362,620**
- Fraudulent transactions: **8,213** (<0.13% — heavy class imbalance)
- Features: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest

---

## Models

### XGBoost (Primary Classifier)
| Metric | Value |
|--------|-------|
| Precision | 0.34 |
| Recall | **0.99** |
| F1-score | 0.51 |
| **PR-AUC** | **0.9651** |

> Optimised for recall — catching 99% of fraud at the cost of some false positives, appropriate for financial fraud detection.

### Graph Neural Network (Fraud Ring Detection)
- Framework: **PyTorch Geometric**
- Detects connected fraud rings via account-transaction graph
- Candidate fraud groups identified: **9,643**
- Groups with fraud-labeled edges: **111**
- Temporal bonus improves match rate by **+87%** over baseline (top-500 candidates)

### Hybrid Detector
XGBoost score blended with GNN embedding for final fraud probability.

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| ML Model | XGBoost, PyTorch Geometric (GNN) |
| Feature Store | Feast (SQLite online store) |
| Experiment Tracking | MLflow |
| Streaming | Apache Kafka |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Drift Monitoring | Evidently AI + PSI-based drift check |
| Containerisation | Docker |
| Orchestration | Kubernetes (Kind) |
| Deployment | Render (cloud) |
| Language | Python 3.11 |

---

## Repository Structure

```
AIML-01-Fraud-Detection/
├── api/                        # FastAPI prediction service
│   └── main.py
├── dashboard/                  # Streamlit analyst dashboard
│   ├── app.py
│   └── database.py
├── data/
│   ├── raw/transactions.csv    # PaySim dataset
│   └── processed/              # Engineered features
├── feature_store/
│   └── feature_repo/           # Feast feature definitions + registry
├── models/
│   ├── fraud_xgboost.joblib    # Trained XGBoost model
│   ├── fraud_gnn.pt            # Trained GNN model
│   └── fraud_gnn_scaler.joblib
├── reports/
│   ├── streaming/              # Kafka consumer prediction logs
│   ├── fraud_rings/            # Fraud ring detection results
│   ├── data_drift_report.html  # Evidently AI HTML report
│   ├── drift_check_smoke.json  # PSI drift check results
│   ├── retrain_log.json        # Auto-retraining decision log
│   └── model_metrics.json
├── src/
│   ├── streaming/
│   │   ├── kafka_producer.py   # Publishes transactions to Kafka
│   │   └── kafka_consumer.py   # Consumes, scores, and logs predictions
│   ├── monitoring/
│   │   └── drift_check.py      # PSI-based feature drift detection
│   ├── train_model.py          # XGBoost training pipeline
│   ├── train_gnn.py            # GNN training pipeline
│   ├── hybrid_fraud_detector.py
│   ├── fraud_ring_detector.py  # Graph-based fraud ring analysis
│   ├── drift_monitor.py        # Evidently AI drift report generator
│   └── retrain_trigger.py      # Auto-retraining trigger
├── tests/                      # 20 automated tests (all passing)
├── k8s-deployment.yaml         # Kubernetes deployment manifest
├── k8s-service.yaml            # Kubernetes NodePort service
├── k8s-pvc.yaml                # Persistent volume claim
├── Dockerfile
└── requirements.txt
```

---

## Quick Start — Local Setup

### Prerequisites
- Python 3.11
- Docker Desktop
- kubectl
- kind (Kubernetes in Docker)

### 1. Clone and create environment
```powershell
git clone <repo-url>
cd AIML-01-Fraud-Detection
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start Kafka
```powershell
docker start fraud-kafka
# Verify
docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

### 3. Start the API
```powershell
docker start fraud-api-test
curl http://localhost:8001/health
```

### 4. Test fraud prediction
```powershell
Invoke-RestMethod -Uri http://localhost:8001/predict -Method POST `
  -ContentType "application/json" -Body '{
    "step": 1, "amount": 9999.99,
    "oldbalanceOrg": 10000.00, "newbalanceOrig": 0.01,
    "oldbalanceDest": 0.0, "newbalanceDest": 9999.99,
    "type": "TRANSFER"
  }'
```

### 5. Run Kafka producer → consumer integration
```powershell
# Terminal 1 — Consumer
python src/streaming/kafka_consumer.py

# Terminal 2 — Producer
python src/streaming/kafka_producer.py
```

### 6. Feast online feature retrieval
```powershell
python -c "
from feast import FeatureStore
store = FeatureStore(repo_path='feature_store/feature_repo')
print(store.get_online_features(
    features=['origin_transaction_features:amount'],
    entity_rows=[{'nameOrig': 'C1231006815'}],
).to_dict())
"
```

### 7. Run drift check and auto-retraining trigger
```powershell
python src/retrain_trigger.py \
  --reference tests/drift_reference_smoke.csv \
  --current   tests/drift_current_smoke.csv
```

### 8. Deploy to Kubernetes (Kind)
```powershell
kind load docker-image fraud-detection-api:v1 --name fraud-detection
kubectl apply -f k8s-pvc.yaml
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl get pods   # wait for 1/1 Running
kubectl port-forward service/fraud-detection-api-service 9090:8000
curl http://localhost:9090/health
```

### 9. Run all tests
```powershell
python -m pytest tests/ -v
# Expected: 20 passed
```

---

## Fraud Ring Detection

Runs graph-based analysis on raw transaction data to identify coordinated fraud networks:

```powershell
# Quick smoke test (steps 1–10)
python src/fraud_ring_detector.py --start-step 1 --end-step 10

# Results saved to reports/fraud_rings/
```

**Results summary:**
- Candidate groups: 9,643
- Groups with fraud-labeled edges: 111
- Temporal bonus vs baseline: +87% match rate improvement at top-500

---

## Data Drift Monitoring

```powershell
# Generate Evidently AI HTML report
python src/drift_monitor.py

# PSI-based drift check
python src/monitoring/drift_check.py \
  --reference data/processed/transactions_processed.csv \
  --current   data/processed/transactions_processed.csv \
  --output    reports/drift_check.json

# Auto-retraining trigger (runs drift check → retrains if drift > 0.2 PSI)
python src/retrain_trigger.py \
  --reference <reference.csv> \
  --current   <current.csv>
```

---

## Test Results

```
20 tests collected — 20 passed
```

| Test Module | Tests |
|-------------|-------|
| test_api.py | API endpoints, validation, prediction |
| test_streaming.py | Kafka producer/consumer |
| test_feast_integration.py | Feature store definitions |
| test_analyst_cases.py | Case management CRUD |
| test_drift_check.py | PSI drift detection |
| test_hybrid_fraud_detector.py | Hybrid model scoring |

---

## Limitations

This is an educational demonstration, not a certified financial security system. A high-risk prediction does not prove fraud. Human review is required before any real-world action. PSI drift scores are screening signals, not proof of model degradation.
