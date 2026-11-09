# AIML-01 — Real-Time Financial Fraud Detection Platform

**Final documentation update:** 9 November 2026

## 1. Project overview

A real-time financial fraud detection platform that uses machine learning and supporting data-engineering components to identify potentially fraudulent financial transactions and provide predictions through an API and dashboard.

## 2. Dataset and machine learning

- Dataset: PaySim
- Transactions prepared: 6,362,620
- Primary model: XGBoost
- Model evaluation report: saved
- Precision: 0.34
- Recall: 0.99
- F1-score: 0.51
- PR-AUC: 0.9651

## 3. Implemented components

- XGBoost fraud detection model and saved model artifact.
- FastAPI prediction service.
- Streamlit monitoring and prediction dashboard.
- Kafka producer and consumer implementation.
- MLflow experiment tracking configuration.
- GNN model artifact and fraud-ring detection implementation.
- Feast feature-view definitions and integration tests.
- Drift-monitoring implementation and smoke tests.
- Real-data drift report.
- Docker and Kubernetes deployment configuration.

## 4. Verified working components

- FastAPI service deployed and health-checked.
- Live prediction endpoint tested successfully.
- Streamlit dashboard deployed and confirmed working.
- Dashboard-to-API connection verified.
- Kubernetes deployment performed and verified using Kind.
- Drift-monitoring smoke test completed.
- Real-data drift report generated.

## 5. Drift monitoring

The latest recorded real-data drift report showed no major drift across the five monitored features, using a Population Stability Index (PSI) threshold of 0.2.

## 6. Deployment and repository links

- Dashboard: https://aiml-01-fraud-dashboard.onrender.com
- API: https://aiml-01-fraud-detection.onrender.com
- API documentation: https://aiml-01-fraud-detection.onrender.com/docs
- GitHub repository: https://github.com/vigneshwarlankala/Real-time-financial-fraud-detection

## 7. Verification scope and limitations

The API, prediction endpoint, dashboard, dashboard-to-API connection, and Kind deployment have been verified as described above.

Kafka streaming, production Feast online feature retrieval, complete fraud-ring detection integration, and the analyst case-review workflow should not be represented as fully end-to-end production-verified unless separate test evidence is available.

The reported model metrics are evaluation results and should not be interpreted as proof of production performance.

## 8. Completion status

The core fraud detection platform, API, dashboard, and Kind deployment are completed and verified. Supporting advanced components are included according to their implementation and available test evidence.

This report distinguishes implemented functionality from components with confirmed end-to-end verification.
