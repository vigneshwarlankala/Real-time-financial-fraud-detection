# AIML-01 — Real-Time Financial Fraud Detection Platform

**Last updated:** 4 November 2026

## Verified deliverables

- PaySim dataset prepared: 6,362,620 transactions.
- XGBoost fraud detection model trained and saved.
- Model evaluation report saved; PR-AUC: 0.9651.
- FastAPI prediction API deployed and health-checked.
- Live prediction endpoint tested successfully.
- Streamlit dashboard deployed and confirmed working.
- Dashboard-to-API connection verified by user.
- Kafka producer/consumer implementation available.
- MLflow experiment tracking configured.
- GNN model artifact and fraud-ring detection implementation available.
- Feast feature-view definitions and integration tests available.
- Drift-monitoring implementation smoke-tested.
- Real-data drift report generated.
- GitHub repository updated with deployment configuration.

## Deployment links

- Dashboard: https://aiml-01-fraud-dashboard.onrender.com
- API: https://aiml-01-fraud-detection.onrender.com
- API documentation: https://aiml-01-fraud-detection.onrender.com/docs
- GitHub: https://github.com/vigneshwarlankala/Real-time-financial-fraud-detection

## Model evaluation

- Precision: 0.34
- Recall: 0.99
- F1-score: 0.51
- PR-AUC: 0.9651

## Drift monitoring

The latest recorded real-data drift report showed no major drift across the five monitored features, using a PSI threshold of 0.2.

## Remaining verification

- Verify production Feast online feature retrieval.
- Verify fraud-ring detection and streaming end to end.
- Verify analyst case-review workflow.
- Review Kubernetes deployment configuration.
- Complete README, architecture documentation, and final demo evidence.

## Completion note

The API and dashboard deployments have been confirmed working. Other advanced components are recorded as implemented or tested where evidence is available; they should not be described as fully production-verified without further integration testing.
