# Real-Time Financial Fraud Detection Platform

An AI/ML-based financial fraud detection platform that estimates the risk of fraudulent transactions using machine learning. The project includes a trained XGBoost model, a FastAPI prediction service, an interactive Streamlit dashboard, fraud-analysis components, monitoring utilities, and Kubernetes deployment configuration.

## Project Links

- **GitHub Repository:** https://github.com/vigneshwarlankala/Real-time-financial-fraud-detection
- **Live Dashboard:** https://aiml-01-fraud-dashboard.onrender.com
- **Live API:** https://aiml-01-fraud-detection.onrender.com
- **API Documentation:** https://aiml-01-fraud-detection.onrender.com/docs

---

## 1. Project Overview

Financial fraud detection is an important application of machine learning. This project uses transaction data to train a classification model that identifies potentially fraudulent financial activities.

The platform contains the following major components:

- **Machine Learning:** XGBoost-based fraud classification.
- **REST API:** FastAPI service for health checks and fraud predictions.
- **Dashboard:** Streamlit interface for transaction analysis.
- **Fraud Analysis:** Graph-based techniques for identifying potentially suspicious transaction relationships.
- **Monitoring:** Data drift checks using Population Stability Index (PSI) and Evidently AI.
- **Experiment Tracking:** MLflow configuration for tracking machine learning experiments.
- **Streaming Components:** Kafka producer and consumer implementation.
- **Feature Store:** Feast feature-store definitions.
- **Deployment:** Docker and Kubernetes configuration, including deployment using Kind.

The project is developed as an AI/ML internship project and demonstrates important stages of a financial fraud detection workflow.

## 2. System Architecture

The following diagram presents the main fraud detection workflow:

```text
             PaySim Transaction Data
                       |
                       v
               Data Preparation
                       |
                       v
                XGBoost Model
                       |
                       v
                 Fraud Scoring
                       |
                +------+------+
                |             |
                v             v
             FastAPI       Streamlit
               API         Dashboard
                |             |
                +------+------+
                       |
                       v
                 Analyst Review


          Supporting Components
          ---------------------

          Kafka Producer / Consumer
          Feast Feature Store
          GNN Fraud-Ring Analysis
          MLflow Experiment Tracking
          PSI / Evidently Monitoring
          Automated Retraining Trigger
          Docker / Kubernetes (Kind)