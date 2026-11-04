# Real-Time Financial Fraud Detection Platform

An AI/ML internship project for identifying potentially fraudulent financial transactions.

## Live Demo
- Dashboard: https://aiml-01-fraud-dashboard.onrender.com
- API: https://aiml-01-fraud-detection.onrender.com
- API documentation: https://aiml-01-fraud-detection.onrender.com/docs

## Dataset
PaySim: 6,362,620 transactions, including 8,213 fraudulent transactions.

## Model
- Algorithm: XGBoost
- Precision: 0.34
- Recall: 0.99
- F1-score: 0.51
- PR-AUC: 0.9651

## Technology
Python, Pandas, NumPy, XGBoost, FastAPI, Streamlit, MLflow, Kafka, Feast, Docker and Render.

## Architecture
1. Transaction preprocessing and feature engineering.
2. XGBoost model training and evaluation.
3. FastAPI prediction service.
4. Streamlit analyst dashboard.
5. Supporting drift-monitoring, streaming and fraud-ring modules.

## Run Locally
Create the environment:
py -3.11 -m venv .venv

Activate it:
.\.venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Start the API:
python -m uvicorn api.main:app --reload

Start the dashboard in another terminal:
python -m streamlit run dashboard\app.py

## Repository Structure
- api/ — API application
- dashboard/ — Streamlit dashboard
- data/ — Dataset files
- models/ — Model artifacts
- reports/ — Metrics and monitoring reports
- src/ — ML source code
- tests/ — Automated tests

## Limitations
This is an educational demonstration, not a certified financial security system. A high-risk prediction does not prove fraud. Human review is required before real-world action.

## Project Status
The API and dashboard have been deployed and manually verified. See reports/project_progress.md for further project status and verification details.
