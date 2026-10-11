\# Real-Time Financial Fraud Detection Platform



An AI/ML project that detects potentially fraudulent financial transactions using machine learning and provides a REST API and interactive dashboard for transaction risk assessment.



\## Project Overview



This project uses the PaySim synthetic financial transaction dataset to train an XGBoost classification model. The model estimates the probability that a transaction is fraudulent and classifies its risk level.



\*\*Important:\*\* This is an educational project. Predictions are model estimates and must not be treated as confirmed fraud.



\## Features



\- Transaction data preprocessing and feature engineering

\- Analysis of severe class imbalance

\- XGBoost fraud classification

\- Imbalanced classification using class weights

\- Model evaluation using Precision, Recall, PR-AUC, and a confusion matrix

\- MLflow experiment tracking

\- FastAPI prediction service

\- Interactive Streamlit dashboard

\- Automated API tests



\## Technology Stack



\- Python 3.11

\- Pandas and NumPy

\- Scikit-learn

\- XGBoost

\- MLflow

\- FastAPI and Uvicorn

\- Streamlit and Plotly

\- Pytest



\## Dataset



The project uses the PaySim synthetic mobile-money transaction dataset.



The original dataset contains approximately 6.36 million transactions. The target column is `isFraud`, where `1` represents a fraudulent transaction and `0` represents a legitimate transaction.



The dataset is not included in this repository by default because of its size. Obtain it from the original dataset source and place the CSV file at:



`data/raw/transactions.csv`



\## Project Structure



```text

AIML-01-Fraud-Detection/

├── api/

│   └── main.py

├── dashboard/

│   └── app.py

├── data/

│   ├── raw/

│   └── processed/

├── models/

│   └── fraud\_xgboost.joblib

├── reports/

│   └── model\_metrics.json

├── src/

│   ├── prepare\_data.py

│   └── train\_model.py

├── tests/

│   └── test\_api.py

├── README.md

└── .gitignore

```



\## Setup Instructions



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd AIML-01-Fraud-Detection

```



Replace `YOUR\_GITHUB\_REPOSITORY\_URL` with the actual repository URL.



\### 2. Create a virtual environment



Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



Install the required packages:



```powershell

python -m pip install pandas numpy scikit-learn xgboost joblib fastapi "uvicorn\[standard]" streamlit plotly pytest requests mlflow httpx2

```



\### 4. Prepare the dataset



After placing the dataset at `data/raw/transactions.csv`, run:



```powershell

python src/prepare\_data.py

```



\### 5. Train the model



```powershell

python src/train\_model.py

```



The trained model and evaluation metrics are saved in the `models/` and `reports/` directories.



\## Run the Application



\### Start the API



Open a terminal in the project root and run:



```powershell

python -m uvicorn api.main:app --reload

```



API base URL: `http://127.0.0.1:8000`



Interactive API documentation: `http://127.0.0.1:8000/docs`



Health endpoint: `http://127.0.0.1:8000/health`



\### Start the dashboard



Open another terminal, activate the virtual environment, and run:



```powershell

.\\.venv\\Scripts\\Activate.ps1

python -m streamlit run dashboard/app.py

```



The dashboard will be available at `http://localhost:8501`.



\### Run automated tests



```powershell

python -m pytest tests/test\_api.py -v

```



\## Model Evaluation



The initial evaluation on a stratified random holdout set produced these results:



| Metric | Result |

|---|---:|

| Average Precision (PR-AUC) | 0.9651 |

| Fraud precision | 0.3414 |

| Fraud recall | 0.9915 |

| True negatives | 1,267,738 |

| False positives | 3,143 |

| False negatives | 14 |

| True positives | 1,629 |



These results are from the initial experimental setup. The dataset is highly imbalanced, so accuracy alone is not an appropriate evaluation measure.



The random split and engineered balance-change features may produce optimistic results. Further validation, threshold tuning, leakage analysis, and testing on unseen data are needed before drawing conclusions about real-world performance.



\## API Example



Send a POST request to `/predict` with a JSON transaction:



```json

{

&#x20; "step": 1,

&#x20; "amount": 181,

&#x20; "oldbalanceOrg": 181,

&#x20; "newbalanceOrig": 0,

&#x20; "oldbalanceDest": 0,

&#x20; "newbalanceDest": 0,

&#x20; "type": "TRANSFER"

}

```



The response includes the estimated fraud probability, binary prediction, risk level, and classification threshold.



\## Experiment Tracking



MLflow is used to record model parameters, evaluation metrics, and experiment runs.



To open the local MLflow interface, run:



```powershell

python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

```



Then visit `http://127.0.0.1:5000`.



\## Limitations and Future Improvements



\- Add streaming transaction ingestion using Apache Kafka.

\- Introduce feature management with Feast.

\- Explore graph-based fraud detection with PyTorch Geometric.

\- Add drift monitoring using Evidently AI.

\- Implement feedback-driven retraining workflows.

\- Improve threshold selection and evaluate temporal generalization.

\- Add containerization and Kubernetes deployment.

\- Strengthen monitoring, security, and production testing.



These advanced components are planned improvements and should not be considered implemented until they are added and tested.



\## Disclaimer



This project is for educational and portfolio purposes only. Predictions are estimates based on synthetic data and should not be used as the sole basis for real financial decisions.

