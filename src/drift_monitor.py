
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_processed.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_drift_report.html"

# Load the processed transaction dataset.
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# Compare earlier and later transaction periods.
reference_data = df[df["step"] <= 500].copy()
current_data = df[df["step"] > 500].copy()

# Exclude the target label from feature-drift analysis.
reference_data = reference_data.drop(columns=["isFraud"], errors="ignore")
current_data = current_data.drop(columns=["isFraud"], errors="ignore")

print(f"Reference transactions: {len(reference_data):,}")
print(f"Current transactions: {len(current_data):,}")

if reference_data.empty or current_data.empty:
    raise ValueError("One of the selected time periods contains no transactions.")

# Generate the drift report.
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data,
)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
snapshot.save_html(str(REPORT_PATH))

print("\nTime-period drift report generated successfully.")
print(f"Report location: {REPORT_PATH}")
