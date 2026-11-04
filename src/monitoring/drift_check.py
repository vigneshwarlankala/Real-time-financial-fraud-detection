import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(reference) < 2 or len(current) < 2:
        raise ValueError("Each dataset needs at least two valid values per feature.")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference and current transaction data for numeric feature drift."
    )
    parser.add_argument("--reference", required=True, help="Reference CSV path")
    parser.add_argument("--current", required=True, help="Current CSV path")
    parser.add_argument("--output", default="reports/drift_check.json")
    parser.add_argument("--rows", type=int, default=100000)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    current_path = Path(args.current)

    for path in (reference_path, current_path):
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

    reference = pd.read_csv(reference_path, nrows=args.rows)
    current = pd.read_csv(current_path, nrows=args.rows)

    common = [c for c in NUMERIC_FEATURES if c in reference.columns and c in current.columns]
    if not common:
        raise ValueError("No supported numeric transaction features are shared by both CSV files.")

    results = {}
    for feature in common:
        psi = calculate_psi(reference[feature], current[feature])
        results[feature] = {
            "psi": round(psi, 6),
            "status": "potential_drift" if psi >= 0.2 else "no_major_drift",
        }

    report = {
        "reference_file": str(reference_path),
        "current_file": str(current_path),
        "reference_rows_read": len(reference),
        "current_rows_read": len(current),
        "threshold": 0.2,
        "features": results,
        "overall_status": (
            "potential_drift"
            if any(item["status"] == "potential_drift" for item in results.values())
            else "no_major_drift"
        ),
        "note": "PSI is a screening signal, not proof of model degradation or a diagnosis of concept drift.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
