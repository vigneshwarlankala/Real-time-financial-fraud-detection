"""
Automatic retraining trigger for the fraud-detection model.

Workflow
--------
1. Run the PSI drift check (drift_check.py) on reference vs current data.
2. Read the resulting JSON report.
3. If overall_status == "potential_drift", launch train_model.py and log the
   decision to reports/retrain_log.json.
4. Exit with code 0 on success, 1 if retraining itself fails.

Usage
-----
    python src/retrain_trigger.py \\
        --reference data/processed/transactions_processed.csv \\
        --current   data/processed/transactions_processed.csv \\
        --rows      50000

The --reference and --current paths would normally point to different batches.
Using the same file is acceptable for a smoke-test.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIFT_CHECK_SCRIPT = ROOT / "src" / "monitoring" / "drift_check.py"
TRAIN_SCRIPT = ROOT / "src" / "train_model.py"
DRIFT_REPORT = ROOT / "reports" / "drift_check_trigger.json"
RETRAIN_LOG = ROOT / "reports" / "retrain_log.json"


def run_drift_check(reference: str, current: str, rows: int) -> dict:
    """Run drift_check.py and return the parsed JSON report."""
    cmd = [
        sys.executable,
        str(DRIFT_CHECK_SCRIPT),
        "--reference", reference,
        "--current", current,
        "--output", str(DRIFT_REPORT),
        "--rows", str(rows),
    ]
    print("Running drift check …")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Drift check failed:")
        print(result.stderr)
        sys.exit(1)

    return json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))


def trigger_retraining() -> bool:
    """Launch train_model.py and return True if it succeeds."""
    print("Drift detected — triggering model retraining …")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("Retraining completed successfully.")
        return True

    print("Retraining failed:")
    print(result.stderr[-2000:])   # last 2 000 chars to keep output readable
    return False


def save_log(drift_report: dict, retrained: bool, retrain_succeeded: bool | None) -> None:
    """Append a decision record to reports/retrain_log.json."""
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    if RETRAIN_LOG.exists():
        try:
            history = json.loads(RETRAIN_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_drift_status": drift_report.get("overall_status"),
        "retraining_triggered": retrained,
        "retraining_succeeded": retrain_succeeded,
        "feature_psi": {
            feat: vals["psi"]
            for feat, vals in drift_report.get("features", {}).items()
        },
    }
    history.append(entry)
    RETRAIN_LOG.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Decision logged to: {RETRAIN_LOG}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for data drift and auto-trigger model retraining if needed."
    )
    parser.add_argument("--reference", required=True, help="Reference CSV path")
    parser.add_argument("--current",   required=True, help="Current CSV path")
    parser.add_argument("--rows", type=int, default=50_000,
                        help="Max rows to read from each CSV (default: 50 000)")
    args = parser.parse_args()

    # Step 1 — Drift check
    drift_report = run_drift_check(args.reference, args.current, args.rows)
    overall = drift_report.get("overall_status", "no_major_drift")
    print(f"Drift status: {overall}")

    # Step 2 — Decision
    if overall == "potential_drift":
        retrain_ok = trigger_retraining()
        save_log(drift_report, retrained=True, retrain_succeeded=retrain_ok)
        if not retrain_ok:
            sys.exit(1)
    else:
        print("No significant drift detected — retraining not required.")
        save_log(drift_report, retrained=False, retrain_succeeded=None)

    print("Done.")


if __name__ == "__main__":
    main()
