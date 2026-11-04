import json
import subprocess
import sys
from pathlib import Path


def test_drift_check_detects_distribution_shift(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    reference = project_root / "tests" / "drift_reference_smoke.csv"
    current = project_root / "tests" / "drift_current_smoke.csv"
    report = tmp_path / "drift_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "src" / "monitoring" / "drift_check.py"),
            "--reference", str(reference),
            "--current", str(current),
            "--output", str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert report.exists()

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["overall_status"] == "potential_drift"
    assert data["features"]["amount"]["psi"] >= 0.2
