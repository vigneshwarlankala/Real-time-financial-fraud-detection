import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parents[1] / "reports" / "analyst_cases.db"

VALID_STATUSES = {
    "OPEN", "IN_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"
}


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            fraud_probability REAL NOT NULL CHECK (
                fraud_probability >= 0 AND fraud_probability <= 1
            ),
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            analyst_notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def create_case(transaction_id, fraud_probability, risk_level="HIGH"):
    transaction_id = str(transaction_id).strip()
    if not transaction_id:
        raise ValueError("transaction_id cannot be empty")

    probability = float(fraud_probability)
    if not 0 <= probability <= 1:
        raise ValueError("fraud_probability must be between 0 and 1")

    now = datetime.now(timezone.utc).isoformat()

    with connect() as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO analyst_cases
            (transaction_id, fraud_probability, risk_level, status,
             analyst_notes, created_at, updated_at)
            VALUES (?, ?, ?, 'OPEN', '', ?, ?)
        """, (transaction_id, probability, str(risk_level).upper(), now, now))

        row = conn.execute(
            "SELECT * FROM analyst_cases WHERE transaction_id = ?",
            (transaction_id,)
        ).fetchone()

        return dict(row), cursor.rowcount == 1


def update_case(case_id, status, analyst_notes=""):
    status = str(status).upper().strip()
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status. Choose one of: {sorted(VALID_STATUSES)}")

    now = datetime.now(timezone.utc).isoformat()

    with connect() as conn:
        cursor = conn.execute("""
            UPDATE analyst_cases
            SET status = ?, analyst_notes = ?, updated_at = ?
            WHERE case_id = ?
        """, (status, str(analyst_notes), now, int(case_id)))

        if cursor.rowcount == 0:
            return None

        row = conn.execute(
            "SELECT * FROM analyst_cases WHERE case_id = ?",
            (int(case_id),)
        ).fetchone()

        return dict(row)


def list_cases(status=None, limit=100):
    limit = int(limit)
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")

    with connect() as conn:
        if status is None:
            rows = conn.execute(
                "SELECT * FROM analyst_cases ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        else:
            status = str(status).upper().strip()
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status. Choose one of: {sorted(VALID_STATUSES)}")

            rows = conn.execute(
                "SELECT * FROM analyst_cases WHERE status = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (status, limit)
            ).fetchall()

        return [dict(row) for row in rows]


if __name__ == "__main__":
    import json
    print(json.dumps(list_cases(limit=20), indent=2))
