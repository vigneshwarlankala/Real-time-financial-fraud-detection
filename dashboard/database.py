
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "fraud_alerts.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fraud_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                case_status TEXT NOT NULL DEFAULT 'Open',
                analyst_feedback TEXT NOT NULL DEFAULT 'Pending',
                analyst_notes TEXT DEFAULT ''
            )
            """
        )


def save_alert(
    transaction_type,
    amount,
    fraud_probability,
    prediction,
    risk_level,
):
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO fraud_alerts (
                created_at,
                transaction_type,
                amount,
                fraud_probability,
                prediction,
                risk_level
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                transaction_type,
                amount,
                fraud_probability,
                prediction,
                risk_level,
            ),
        )
        return cursor.lastrowid


def get_alerts():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM fraud_alerts
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def update_alert(alert_id, case_status, analyst_feedback, analyst_notes):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE fraud_alerts
            SET case_status = ?,
                analyst_feedback = ?,
                analyst_notes = ?
            WHERE id = ?
            """,
            (case_status, analyst_feedback, analyst_notes, alert_id),
        )
        return cursor.rowcount > 0


def delete_alert(alert_id):
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM fraud_alerts WHERE id = ?",
            (alert_id,),
        )
        return cursor.rowcount > 0


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DATABASE_PATH}")
