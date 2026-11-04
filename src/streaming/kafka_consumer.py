
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from kafka import KafkaConsumer

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "fraud-transactions"
API_URL = "https://fraud-detection-api-er42.onrender.com/predict"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = (
    PROJECT_ROOT / "reports" / "streaming" / "prediction_results.csv"
)

API_FIELDS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CSV_FIELDS = [
    "timestamp_utc",
    "transaction_id",
    "amount",
    "transaction_type",
    "fraud_probability",
    "prediction",
    "risk_level",
    "processing_time_ms",
]


def prepare_payload(transaction):
    """Validate and prepare a transaction for the prediction API."""
    missing = [field for field in API_FIELDS if field not in transaction]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return {key: transaction[key] for key in API_FIELDS}


def predict_transaction(transaction, api_url=API_URL):
    """Send a transaction to the fraud prediction API."""
    response = requests.post(
        api_url,
        json=prepare_payload(transaction),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def save_result(row):
    """Append one prediction to the CSV results log."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="fraud-detection-api-consumer",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print("Kafka fraud consumer started.")
    print(f"Kafka server: {KAFKA_SERVER}")
    print(f"Topic: {TOPIC_NAME}")
    print(f"Prediction API: {API_URL}")
    print(f"Results CSV: {RESULTS_FILE}")
    print("Press Ctrl+C to stop.\n")

    try:
        for message in consumer:
            transaction = message.value
            transaction_id = transaction.get("transaction_id", "Unknown")
            started = time.perf_counter()

            print(f"Processing transaction: {transaction_id}")

            try:
                result = predict_transaction(transaction)
                processing_time_ms = round(
                    (time.perf_counter() - started) * 1000, 2
                )

                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "transaction_id": transaction_id,
                    "amount": transaction["amount"],
                    "transaction_type": transaction["type"],
                    "fraud_probability": result.get("fraud_probability"),
                    "prediction": result.get("prediction"),
                    "risk_level": result.get("risk_level"),
                    "processing_time_ms": processing_time_ms,
                }

                # Save results before committing the Kafka offset.
                save_result(row)

                print("Fraud probability:", row["fraud_probability"])
                print("Prediction:", row["prediction"])
                print("Risk level:", row["risk_level"])
                print("Processing time (ms):", processing_time_ms)

                consumer.commit()
                print("Result saved; Kafka offset committed.")

            except Exception as error:
                print(f"Processing failed: {error}")
                print("Stopping without committing the failed message.")
                break

            print("-" * 50)

    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")

    finally:
        consumer.close()
        print("Kafka consumer closed.")


if __name__ == "__main__":
    main()
