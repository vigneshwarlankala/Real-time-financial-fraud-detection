
import json
import requests
from kafka import KafkaConsumer

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "fraud-transactions"
API_URL = "https://fraud-detection-api-er42.onrender.com/predict"

API_FIELDS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def prepare_payload(transaction):
    """Keep only the fields expected by the prediction API."""
    return {key: transaction[key] for key in API_FIELDS}


def predict_transaction(transaction, api_url=API_URL):
    """Send a transaction to the fraud detection API."""
    payload = prepare_payload(transaction)

    response = requests.post(
        api_url,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main():
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="fraud-detection-api-consumer",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print("Listening for transactions...")
    print("Sending transactions to the fraud detection API.")
    print("Press Ctrl+C to stop.\n")

    try:
        for message in consumer:
            transaction = message.value
            transaction_id = transaction.get("transaction_id", "Unknown")

            print(f"Transaction: {transaction_id}")

            try:
                result = predict_transaction(transaction)

                print(f"Fraud probability: {result.get('fraud_probability')}")
                print(f"Prediction: {result.get('prediction')}")
                print(f"Risk level: {result.get('risk_level')}")

            except requests.RequestException as error:
                print(f"API request failed: {error}")

            except KeyError as error:
                print(f"Transaction is missing a required field: {error}")

            print("-" * 50)

    except KeyboardInterrupt:
        print("\nConsumer stopped.")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
