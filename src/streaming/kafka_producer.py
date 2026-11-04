import json
import time
from kafka import KafkaProducer

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "fraud-transactions"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

def publish_transaction(transaction):
    """Publish one transaction to Kafka."""
    future = producer.send(TOPIC_NAME, value=transaction)
    metadata = future.get(timeout=10)

    print(
        f"Published transaction to {metadata.topic}, "
        f"partition {metadata.partition}, "
        f"offset {metadata.offset}"
    )

def main():
    transactions = [
        {
            "transaction_id": "TXN-1001",
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "M987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
        {
            "transaction_id": "TXN-1002",
            "step": 2,
            "type": "CASH_OUT",
            "amount": 5000.0,
            "nameOrig": "C555555555",
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C666666666",
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 6000.0,
        },
    ]

    try:
        for transaction in transactions:
            publish_transaction(transaction)
            time.sleep(1)
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    main()