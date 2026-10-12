
import json
from datetime import datetime, timezone

from kafka import KafkaProducer

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "fraud-transactions"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

transaction = {
    "transaction_id": f"TXN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 181.0,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

try:
    result = producer.send(TOPIC_NAME, value=transaction)
    metadata = result.get(timeout=10)

    print("Transaction sent successfully!")
    print(f"Transaction ID: {transaction['transaction_id']}")
    print(f"Topic: {metadata.topic}")
    print(f"Partition: {metadata.partition}")
    print(f"Offset: {metadata.offset}")
    print(f"Transaction: {json.dumps(transaction, indent=4)}")

finally:
    producer.close()
