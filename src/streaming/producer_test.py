
import json
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

transaction = {
    "transaction_id": "TXN-001",
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 181.0,
}

try:
    metadata = producer.send(
        "fraud-transactions", transaction
    ).get(timeout=10)

    print("Transaction sent successfully!")
    print("Topic:", metadata.topic)
    print("Partition:", metadata.partition)
    print("Offset:", metadata.offset)
finally:
    producer.close()
