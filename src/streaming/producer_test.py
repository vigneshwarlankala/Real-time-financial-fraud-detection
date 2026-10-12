
import json
from kafka import KafkaProducer

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "fraud-transactions"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

transaction = {
    "transaction_id": "TXN-001",
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 181.0,
}

try:
    metadata = producer.send(
        TOPIC_NAME, transaction
    ).get(timeout=10)

    print("Transaction sent successfully!")
    print("Transaction ID:", transaction["transaction_id"])
    print("Topic:", metadata.topic)
    print("Partition:", metadata.partition)
    print("Offset:", metadata.offset)

except Exception as error:
    print("Failed to send transaction:", error)

finally:
    producer.close()
