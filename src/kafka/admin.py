from __future__ import annotations

import time
import logging

from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from src.config import Settings

LOGGER = logging.getLogger(__name__)


def ensure_topics(settings: Settings) -> None:
    LOGGER.info("Connecting to Kafka admin at %s", settings.kafka_bootstrap_servers)
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    existing = _wait_for_metadata(admin)
    LOGGER.info("Kafka metadata available. Existing topics: %s", sorted(existing))
    topics = [
        NewTopic(dataset.topic, num_partitions=1, replication_factor=1)
        for dataset in settings.datasets
        if dataset.topic not in existing
    ]
    if not topics:
        LOGGER.info("Kafka topics already exist")
        return
    LOGGER.info("Creating Kafka topics: %s", [topic.topic for topic in topics])
    futures = admin.create_topics(topics)
    for topic_name, future in futures.items():
        future.result(timeout=20)
        LOGGER.info("Kafka topic ready: %s", topic_name)


def _wait_for_metadata(admin: AdminClient, timeout_seconds: int = 90) -> set[str]:
    deadline = time.monotonic() + timeout_seconds
    last_error: KafkaException | None = None
    while time.monotonic() < deadline:
        try:
            return set(admin.list_topics(timeout=10).topics.keys())
        except KafkaException as exc:
            last_error = exc
            LOGGER.info("Kafka metadata not ready yet: %s", exc)
            time.sleep(3)
    raise TimeoutError(f"Kafka metadata unavailable after {timeout_seconds}s: {last_error}")
