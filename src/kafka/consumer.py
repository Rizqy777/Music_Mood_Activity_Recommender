from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any

from confluent_kafka import Consumer, KafkaError

from src.config import Settings
from src.storage import LakeWriter

LOGGER = logging.getLogger(__name__)


def consume_to_bronze(settings: Settings, expected_counts: dict[str, int] | None = None) -> dict[str, int]:
    writer = LakeWriter(settings)
    writer.prepare()
    topics = [dataset.topic for dataset in settings.datasets]
    topic_to_dataset = {dataset.topic: dataset.name for dataset in settings.datasets}
    LOGGER.info("Consuming topics into Bronze: %s", topics)
    if expected_counts:
        LOGGER.info("Expected event counts: %s", expected_counts)

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe(topics)

    counts: dict[str, int] = defaultdict(int)
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    deadline = time.monotonic() + settings.consumer_timeout_seconds

    try:
        while time.monotonic() < deadline:
            msg = consumer.poll(1.0)
            if msg is None:
                if expected_counts and _counts_met(counts, expected_counts):
                    break
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(msg.error())

            dataset_name = topic_to_dataset[msg.topic()]
            event = json.loads(msg.value().decode("utf-8"))
            buffers[dataset_name].append(event["payload"])
            counts[dataset_name] += 1

            if len(buffers[dataset_name]) >= settings.producer_batch_size:
                _flush_buffer(writer, dataset_name, buffers[dataset_name])
                LOGGER.info("Bronze count for %s reached %s rows", dataset_name, counts[dataset_name])
                buffers[dataset_name].clear()

            if expected_counts and _counts_met(counts, expected_counts):
                break

        for dataset_name, rows in buffers.items():
            if rows:
                _flush_buffer(writer, dataset_name, rows)
                LOGGER.info("Flushed final Bronze buffer for %s with %s rows", dataset_name, len(rows))
        consumer.commit(asynchronous=False)
        LOGGER.info("Committed Kafka consumer offsets")
    finally:
        consumer.close()
        LOGGER.info("Kafka consumer closed")

    LOGGER.info("Finished consuming Kafka to Bronze: %s", dict(counts))
    return dict(counts)


def _flush_buffer(writer: LakeWriter, dataset_name: str, rows: list[dict[str, Any]]) -> None:
    file_name = f"part-{int(time.time() * 1000)}.jsonl"
    path = writer.append_jsonl("bronze", dataset_name, rows, file_name)
    LOGGER.info("Wrote %s Bronze rows to %s", len(rows), path)


def _counts_met(counts: dict[str, int], expected_counts: dict[str, int]) -> bool:
    return all(counts.get(dataset, 0) >= expected for dataset, expected in expected_counts.items())
