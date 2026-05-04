from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
from confluent_kafka import Producer

from src.config import DatasetConfig, Settings

LOGGER = logging.getLogger(__name__)


def _delivery_report(err: object, msg: object) -> None:
    if err is not None:
        raise RuntimeError(f"Kafka delivery failed: {err}")


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def produce_dataset(settings: Settings, dataset: DatasetConfig) -> int:
    LOGGER.info("Producing dataset %s from %s into topic %s", dataset.name, dataset.path, dataset.topic)
    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    total = 0
    remaining = settings.max_rows_per_dataset

    for chunk_number, chunk in enumerate(pd.read_csv(dataset.path, chunksize=settings.producer_batch_size), start=1):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        LOGGER.info(
            "Producing chunk %s for %s with %s rows",
            chunk_number,
            dataset.name,
            len(chunk),
        )

        for row in chunk.to_dict(orient="records"):
            payload = {
                "dataset": dataset.name,
                "payload": {key: _clean_value(value) for key, value in row.items()},
            }
            key = str(payload["payload"].get(dataset.id_column, ""))
            producer.produce(
                dataset.topic,
                key=key.encode("utf-8"),
                value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                callback=_delivery_report,
            )
            total += 1
            producer.poll(0)

        if remaining is not None:
            remaining -= len(chunk)
        LOGGER.info("Produced %s rows so far for %s", total, dataset.name)
        producer.flush()
        if settings.producer_batch_delay_seconds > 0:
            LOGGER.info(
                "Waiting %.2fs before next chunk for %s",
                settings.producer_batch_delay_seconds,
                dataset.name,
            )
            time.sleep(settings.producer_batch_delay_seconds)

    producer.flush()
    LOGGER.info("Finished producing %s rows for %s", total, dataset.name)
    return total


def produce_all(settings: Settings) -> dict[str, int]:
    counts = {dataset.name: produce_dataset(settings, dataset) for dataset in settings.datasets}
    LOGGER.info("All Kafka production completed: %s", counts)
    return counts
