from __future__ import annotations

from datetime import datetime, timezone

from pymongo import MongoClient

from src.config import Settings


def record_layer_metadata(settings: Settings, dataset: str, layer: str, path: str) -> None:
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
    try:
        collection = client[settings.mongo_database]["data_lake_layers"]
        collection.insert_one(
            {
                "dataset": dataset,
                "layer": layer,
                "path": path,
                "created_at": datetime.now(timezone.utc),
            }
        )
    finally:
        client.close()
