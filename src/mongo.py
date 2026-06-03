from __future__ import annotations

from datetime import datetime, timezone

from pymongo import MongoClient

from src.config import Settings


def record_layer_metadata(
    settings: Settings,
    dataset: str,
    layer: str,
    path: str,
    *,
    run_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
    try:
        file_stats = (metadata or {}).get("file_stats") if metadata else None
        if not file_stats:
            return
        collection = client[settings.mongo_database]["data_lake_files"]
        created_at = datetime.now(timezone.utc)
        documents = []
        for item in file_stats:
            document = {
                "run_id": run_id,
                "dataset": dataset,
                "layer": layer,
                "path": path,
                "created_at": created_at,
            }
            document.update(item)
            documents.append(document)
        if documents:
            collection.insert_many(documents)
    finally:
        client.close()
