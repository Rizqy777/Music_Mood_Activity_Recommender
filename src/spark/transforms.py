from __future__ import annotations

import logging
from pathlib import Path
import shutil

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import Settings
from src.storage import LakeWriter

LOGGER = logging.getLogger(__name__)


UNIT_INTERVAL_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

MOOD_AUDIO_FEATURES = [
    *UNIT_INTERVAL_FEATURES,
    "loudness",
    "tempo",
    "spec_rate",
]

TRACK_AUDIO_FEATURES = [
    *UNIT_INTERVAL_FEATURES,
    "loudness",
    "tempo",
]

MOOD_LABELS = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm",
}


def run_bronze_to_silver(settings: Settings, spark: SparkSession | None = None) -> dict[str, Path]:
    own_session = spark is None
    spark = spark or _build_session()
    writer = LakeWriter(settings)
    LOGGER.info("Starting Bronze to Silver transformations")
    outputs = {
        "mood_dataset": _write_silver(
            _clean_mood(_read_bronze(spark, settings, "mood_dataset")),
            writer,
            "mood_dataset",
        ),
        "tracks_dataset": _write_silver(
            _clean_tracks(_read_bronze(spark, settings, "tracks_dataset")),
            writer,
            "tracks_dataset",
        ),
    }
    if own_session:
        spark.stop()
    LOGGER.info("Bronze to Silver complete: %s", outputs)
    return outputs


def run_silver_to_gold(settings: Settings, spark: SparkSession | None = None) -> dict[str, Path]:
    own_session = spark is None
    spark = spark or _build_session()
    writer = LakeWriter(settings)
    LOGGER.info("Starting Silver to Gold transformations")
    outputs = {
        "mood_dataset": _write_gold(
            _prepare_gold_mood(_read_layer(spark, settings, "silver", "mood_dataset")),
            writer,
            "mood_dataset",
        ),
        "tracks_dataset": _write_gold(
            _prepare_gold_tracks(_read_layer(spark, settings, "silver", "tracks_dataset")),
            writer,
            "tracks_dataset",
        ),
    }
    if own_session:
        spark.stop()
    LOGGER.info("Silver to Gold complete: %s", outputs)
    return outputs


def _build_session() -> SparkSession:
    from src.spark.session import build_spark

    return build_spark()


def _read_bronze(spark: SparkSession, settings: Settings, dataset: str) -> DataFrame:
    files = _layer_files(settings.local_lake_dir / "bronze" / dataset, "*.jsonl")
    LOGGER.info("Reading Bronze dataset %s from %s files", dataset, len(files))
    return spark.read.json(files)


def _read_layer(spark: SparkSession, settings: Settings, layer: str, dataset: str) -> DataFrame:
    files = _layer_files(settings.local_lake_dir / layer / dataset, "*.parquet")
    LOGGER.info("Reading %s dataset %s from %s parquet files", layer, dataset, len(files))
    return spark.read.parquet(*files)


def _clean_mood(df: DataFrame) -> DataFrame:
    cleaned = (
        df.drop("Unnamed: 0", "Unnamed: 0.1")
        .withColumnRenamed("duration (ms)", "duration_ms")
        .withColumnRenamed("labels", "mood_label")
    )
    for column in MOOD_AUDIO_FEATURES:
        if column in cleaned.columns:
            cleaned = cleaned.withColumn(column, F.col(column).cast("double"))
    cleaned = cleaned.withColumn("duration_ms", F.col("duration_ms").cast("long"))
    cleaned = cleaned.withColumn("mood_label", F.col("mood_label").cast("int"))
    cleaned = _fill_nulls(cleaned, text_columns=["uri"], numeric_columns=["duration_ms", "mood_label", *MOOD_AUDIO_FEATURES])
    cleaned = cleaned.dropna(subset=["uri"]).dropDuplicates(["uri"])
    cleaned = _filter_audio_ranges(cleaned)
    return cleaned


def _clean_tracks(df: DataFrame) -> DataFrame:
    cleaned = df.drop("Unnamed: 0")
    numeric_columns = ["popularity", "duration_ms", "key", "mode", "time_signature", *TRACK_AUDIO_FEATURES]
    for column in numeric_columns:
        if column in cleaned.columns:
            target_type = "double" if column in TRACK_AUDIO_FEATURES else "int"
            cleaned = cleaned.withColumn(column, F.col(column).cast(target_type))
    cleaned = cleaned.withColumn("explicit", F.col("explicit").cast("boolean"))
    text_columns = ["track_id", "artists", "album_name", "track_name", "track_genre"]
    for column in text_columns:
        cleaned = cleaned.withColumn(column, F.trim(F.regexp_replace(F.col(column).cast("string"), r"\s+", " ")))
    cleaned = _fill_nulls(cleaned, text_columns=text_columns, numeric_columns=numeric_columns)
    cleaned = cleaned.fillna({"explicit": False})
    cleaned = cleaned.dropna(subset=["track_id"]).dropDuplicates(["track_id"])
    cleaned = _filter_audio_ranges(cleaned)
    return cleaned


def _prepare_gold_mood(df: DataFrame) -> DataFrame:
    label_map = F.create_map([item for pair in MOOD_LABELS.items() for item in (F.lit(pair[0]), F.lit(pair[1]))])
    selected = [
        "uri",
        "duration_ms",
        *MOOD_AUDIO_FEATURES,
        "mood_label",
    ]
    available = [column for column in selected if column in df.columns]
    return df.select(*available).withColumn("mood_name", label_map[F.col("mood_label")])


def _prepare_gold_tracks(df: DataFrame) -> DataFrame:
    base = df.withColumn(
        "context_tag",
        F.when((F.col("energy") >= 0.65) & (F.col("valence") >= 0.55), F.lit("active_positive"))
        .when((F.col("energy") < 0.45) & (F.col("valence") < 0.45), F.lit("calm_sad_candidate"))
        .when(F.col("acousticness") >= 0.65, F.lit("acoustic"))
        .otherwise(F.lit("general")),
    )
    return (
        base.withColumn("predicted_mood", F.lit(None).cast("string"))
        .withColumn(
            "recommendation_score",
            (
                F.coalesce(F.col("popularity"), F.lit(0)) / F.lit(100.0)
                + F.coalesce(F.col("valence"), F.lit(0.0))
                + F.coalesce(F.col("energy"), F.lit(0.0))
            )
            / F.lit(3.0),
        )
    )


def _fill_nulls(df: DataFrame, text_columns: list[str], numeric_columns: list[str]) -> DataFrame:
    text_defaults = {column: "unknown" for column in text_columns if column in df.columns}
    numeric_defaults = {column: 0 for column in numeric_columns if column in df.columns}
    return df.fillna({**text_defaults, **numeric_defaults})


def _filter_audio_ranges(df: DataFrame) -> DataFrame:
    for column in UNIT_INTERVAL_FEATURES:
        if column in df.columns:
            df = df.filter(F.col(column).between(0.0, 1.0))
    if "duration_ms" in df.columns:
        df = df.filter(F.col("duration_ms") > 0)
    if "tempo" in df.columns:
        df = df.filter(F.col("tempo") >= 0)
    if "loudness" in df.columns:
        df = df.filter(F.col("loudness").between(-80.0, 10.0))
    return df


def _write_silver(df: DataFrame, writer: LakeWriter, dataset: str) -> Path:
    return _write_layer(df, writer, "silver", dataset)


def _write_gold(df: DataFrame, writer: LakeWriter, dataset: str) -> Path:
    return _write_layer(df, writer, "gold", dataset)


def _write_layer(df: DataFrame, writer: LakeWriter, layer: str, dataset: str) -> Path:
    target = writer.local_dataset_dir(layer, dataset)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    row_count = df.count()
    LOGGER.info("Writing %s rows to %s layer for %s at %s", row_count, layer, dataset, target)
    df.write.mode("overwrite").parquet(target.as_uri())
    writer.mirror_directory_to_s3(layer, dataset)
    LOGGER.info("Finished writing %s/%s at %s", layer, dataset, target)
    return target


def _layer_files(path: Path, pattern: str) -> list[str]:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {path}")
    return [file.resolve().as_uri() for file in files]
