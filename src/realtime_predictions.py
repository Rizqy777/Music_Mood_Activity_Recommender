from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCALER_STATS_ENV = "MOOD_SCALER_STATS_PATH"
SCALER_STATS_COLUMNS = {"feature", "mean", "std"}
SCALER_STATS_CANDIDATES = [
    ROOT / "data_lake" / "tmp_gold" / "mood_prepared" / "scaler_stats" / "scaler_stats.parquet",
    ROOT / "data_lake" / "tmp_gold" / "mood_prepared" / "scaler_stats" / "scaler_stats.csv",
    ROOT / "data_lake" / "s3_cache" / "gold" / "mood_prepared" / "scaler_stats" / "scaler_stats.parquet",
    ROOT / "data_lake" / "s3_cache" / "gold" / "mood_prepared" / "scaler_stats" / "scaler_stats.csv",
]

LABEL_NAMES = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm",
}

UNIT_INTERVAL_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

MODEL_FALLBACK_FEATURES = [
    *UNIT_INTERVAL_FEATURES,
    "loudness",
    "tempo",
    "duration_ms",
]

RECOMMENDER_FEATURES = [
    *UNIT_INTERVAL_FEATURES,
    "loudness",
    "tempo",
    "spec_rate",
    "duration_ms",
]

METADATA_DEFAULTS: dict[str, Any] = {
    "track_name": "Track sin nombre",
    "artists": "Artista desconocido",
    "album_name": "Album desconocido",
    "track_genre": "realtime_input",
    "popularity": 0,
    "explicit": False,
}

CATALOG_PATH = ROOT / "data_lake" / "recommender" / "classified_tracks.parquet"
CATALOG_CSV_PATH = ROOT / "data_lake" / "recommender" / "classified_tracks.csv"
HISTORY_PATH = ROOT / "data_lake" / "realtime_predictions" / "classified_inputs.csv"
SPOTIFY_TRACK_URL_PATTERN = re.compile(
    r"^https://open\.spotify\.com/(?:intl-[a-z]{2}/)?track/([A-Za-z0-9]{22})(?:[?/#].*)?$"
)


class RealtimeMoodClassifier:
    def __init__(
        self,
        model_path: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or ROOT / "models" / "mood_best_model.joblib"
        self.catalog_path = catalog_path or CATALOG_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo de clasificacion: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.feature_cols = self._resolve_feature_columns()
        self.scaler_stats = load_scaler_stats()
        self._has_internal_scaler = "scaler" in getattr(self.model, "named_steps", {})

    def classify(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = prepare_input_frame(frame, self.feature_cols)
        model_input = prepared
        if self.scaler_stats is not None and not self._has_internal_scaler:
            model_input = apply_scaler_stats(prepared, self.feature_cols, self.scaler_stats)
        proba = self.model.predict_proba(model_input[self.feature_cols])
        labels = self.model.predict(model_input[self.feature_cols])

        prepared["audio_predicted_mood_label"] = labels.astype(int)
        prepared["predicted_mood_label"] = labels.astype(int)
        prepared["audio_predicted_mood"] = [LABEL_NAMES[int(label)] for label in labels]
        prepared["predicted_mood"] = prepared["audio_predicted_mood"]
        for idx, label in enumerate(sorted(LABEL_NAMES)):
            mood = LABEL_NAMES[label]
            prepared[f"proba_{mood}"] = proba[:, idx]
            prepared[f"audio_proba_{mood}"] = proba[:, idx]
        prepared["mood_confidence"] = proba.max(axis=1)
        prepared["lyrics_predicted_mood"] = None
        prepared["mood_contrast"] = False
        prepared["source"] = "realtime_prediction"
        prepared["ingested_at"] = datetime.now(timezone.utc).isoformat()
        return prepared

    def classify_and_append(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        classified = self.classify(frame)
        self._append_to_catalog(classified)
        append_history(classified)
        return classified, {
            "rows_classified": int(len(classified)),
            "catalog_path": str(self.catalog_path),
            "history_path": str(HISTORY_PATH),
        }

    def _resolve_feature_columns(self) -> list[str]:
        feature_names = getattr(self.model, "feature_names_in_", None)
        if feature_names is not None:
            return list(feature_names)
        imputer = getattr(self.model, "named_steps", {}).get("imputer")
        imputer_features = getattr(imputer, "feature_names_in_", None)
        if imputer_features is not None:
            return list(imputer_features)
        return MODEL_FALLBACK_FEATURES

    def _append_to_catalog(self, classified: pd.DataFrame) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        
        if CATALOG_CSV_PATH.exists():
            try:
                # Leemos SOLO los encabezados del CSV existente (tarda microsegundos)
                existing_headers = list(pd.read_csv(CATALOG_CSV_PATH, nrows=0).columns)
                
                # Si falta alguna columna en el nuevo track, la rellenamos con NaN
                for col in existing_headers:
                    if col not in classified.columns:
                        classified[col] = np.nan
                
                # Forzamos a que el nuevo track tenga EXACTAMENTE el mismo orden de columnas
                classified_aligned = classified[existing_headers]
                
                # Hacemos el append seguro sin cabecera
                classified_aligned.to_csv(CATALOG_CSV_PATH, mode='a', header=False, index=False)
            except Exception:
                # Fallback por si el archivo estaba vacío o corrupto
                classified.to_csv(CATALOG_CSV_PATH, mode='a', header=False, index=False)
        else:
            # Si el archivo no existe, lo creamos de cero con sus cabeceras
            classified.to_csv(CATALOG_CSV_PATH, index=False)


def prepare_input_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("No hay filas para clasificar.")

    prepared = normalize_columns(frame.copy())
    prepared = add_engineered_audio_features(prepared)
    missing_required = [col for col in feature_cols if col not in prepared.columns]
    if missing_required:
        raise ValueError(
            "Faltan columnas obligatorias para el modelo: " + ", ".join(missing_required)
        )
    if "spotify_url" not in prepared.columns:
        raise ValueError("Falta la columna obligatoria spotify_url.")
    prepared["spotify_url"] = prepared["spotify_url"].fillna("").astype(str).str.strip()
    missing_url = prepared["spotify_url"] == ""
    if missing_url.any():
        bad_rows = [str(idx + 1) for idx in prepared.index[missing_url].tolist()[:10]]
        raise ValueError("spotify_url es obligatorio. Filas sin URL: " + ", ".join(bad_rows))
    prepared["track_id_from_url"] = prepared["spotify_url"].apply(extract_spotify_track_id)

    for col in feature_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    rows_with_missing = prepared[feature_cols].isna().any(axis=1)
    if rows_with_missing.any():
        bad_rows = [str(idx + 1) for idx in prepared.index[rows_with_missing].tolist()[:10]]
        raise ValueError(
            "Hay valores numericos vacios o invalidos en las filas: " + ", ".join(bad_rows)
        )

    validate_ranges(prepared)
    for col in RECOMMENDER_FEATURES:
        if col not in prepared.columns:
            prepared[col] = 0.0
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)

    if "track_id" not in prepared.columns:
        prepared["track_id"] = prepared["track_id_from_url"]
    prepared["track_id"] = prepared["track_id"].fillna("").astype(str).str.strip()
    prepared.loc[prepared["track_id"] == "", "track_id"] = prepared.loc[
        prepared["track_id"] == "", "track_id_from_url"
    ]

    for col, default in METADATA_DEFAULTS.items():
        if col not in prepared.columns:
            prepared[col] = default
        else:
            prepared[col] = prepared[col].fillna(default)

    prepared["popularity"] = pd.to_numeric(prepared["popularity"], errors="coerce").fillna(0).clip(0, 100)
    prepared["explicit"] = prepared["explicit"].astype(str).str.lower().isin({"true", "1", "yes", "si", "sí"})
    return prepared


def load_scaler_stats() -> pd.DataFrame | None:
    explicit = os.getenv(SCALER_STATS_ENV, "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(SCALER_STATS_CANDIDATES)
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            stats = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            stats = pd.read_csv(path)
        else:
            continue
        if not stats.empty and SCALER_STATS_COLUMNS.issubset(stats.columns):
            return stats
    return None


def apply_scaler_stats(
    frame: pd.DataFrame,
    feature_cols: list[str],
    scaler_stats: pd.DataFrame,
) -> pd.DataFrame:
    if scaler_stats is None or scaler_stats.empty:
        return frame
    if not SCALER_STATS_COLUMNS.issubset(scaler_stats.columns):
        return frame
    stats_map = (
        scaler_stats.set_index("feature")[["mean", "std"]]
        .to_dict(orient="index")
    )
    scaled = frame.copy()
    for col in feature_cols:
        if col not in stats_map:
            continue
        mean_val = float(stats_map[col]["mean"])
        std_val = float(stats_map[col]["std"] or 1.0)
        if std_val == 0:
            std_val = 1.0
        scaled[col] = (pd.to_numeric(scaled[col], errors="coerce") - mean_val) / std_val
    return scaled


def add_engineered_audio_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for col in ["energy", "valence", "tempo", "duration_ms", "acousticness", "instrumentalness"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    if {"energy", "valence"}.issubset(prepared.columns):
        prepared["energy_valence_interaction"] = prepared["energy"] * prepared["valence"]
        prepared["energy_squared"] = prepared["energy"] ** 2
        energy_bucket = pd.cut(
            prepared["energy"],
            bins=[-float("inf"), 0.33, 0.66, float("inf")],
            labels=["low", "mid", "high"],
        )
        for bucket in ["low", "mid", "high"]:
            prepared[f"energy_bucket_{bucket}"] = (energy_bucket == bucket).astype(float)
    if "valence" in prepared.columns:
        valence_bucket = pd.cut(
            prepared["valence"],
            bins=[-float("inf"), 0.33, 0.66, float("inf")],
            labels=["low", "mid", "high"],
        )
        for bucket in ["low", "mid", "high"]:
            prepared[f"valence_bucket_{bucket}"] = (valence_bucket == bucket).astype(float)
    if "tempo" in prepared.columns:
        prepared["tempo_log1p"] = np.log1p(prepared["tempo"].clip(lower=0))
    if "duration_ms" in prepared.columns:
        prepared["duration_sqrt"] = np.sqrt(prepared["duration_ms"].clip(lower=0))
    if {"acousticness", "instrumentalness"}.issubset(prepared.columns):
        prepared["acoustic_instrumental_mix"] = prepared["acousticness"] * prepared["instrumentalness"]
    return prepared


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
    aliases = {
        "duration_(ms)": "duration_ms",
        "duration": "duration_ms",
        "name": "track_name",
        "artist": "artists",
        "genre": "track_genre",
        "id": "track_id",
        "url": "spotify_url",
        "track_url": "spotify_url",
        "spotify_track_url": "spotify_url",
    }
    return frame.rename(columns={old: new for old, new in aliases.items() if old in frame.columns})


def extract_spotify_track_id(url: str) -> str:
    match = SPOTIFY_TRACK_URL_PATTERN.match(str(url).strip())
    if not match:
        raise ValueError(
            "spotify_url debe ser una URL valida de track de Spotify, por ejemplo "
            "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC"
        )
    return match.group(1)


def validate_ranges(frame: pd.DataFrame) -> None:
    problems = []
    for col in UNIT_INTERVAL_FEATURES:
        if col in frame.columns and (~frame[col].between(0.0, 1.0)).any():
            problems.append(f"{col} debe estar entre 0 y 1")
    if "tempo" in frame.columns and (frame["tempo"] < 0).any():
        problems.append("tempo debe ser mayor o igual que 0")
    if "duration_ms" in frame.columns and (frame["duration_ms"] <= 0).any():
        problems.append("duration_ms debe ser mayor que 0")
    if "loudness" in frame.columns and (~frame["loudness"].between(-80.0, 10.0)).any():
        problems.append("loudness debe estar entre -80 y 10")
    if problems:
        raise ValueError("; ".join(problems))


def read_uploaded_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError("Formato no soportado. Usa CSV o Parquet.")


def append_history(classified: pd.DataFrame) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Escribimos directo al final del archivo con mode='a'. 
    # Ni siquiera cargamos el CSV en memoria.
    write_header = not HISTORY_PATH.exists()
    classified.to_csv(HISTORY_PATH, mode='a', header=write_header, index=False)
