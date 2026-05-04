from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("build_recommender")

RANDOM_STATE = 42

FEATURE_COLS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
    "spec_rate",
    "duration_ms",
]

LABEL_NAMES = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm",
}

MOOD_TO_LABEL = {value: key for key, value in LABEL_NAMES.items()}

ACTIVITY_PROFILE_COLS = [
    "activity_movement",
    "activity_energy",
    "activity_positivity",
    "activity_focus",
    "activity_calm",
    "activity_acoustic",
]

KNOWN_ACTIVITIES = {
    "llorar": [0.00, 0.05, 0.05, 0.30, 0.95, 0.85],
    "desahogarme": [0.00, 0.10, 0.10, 0.35, 0.90, 0.80],
    "estar triste": [0.05, 0.10, 0.10, 0.35, 0.90, 0.80],
    "gym": [0.95, 0.95, 0.65, 0.45, 0.10, 0.15],
    "entrenar": [0.95, 0.95, 0.65, 0.45, 0.10, 0.15],
    "correr": [1.00, 0.95, 0.70, 0.35, 0.05, 0.10],
    "bailar": [0.95, 0.90, 0.95, 0.20, 0.05, 0.05],
    "limpiar": [0.75, 0.70, 0.80, 0.35, 0.15, 0.20],
    "estudiar": [0.15, 0.25, 0.45, 0.95, 0.70, 0.75],
    "trabajar": [0.20, 0.35, 0.55, 0.90, 0.55, 0.55],
    "relajarse": [0.05, 0.15, 0.55, 0.55, 0.95, 0.85],
    "dormir": [0.00, 0.05, 0.35, 0.60, 1.00, 0.95],
    "caminar": [0.45, 0.45, 0.65, 0.45, 0.50, 0.45],
    "conducir": [0.45, 0.60, 0.75, 0.45, 0.35, 0.25],
}

ACTIVITY_KEYWORDS = {
    "activity_movement": ["gym", "entren", "correr", "run", "bail", "limpiar", "caminar", "deporte"],
    "activity_energy": ["gym", "entren", "correr", "run", "bail", "fiesta", "deporte"],
    "activity_positivity": ["bail", "fiesta", "limpiar", "paseo", "caminar", "animar"],
    "activity_focus": ["estudi", "trabaj", "leer", "program", "concentr", "oficina"],
    "activity_calm": ["relaj", "dormir", "meditar", "descans", "leer", "calma"],
    "activity_acoustic": ["relaj", "dormir", "estudi", "leer", "meditar", "acust"],
}

MODEL_INPUT_COLS = [
    *FEATURE_COLS,
    "proba_sad",
    "proba_happy",
    "proba_energetic",
    "proba_calm",
    "user_mood_sad",
    "user_mood_happy",
    "user_mood_energetic",
    "user_mood_calm",
    *ACTIVITY_PROFILE_COLS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica el catalogo de tracks y entrena un recomendador neuronal."
    )
    parser.add_argument(
        "--mood-model",
        type=Path,
        default=ROOT / "models" / "mood_best_model.joblib",
        help="Modelo emocional entrenado en el punto 5-7.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models",
        help="Carpeta donde guardar el recomendador.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    mood_model = joblib.load(args.mood_model)
    tracks = load_tracks_catalog()
    classified_tracks = classify_tracks(tracks, mood_model)

    output_data_dir = ROOT / "data_lake" / "recommender"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    classified_tracks.to_parquet(output_data_dir / "classified_tracks.parquet", index=False)
    classified_tracks.to_csv(output_data_dir / "classified_tracks.csv", index=False)

    train_df = build_weak_supervision_dataset(classified_tracks)
    X = train_df[MODEL_INPUT_COLS]
    y = train_df["target_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    recommender = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    recommender.fit(X_train, y_train)
    predictions = recommender.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(recommender, args.output_dir / "activity_recommender_mlp.joblib")
    train_df.sample(min(5000, len(train_df)), random_state=RANDOM_STATE).to_csv(
        output_data_dir / "recommender_training_sample.csv",
        index=False,
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "model_type": "MLPRegressor",
        "input_columns": MODEL_INPUT_COLS,
        "activity_profile_columns": ACTIVITY_PROFILE_COLS,
        "known_activities": KNOWN_ACTIVITIES,
        "tracks_rows": int(len(classified_tracks)),
        "training_rows": int(len(train_df)),
        "metrics": metrics,
        "note": (
            "El recomendador se entrena con weak supervision: las etiquetas de actividad "
            "se generan mediante reglas musicales interpretables porque no existe feedback "
            "real de usuarios en esta fase."
        ),
    }
    (args.output_dir / "activity_recommender_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Catalogo clasificado:", output_data_dir / "classified_tracks.parquet")
    print("Recomendador:", args.output_dir / "activity_recommender_mlp.joblib")
    print("Metricas:", json.dumps(metrics, indent=2))


def load_tracks_catalog() -> pd.DataFrame:
    scaled_path = ROOT / "data_lake" / "tmp_gold" / "tracks_prepared" / "full"
    metadata_path = ROOT / "data_lake" / "gold" / "tracks_dataset"
    if not scaled_path.exists():
        raise FileNotFoundError(f"No existe {scaled_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe {metadata_path}")

    scaled = pd.read_parquet(scaled_path)
    metadata = pd.read_parquet(metadata_path)
    metadata_cols = [
        col
        for col in [
            "track_id",
            "track_name",
            "artists",
            "album_name",
            "track_genre",
            "popularity",
            "explicit",
        ]
        if col in metadata.columns
    ]
    return scaled.merge(metadata[metadata_cols].drop_duplicates("track_id"), on="track_id", how="left")


def classify_tracks(tracks: pd.DataFrame, mood_model: Pipeline) -> pd.DataFrame:
    classified = tracks.copy()
    expected_features = list(
        getattr(mood_model.named_steps.get("imputer"), "feature_names_in_", FEATURE_COLS)
    )
    for col in expected_features:
        if col not in classified.columns:
            LOGGER.warning("La feature %s no existe en tracks; se rellena con 0.", col)
            classified[col] = 0.0
    for col in FEATURE_COLS:
        if col not in classified.columns:
            classified[col] = 0.0
    proba = mood_model.predict_proba(classified[expected_features])
    predicted_labels = mood_model.predict(classified[expected_features])
    classified["predicted_mood_label"] = predicted_labels.astype(int)
    classified["predicted_mood"] = [LABEL_NAMES[int(label)] for label in predicted_labels]
    for idx, label in enumerate(sorted(LABEL_NAMES)):
        classified[f"proba_{LABEL_NAMES[label]}"] = proba[:, idx]
    classified["mood_confidence"] = proba.max(axis=1)
    return classified


def build_weak_supervision_dataset(classified_tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mood_name in MOOD_TO_LABEL:
        for activity_name, activity_values in KNOWN_ACTIVITIES.items():
            activity_profile = dict(zip(ACTIVITY_PROFILE_COLS, activity_values))
            frame = classified_tracks.copy()
            for mood in MOOD_TO_LABEL:
                frame[f"user_mood_{mood}"] = 1.0 if mood == mood_name else 0.0
            for col, value in activity_profile.items():
                frame[col] = value
            frame["activity"] = activity_name
            frame["user_mood"] = mood_name
            frame["target_score"] = calculate_target_score(frame, mood_name, activity_profile)
            rows.append(frame[[*MODEL_INPUT_COLS, "activity", "user_mood", "target_score"]])
    return pd.concat(rows, ignore_index=True)


def calculate_target_score(
    frame: pd.DataFrame, user_mood: str, activity_profile: dict[str, float]
) -> pd.Series:
    mood_weights = desired_mood_weights(user_mood, activity_profile)
    mood_fit = sum(frame[f"proba_{mood}"] * weight for mood, weight in mood_weights.items())

    movement = activity_profile["activity_movement"]
    energy_need = activity_profile["activity_energy"]
    positivity_need = activity_profile["activity_positivity"]
    focus_need = activity_profile["activity_focus"]
    calm_need = activity_profile["activity_calm"]
    acoustic_need = activity_profile["activity_acoustic"]

    activity_fit = (
        energy_need * normalized_closeness(frame["energy"], 0.85)
        + movement * normalized_closeness(frame["danceability"], 0.80)
        + positivity_need * normalized_closeness(frame["valence"], 0.75)
        + focus_need * (1.0 - normalized_abs(frame["speechiness"]))
        + calm_need * normalized_closeness(frame["energy"], -0.55)
        + acoustic_need * normalized_closeness(frame["acousticness"], 0.85)
    ) / max(energy_need + movement + positivity_need + focus_need + calm_need + acoustic_need, 1e-6)

    popularity = frame["popularity"].fillna(0).clip(0, 100) / 100 if "popularity" in frame else 0.0
    confidence = frame["mood_confidence"].fillna(0)

    score = 0.50 * mood_fit + 0.35 * activity_fit + 0.10 * popularity + 0.05 * confidence
    return score.clip(0.0, 1.0)


def desired_mood_weights(user_mood: str, activity_profile: dict[str, float]) -> dict[str, float]:
    energy = activity_profile["activity_energy"]
    calm = activity_profile["activity_calm"]
    positivity = activity_profile["activity_positivity"]

    weights = {"sad": 0.05, "happy": 0.25, "energetic": 0.25, "calm": 0.20}
    weights[user_mood] = weights.get(user_mood, 0.0) + 0.35

    if energy > 0.65:
        weights["energetic"] += 0.35
        weights["happy"] += 0.20
        weights["sad"] += 0.10 if user_mood == "sad" else 0.0
        weights["calm"] -= 0.10
    if calm > 0.65:
        weights["calm"] += 0.35
        weights["energetic"] -= 0.15
    if positivity > 0.70:
        weights["happy"] += 0.25
        weights["sad"] -= 0.05

    weights = {mood: max(value, 0.0) for mood, value in weights.items()}
    total = sum(weights.values())
    return {mood: value / total for mood, value in weights.items()}


def normalized_closeness(series: pd.Series, target: float) -> pd.Series:
    return (1.0 - (series - target).abs() / 4.0).clip(0.0, 1.0)


def normalized_abs(series: pd.Series) -> pd.Series:
    max_abs = max(float(series.abs().max()), 1.0)
    return (series.abs() / max_abs).clip(0.0, 1.0)


if __name__ == "__main__":
    main()
