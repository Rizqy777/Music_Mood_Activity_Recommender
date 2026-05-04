from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.activity_text_model import ActivityTextInterpreter

ROOT = Path(__file__).resolve().parents[1]

LABEL_NAMES = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm",
}

MOOD_TO_LABEL = {value: key for key, value in LABEL_NAMES.items()}

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

ACTIVITY_PROFILE_COLS = [
    "activity_movement",
    "activity_energy",
    "activity_positivity",
    "activity_focus",
    "activity_calm",
    "activity_acoustic",
]

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

class MusicActivityRecommender:
    def __init__(
        self,
        model_path: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or ROOT / "models" / "activity_recommender_mlp.joblib"
        self.catalog_path = catalog_path or ROOT / "data_lake" / "recommender" / "classified_tracks.parquet"
        self.model = joblib.load(self.model_path)
        self.catalog = pd.read_parquet(self.catalog_path)
        self.activity_interpreter = ActivityTextInterpreter()

    def recommend(self, user_mood: str, activity_text: str, top_k: int = 5) -> pd.DataFrame:
        mood = normalize_mood(user_mood)
        activity_profile = self.activity_interpreter.predict_profile(activity_text)
        scored = self.catalog.copy()
        for known_mood in MOOD_TO_LABEL:
            scored[f"user_mood_{known_mood}"] = 1.0 if known_mood == mood else 0.0
        for col, value in activity_profile.items():
            scored[col] = value
        scored["recommendation_score_nn"] = self.model.predict(scored[MODEL_INPUT_COLS]).clip(0.0, 1.0)
        scored["context_score"] = contextual_score(scored, mood, activity_profile)
        mood_probability_col = f"proba_{mood}"
        scored["user_mood_probability"] = scored[mood_probability_col]
        popularity = scored["popularity"].fillna(0).clip(0, 100) / 100 if "popularity" in scored else 0.0
        scored["recommendation_score"] = (
            0.58 * scored["context_score"]
            + 0.27 * scored["user_mood_probability"]
            + 0.10 * scored["recommendation_score_nn"]
            + 0.05 * popularity
        ).clip(0.0, 1.0)
        exact_mood_matches = int(scored["predicted_mood"].eq(mood).sum())
        if exact_mood_matches >= top_k:
            scored.loc[scored["predicted_mood"] != mood, "recommendation_score"] *= 0.62
        scored["activity_interpreted_as"] = activity_profile["activity_name"]
        scored["target_mood"] = mood
        scored["spotify_url"] = scored["track_id"].apply(
            lambda track_id: f"https://open.spotify.com/intl-es/track/{track_id}"
        )
        scored["reason"] = build_reason(mood, activity_profile["activity_name"])

        columns = [
            "track_name",
            "artists",
            "track_genre",
            "target_mood",
            "predicted_mood",
            "mood_confidence",
            "popularity",
            "activity_interpreted_as",
            "recommendation_score",
            "reason",
            "spotify_url",
            "track_id",
        ]
        available = [col for col in columns if col in scored.columns]
        return scored.sort_values("recommendation_score", ascending=False).head(top_k)[available]


def normalize_mood(user_mood: str) -> str:
    text = str(user_mood).strip().lower()
    aliases = {
        "triste": "sad",
        "sad": "sad",
        "feliz": "happy",
        "alegre": "happy",
        "happy": "happy",
        "energico": "energetic",
        "energética": "energetic",
        "energetica": "energetic",
        "energetic": "energetic",
        "calmado": "calm",
        "tranquilo": "calm",
        "calm": "calm",
    }
    return aliases.get(text, "happy")


def contextual_score(
    frame: pd.DataFrame, user_mood: str, activity_profile: dict[str, float | str]
) -> pd.Series:
    mood_weights = desired_mood_weights(user_mood, activity_profile)
    mood_fit = sum(frame[f"proba_{mood}"] * weight for mood, weight in mood_weights.items())

    movement = float(activity_profile["activity_movement"])
    energy_need = float(activity_profile["activity_energy"])
    positivity_need = float(activity_profile["activity_positivity"])
    focus_need = float(activity_profile["activity_focus"])
    calm_need = float(activity_profile["activity_calm"])
    acoustic_need = float(activity_profile["activity_acoustic"])

    activity_fit = (
        energy_need * closeness(frame["energy"], 0.85)
        + movement * closeness(frame["danceability"], 0.80)
        + positivity_need * closeness(frame["valence"], 0.70)
        + focus_need * (1.0 - normalized_abs(frame["speechiness"]))
        + calm_need * closeness(frame["energy"], -0.70)
        + acoustic_need * closeness(frame["acousticness"], 0.85)
    ) / max(energy_need + movement + positivity_need + focus_need + calm_need + acoustic_need, 1e-6)

    popularity = frame["popularity"].fillna(0).clip(0, 100) / 100 if "popularity" in frame else 0.0
    return (0.70 * mood_fit + 0.25 * activity_fit + 0.05 * popularity).clip(0.0, 1.0)


def desired_mood_weights(user_mood: str, activity_profile: dict[str, float | str]) -> dict[str, float]:
    energy = float(activity_profile["activity_energy"])
    calm = float(activity_profile["activity_calm"])
    positivity = float(activity_profile["activity_positivity"])

    weights = {"sad": 0.04, "happy": 0.04, "energetic": 0.04, "calm": 0.04}
    weights[user_mood] = weights.get(user_mood, 0.0) + 0.78

    if user_mood == "sad" and calm > 0.70 and energy < 0.25:
        weights["sad"] += 0.45
        weights["calm"] += 0.25
        weights["happy"] *= 0.20
        weights["energetic"] *= 0.15
    elif user_mood == "sad" and energy > 0.65:
        weights["sad"] += 0.25
        weights["energetic"] += 0.22
        weights["happy"] *= 0.40
    elif calm > 0.65:
        weights["calm"] += 0.22
        weights["energetic"] *= 0.45
    elif energy > 0.65:
        weights["energetic"] += 0.22

    if positivity > 0.70 and user_mood == "happy":
        weights["happy"] += 0.20

    weights = {mood: max(value, 0.0) for mood, value in weights.items()}
    total = sum(weights.values())
    return {mood: value / total for mood, value in weights.items()}


def closeness(series: pd.Series, target: float) -> pd.Series:
    return (1.0 - (series - target).abs() / 4.0).clip(0.0, 1.0)


def normalized_abs(series: pd.Series) -> pd.Series:
    max_abs = max(float(series.abs().max()), 1.0)
    return (series.abs() / max_abs).clip(0.0, 1.0)


def build_reason(user_mood: str, activity_name: str) -> str:
    if user_mood == "sad" and activity_name == "desahogo_emocional":
        return "Prioriza canciones sad/calm para acompanar el estado sin forzar animo feliz."
    if user_mood == "sad" and activity_name in {"entrenamiento_intenso", "correr"}:
        return "Mantiene compatibilidad emocional, pero sube energia para la actividad."
    if activity_name == "limpieza_domestica":
        return "Interpreta la actividad como movimiento moderado y energia funcional."
    if activity_name == "alimentacion":
        return "Asume un momento de pausa: baja energia y un ritmo mas suave."
    if activity_name == "relajacion":
        return "Busca un ambiente calmado y con foco suave para reflexionar o desconectar."
    return "Equilibra emocion indicada, actividad y caracteristicas musicales."
