from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import MusicActivityRecommender


def main() -> None:
    recommender = MusicActivityRecommender()
    cases = [
        ("triste", "voy a bailar", "sad", "baile_fiesta"),
        ("triste", "voy al gym", "sad", "entrenamiento_intenso"),
        ("triste", "quiero llorar", "sad", "desahogo_emocional"),
        ("triste", "quiero fregar el suelo", "sad", "limpieza_domestica"),
        ("feliz", "voy a comer", "happy", "alimentacion"),
        ("feliz", "voy a estudiar", "happy", "estudio_trabajo"),
        ("energico", "voy a estudiar", "energetic", "estudio_trabajo"),
        ("energico", "voy a correr", "energetic", "correr"),
        ("tranquilo", "quiero dormir", "calm", "dormir"),
        ("tranquilo", "quiero bailar", "calm", "baile_fiesta"),
        ("tranquilo", "quiero reflexionar", "calm", "relajacion"),
        ("tranquilo", "quiero", "calm", "actividad_general"),
    ]
    for mood, activity, expected_target, expected_activity in cases:
        result = recommender.recommend(mood, activity, 5)
        target_moods = set(result["target_mood"])
        interpreted = str(result.iloc[0]["activity_interpreted_as"])
        if target_moods != {expected_target}:
            raise AssertionError(f"{mood}+{activity}: target {target_moods}, esperado {expected_target}")
        if interpreted != expected_activity:
            raise AssertionError(f"{mood}+{activity}: actividad {interpreted}, esperada {expected_activity}")
        if expected_target != "calm":
            top_predicted = str(result.iloc[0]["predicted_mood"])
            if top_predicted != expected_target:
                raise AssertionError(
                    f"{mood}+{activity}: top predicted {top_predicted}, esperado {expected_target}"
                )
        print(f"OK: {mood} + {activity} -> {expected_target}, {expected_activity}")


if __name__ == "__main__":
    main()
