from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activity_text_model import (
    ACTIVITY_PROFILE_COLS,
    ACTIVITY_PROTOTYPES,
    TRAINING_PHRASES,
    normalize_text,
)

RANDOM_STATE = 42


def main() -> None:
    rows = []
    for activity_name, phrases in TRAINING_PHRASES.items():
        for phrase in phrases:
            for text in expand_phrase(phrase):
                row = {
                    "text": normalize_text(text),
                    "activity_name": activity_name,
                }
                row.update(dict(zip(ACTIVITY_PROFILE_COLS, ACTIVITY_PROTOTYPES[activity_name])))
                rows.append(row)

    df = pd.DataFrame(rows)
    regressor = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
                    ]
                ),
            ),
            ("regressor", KNeighborsRegressor(n_neighbors=3, weights="distance", metric="cosine")),
        ]
    )
    classifier = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
                    ]
                ),
            ),
            ("classifier", KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine")),
        ]
    )
    regressor.fit(df["text"], df[ACTIVITY_PROFILE_COLS])
    classifier.fit(df["text"], df["activity_name"])
    model = {"regressor": regressor, "classifier": classifier}

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "activity_text_interpreter.joblib")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "model_type": "TFIDF word/char ngrams + KNeighbors regressor + KNeighbors classifier",
        "activity_profile_columns": ACTIVITY_PROFILE_COLS,
        "activity_prototypes": ACTIVITY_PROTOTYPES,
        "training_examples": int(len(df)),
        "note": (
            "Este modelo combina un interprete linguistico con reglas por palabras clave. "
            "Convierte texto libre en dimensiones musicales como energia, movimiento, foco y calma."
        ),
    }
    (models_dir / "activity_text_interpreter_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    df.to_csv(ROOT / "data_lake" / "recommender" / "activity_text_training_examples.csv", index=False)

    examples = [
        "quiero fregar el suelo",
        "estoy triste y voy a limpiar la cocina",
        "quiero llorar",
        "voy a programar un rato",
        "salgo a correr",
    ]
    normalized_examples = [normalize_text(example) for example in examples]
    predictions = regressor.predict(normalized_examples)
    labels = classifier.predict(normalized_examples)
    for example, values, label in zip(examples, predictions, labels):
        pretty = dict(zip(ACTIVITY_PROFILE_COLS, values.clip(0.0, 1.0).round(3).tolist()))
        print(example, label, pretty)


def expand_phrase(phrase: str) -> list[str]:
    normalized = normalize_text(phrase)
    variants = {
        normalized,
        f"{normalized} ahora",
        f"voy a {normalized}",
        f"quiero {normalized}",
        f"tengo que {normalized}",
        f"necesito {normalized}",
        f"me apetece {normalized}",
        f"estoy para {normalized}",
        f"me toca {normalized}",
        f"hora de {normalized}",
        f"plan de {normalized}",
        f"vamos a {normalized}",
    }
    return sorted(variants)


if __name__ == "__main__":
    main()
