from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ACTIVITY_PROFILE_COLS = [
    "activity_movement",
    "activity_energy",
    "activity_positivity",
    "activity_focus",
    "activity_calm",
    "activity_acoustic",
]

ACTIVITY_PROFILE_INDEX = {name: idx for idx, name in enumerate(ACTIVITY_PROFILE_COLS)}

ACTIVITY_PROTOTYPES = {
    "desahogo_emocional": [0.00, 0.05, 0.05, 0.30, 0.95, 0.85],
    "entrenamiento_intenso": [0.95, 0.95, 0.65, 0.45, 0.10, 0.15],
    "correr": [1.00, 0.95, 0.70, 0.35, 0.05, 0.10],
    "baile_fiesta": [0.95, 0.90, 0.95, 0.20, 0.05, 0.05],
    "limpieza_domestica": [0.75, 0.70, 0.55, 0.35, 0.15, 0.20],
    "estudio_trabajo": [0.15, 0.25, 0.45, 0.95, 0.70, 0.75],
    "alimentacion": [0.10, 0.20, 0.55, 0.50, 0.65, 0.55],
    "relajacion": [0.05, 0.15, 0.55, 0.55, 0.95, 0.85],
    "dormir": [0.00, 0.05, 0.35, 0.60, 1.00, 0.95],
    "caminar_paseo": [0.45, 0.45, 0.65, 0.45, 0.50, 0.45],
    "conducir_viaje": [0.45, 0.60, 0.75, 0.45, 0.35, 0.25],
    "actividad_general": [0.45, 0.50, 0.55, 0.45, 0.35, 0.30],
}

TRAINING_PHRASES = {
    "desahogo_emocional": [
        "quiero llorar",
        "necesito llorar",
        "quiero desahogarme",
        "estoy fatal y quiero estar solo",
        "quiero sacar la tristeza",
        "necesito musica para llorar",
        "quiero estar triste un rato",
        "me apetece una cancion triste",
        "quiero algo melancolico",
        "necesito procesar lo que siento",
    ],
    "entrenamiento_intenso": [
        "voy al gym",
        "voy a entrenar",
        "hacer pesas",
        "rutina de fuerza",
        "entrenamiento duro",
        "quiero darle fuerte",
        "hacer ejercicio intenso",
        "entrenar piernas",
        "hacer cardio fuerte",
        "levantar pesas",
    ],
    "correr": [
        "salir a correr",
        "voy a correr",
        "hacer running",
        "trote rapido",
        "preparar una carrera",
        "correr por el parque",
        "hacer sprint",
        "entrenamiento de carrera",
    ],
    "baile_fiesta": [
        "quiero bailar",
        "salir de fiesta",
        "bailar en casa",
        "montar una fiesta",
        "musica para moverme",
        "perrear",
        "hacer una fiesta con amigos",
        "bailar mientras cocino",
    ],
    "limpieza_domestica": [
        "limpiar la casa",
        "fregar el suelo",
        "pasar la fregona",
        "barrer",
        "limpiar el bano",
        "ordenar mi cuarto",
        "lavar los platos",
        "poner lavadoras",
        "hacer tareas de casa",
        "quitar el polvo",
        "limpiar la cocina",
        "recoger la habitacion",
    ],
    "estudio_trabajo": [
        "estudiar",
        "trabajar",
        "programar",
        "leer apuntes",
        "preparar un examen",
        "hacer deberes",
        "concentrarme",
        "trabajar en la oficina",
        "hacer un informe",
        "escribir codigo",
        "revisar documentacion",
        "leer un libro tecnico",
        "hacer una presentacion",
        "tengo reunion de trabajo",
        "resolver ejercicios",
    ],
    "alimentacion": [
        "voy a comer",
        "almorzar",
        "cenar",
        "desayunar",
        "merendar",
        "comer algo",
        "parar a comer",
        "comida rapida",
        "tomar un snack",
    ],
    "relajacion": [
        "relajarme",
        "meditar",
        "descansar",
        "hacer yoga suave",
        "tomar un cafe tranquilo",
        "leer en calma",
        "desconectar",
        "estar tranquilo",
        "respirar y bajar revoluciones",
        "reflexionar",
        "pensar un rato",
        "hacer introspeccion",
        "contemplar",
        "estar en silencio",
    ],
    "dormir": [
        "dormir",
        "irme a la cama",
        "echar una siesta",
        "conciliar el sueno",
        "relajarme para dormir",
        "musica para dormir",
        "bajar la ansiedad antes de dormir",
    ],
    "caminar_paseo": [
        "caminar",
        "dar un paseo",
        "pasear al aire libre",
        "andar por la ciudad",
        "caminar tranquilo",
        "pasear por el parque",
        "salir a andar",
    ],
    "conducir_viaje": [
        "conducir",
        "viajar en coche",
        "hacer un viaje largo",
        "ir en carretera",
        "manejar de noche",
        "conducir al trabajo",
        "road trip",
    ],
    "actividad_general": [
        "hacer algo",
        "no lo tengo claro",
        "actividad normal",
        "pasar el rato",
        "estar en casa",
        "hacer planes",
        "salir un rato",
    ],
}

KEYWORD_ACTIVITY_HINTS = {
    "llor": "desahogo_emocional",
    "desahog": "desahogo_emocional",
    "gym": "entrenamiento_intenso",
    "entren": "entrenamiento_intenso",
    "pesas": "entrenamiento_intenso",
    "correr": "correr",
    "running": "correr",
    "bail": "baile_fiesta",
    "fiesta": "baile_fiesta",
    "limpiar": "limpieza_domestica",
    "fregar": "limpieza_domestica",
    "barrer": "limpieza_domestica",
    "lavar": "limpieza_domestica",
    "ordenar": "limpieza_domestica",
    "cocinar": "limpieza_domestica",
    "estudi": "estudio_trabajo",
    "trabaj": "estudio_trabajo",
    "program": "estudio_trabajo",
    "leer": "estudio_trabajo",
    "reunion": "estudio_trabajo",
    "comer": "alimentacion",
    "almorz": "alimentacion",
    "cenar": "alimentacion",
    "desayun": "alimentacion",
    "merend": "alimentacion",
    "dorm": "dormir",
    "siest": "dormir",
    "relaj": "relajacion",
    "medit": "relajacion",
    "reflex": "relajacion",
    "pensar": "relajacion",
    "contempl": "relajacion",
    "introspec": "relajacion",
    "pase": "caminar_paseo",
    "caminar": "caminar_paseo",
    "conduc": "conducir_viaje",
    "viaj": "conducir_viaje",
}

KEYWORD_PROFILE_DELTAS = {
    "llor": {
        "activity_energy": -0.45,
        "activity_movement": -0.35,
        "activity_positivity": -0.35,
        "activity_calm": 0.45,
        "activity_acoustic": 0.35,
    },
    "desahog": {
        "activity_energy": -0.35,
        "activity_movement": -0.25,
        "activity_positivity": -0.25,
        "activity_calm": 0.40,
        "activity_acoustic": 0.30,
    },
    "gym": {"activity_energy": 0.45, "activity_movement": 0.40},
    "entren": {"activity_energy": 0.45, "activity_movement": 0.40},
    "pesas": {"activity_energy": 0.40, "activity_movement": 0.30},
    "correr": {"activity_energy": 0.55, "activity_movement": 0.55},
    "running": {"activity_energy": 0.55, "activity_movement": 0.55},
    "bail": {
        "activity_energy": 0.40,
        "activity_movement": 0.45,
        "activity_positivity": 0.35,
    },
    "fiesta": {
        "activity_energy": 0.35,
        "activity_movement": 0.40,
        "activity_positivity": 0.45,
    },
    "limpiar": {"activity_energy": 0.25, "activity_movement": 0.35},
    "fregar": {"activity_energy": 0.25, "activity_movement": 0.35},
    "barrer": {"activity_energy": 0.20, "activity_movement": 0.30},
    "lavar": {"activity_energy": 0.20, "activity_movement": 0.25},
    "ordenar": {"activity_energy": 0.20, "activity_movement": 0.25},
    "cocinar": {"activity_energy": 0.20, "activity_movement": 0.20},
    "estudi": {"activity_focus": 0.55, "activity_calm": 0.25, "activity_movement": -0.20},
    "trabaj": {"activity_focus": 0.45, "activity_calm": 0.20, "activity_movement": -0.15},
    "program": {"activity_focus": 0.55, "activity_calm": 0.20, "activity_movement": -0.20},
    "leer": {"activity_focus": 0.45, "activity_calm": 0.25, "activity_movement": -0.20},
    "reunion": {"activity_focus": 0.35, "activity_calm": 0.15, "activity_movement": -0.10},
    "comer": {"activity_energy": -0.20, "activity_movement": -0.20, "activity_calm": 0.30},
    "almorz": {"activity_energy": -0.15, "activity_movement": -0.15, "activity_calm": 0.25},
    "cenar": {"activity_energy": -0.20, "activity_movement": -0.20, "activity_calm": 0.30},
    "desayun": {"activity_energy": -0.15, "activity_movement": -0.15, "activity_calm": 0.20},
    "merend": {"activity_energy": -0.15, "activity_movement": -0.15, "activity_calm": 0.20},
    "dorm": {"activity_energy": -0.60, "activity_movement": -0.55, "activity_calm": 0.65},
    "siest": {"activity_energy": -0.55, "activity_movement": -0.50, "activity_calm": 0.60},
    "relaj": {"activity_energy": -0.35, "activity_movement": -0.25, "activity_calm": 0.45},
    "medit": {"activity_energy": -0.35, "activity_movement": -0.25, "activity_calm": 0.50},
    "reflex": {
        "activity_energy": -0.30,
        "activity_movement": -0.25,
        "activity_focus": 0.35,
        "activity_calm": 0.40,
        "activity_acoustic": 0.20,
    },
    "pensar": {
        "activity_energy": -0.25,
        "activity_movement": -0.20,
        "activity_focus": 0.30,
        "activity_calm": 0.35,
    },
    "contempl": {
        "activity_energy": -0.30,
        "activity_movement": -0.25,
        "activity_focus": 0.25,
        "activity_calm": 0.40,
    },
    "introspec": {
        "activity_energy": -0.35,
        "activity_movement": -0.30,
        "activity_focus": 0.35,
        "activity_calm": 0.45,
        "activity_acoustic": 0.20,
    },
    "pase": {"activity_energy": -0.05, "activity_movement": 0.20, "activity_calm": 0.20},
    "caminar": {"activity_energy": -0.05, "activity_movement": 0.20, "activity_calm": 0.20},
    "conduc": {"activity_focus": 0.20, "activity_movement": 0.20},
    "viaj": {"activity_focus": 0.15, "activity_movement": 0.20, "activity_positivity": 0.15},
}


class ActivityTextInterpreter:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or ROOT / "models" / "activity_text_interpreter.joblib"
        self.model: Any | None = joblib.load(self.model_path) if self.model_path.exists() else None

    def predict_profile(self, text: str) -> dict[str, float | str]:
        clean_text = normalize_text(text)
        if self.model is None:
            return fallback_profile(clean_text)

        if isinstance(self.model, dict):
            values = np.asarray(self.model["regressor"].predict([clean_text])[0], dtype=float).clip(0.0, 1.0)
            classifier = self.model["classifier"]
            activity_name = str(classifier.predict([clean_text])[0])
            confidence = classifier_confidence(classifier, clean_text)
            values = apply_keyword_rules(clean_text, values)
            hint_name = keyword_hint_activity(clean_text)
            if hint_name:
                activity_name = hint_name
            elif confidence < 0.45:
                activity_name = nearest_activity_name(values)
        else:
            values = np.asarray(self.model.predict([clean_text])[0], dtype=float).clip(0.0, 1.0)
            values = apply_keyword_rules(clean_text, values)
            activity_name = keyword_hint_activity(clean_text) or nearest_activity_name(values)
        profile = dict(zip(ACTIVITY_PROFILE_COLS, values.tolist()))
        profile["activity_name"] = activity_name
        return profile


def normalize_text(text: str) -> str:
    value = str(text).lower().strip()
    value = value.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    value = value.replace("ü", "u").replace("ñ", "n")
    return re.sub(r"\s+", " ", value)


def nearest_activity_name(values: np.ndarray) -> str:
    best_name = "actividad_general"
    best_distance = float("inf")
    for name, prototype in ACTIVITY_PROTOTYPES.items():
        distance = float(np.linalg.norm(values - np.asarray(prototype, dtype=float)))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name


def classifier_confidence(classifier: Any, text: str) -> float:
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba([text])[0]
        return float(np.max(proba))
    return 1.0


def keyword_hint_activity(text: str) -> str | None:
    for keyword, activity_name in KEYWORD_ACTIVITY_HINTS.items():
        if keyword in text:
            return activity_name
    return None


def apply_keyword_rules(text: str, values: np.ndarray) -> np.ndarray:
    adjusted = values.astype(float).copy()
    for keyword, deltas in KEYWORD_PROFILE_DELTAS.items():
        if keyword not in text:
            continue
        for col, delta in deltas.items():
            idx = ACTIVITY_PROFILE_INDEX[col]
            adjusted[idx] += float(delta)
    return np.clip(adjusted, 0.0, 1.0)


def fallback_profile(text: str) -> dict[str, float | str]:
    scores = np.asarray(ACTIVITY_PROTOTYPES["actividad_general"], dtype=float)
    scores = apply_keyword_rules(text, scores)
    hint_name = keyword_hint_activity(text)
    profile = dict(zip(ACTIVITY_PROFILE_COLS, scores.clip(0.0, 1.0).tolist()))
    profile["activity_name"] = hint_name or nearest_activity_name(scores)
    return profile
