# Plan de lectura del proyecto (paso a paso)

Este plan indica en que orden leer archivos para entender el flujo completo, de principio a fin.

## 1. Vision general

- DESCRIPCION_PROYECTO_FIN_CURSO.MD
- PROJECT_SPEC.MD

## 2. Arranque del pipeline

- GUIA_ARRANQUE_PIPELINE.md
- README_PIPELINE.md

## 3. Codigo del pipeline

- scripts/run_pipeline.py
- src/config.py
- src/storage.py
- src/aws/s3_client.py
- src/mongo.py
- src/aws/rds_client.py
- src/kafka/producer.py
- src/kafka/consumer.py
- src/kafka/admin.py
- src/spark/session.py
- src/spark/transforms.py

## 4. Preparacion de datos

- docs/PREPARACION_DATOS.md
- notebooks/preparacion_datos.ipynb

## 5. Entrenamiento y validacion del clasificador emocional

- docs/MODELADO_ENTRENAMIENTO.md
- docs/VALIDACION_REPRESENTACION_MODELO.md
- scripts/train_models.py

## 6. Recomendador por emocion y actividad

- docs/RECOMENDADOR_EMOCION_ACTIVIDAD.md
- docs/AJUSTE_INTERPRETE_ACTIVIDAD.md
- scripts/train_activity_text_model.py
- src/activity_text_model.py
- scripts/build_recommender.py
- src/recommender.py

## 7. Interfaz

- app.py

## 8. Pruebas rapidas

- scripts/check_recommender_cases.py

Sugerencia: lee en ese orden y ejecuta cada script una sola vez para ver que archivos genera.
