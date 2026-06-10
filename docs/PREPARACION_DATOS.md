# Preparacion del conjunto de datos

Este documento resume de forma concreta los cambios aplicados al dataset de mood y por que se aplicaron. Todo el flujo se hace con Spark y parte de Silver en S3.

## 1. Carga desde Silver (S3) con Spark
- Se descarga `s3://<bucket>/silver/mood_dataset/` a una carpeta temporal local.
- Spark lee el parquet local para procesar el dataset.
- Motivo: usar Silver real en S3 y mantener el procesamiento con Spark.

## 2. Seleccion de variables base
- Mood: se usa el dataset etiquetado y se elimina `uri` porque no aporta al entrenamiento.
- Tracks: se usa `track_id` y las columnas acusticas disponibles para clasificacion posterior.
- Lyrics: se usa `track_id`, columnas acusticas compatibles y senales derivadas de `lyrics`.
- Se conservan features acusticas y `duration_ms`.

## 3. Limpieza de nulos e inconsistencias
- Mood: se convierten features numericas a `double`, target a `int`, y se eliminan filas sin `mood_label`.
- Tracks: se eliminan filas sin `track_id` y duplicados por `track_id`.
- Lyrics: se eliminan filas sin `track_id`, duplicados por `track_id` y se normaliza el texto de `lyrics`.
- En ambos: se imputan nulos con la mediana por feature.
- Motivo: evitar perder volumen de datos y corregir inconsistencias sin borrar registros utiles.

## 4. Tratamiento de outliers
- Se aplica clipping por IQR por feature (limites Q1-1.5*IQR y Q3+1.5*IQR).
- Motivo: reducir el impacto de valores extremos sin eliminar filas.

## 5. Ingenieria de features
- Mood y Tracks mantienen el esquema acustico compatible con el clasificador.
- Lyrics genera senales textuales compactas: longitud de letra, numero de palabras y tasas lexicas asociadas a sad, happy, energetic y calm.
- Motivo: enriquecer el recomendador sin guardar letras completas en el dataset preparado final.

## 6. Escalado
- Z-score por feature usando media y desviacion estandar calculadas con Spark en el dataset de mood.
- Tracks y Lyrics usan las mismas medias y desviaciones para mantener rangos compatibles con el modelo.
- Se guardan las estadisticas de escalado para reproducibilidad.

## 7. Split train/test
- Split estratificado 80/20 por `mood_label` usando Spark.
- Motivo: mantener la distribucion de clases en ambos conjuntos.

## 8. Guardado en Gold (S3)
- Mood: Spark escribe a `data_lake/tmp_gold/mood_prepared` y se sube a S3:
	- `s3://<bucket>/gold/mood_prepared/train`
	- `s3://<bucket>/gold/mood_prepared/test`
	- `s3://<bucket>/gold/mood_prepared/scaler_stats`
- Tracks: Spark escribe a `data_lake/tmp_gold/tracks_prepared` y se sube a S3:
	- `s3://<bucket>/gold/tracks_prepared/full`
- Lyrics: Spark escribe a `data_lake/tmp_gold/lyrics_prepared` y se sube a S3:
	- `s3://<bucket>/gold/lyrics_prepared/full`
- Motivo: dejar listo el dataset de entrenamiento y el dataset final para clasificacion.
