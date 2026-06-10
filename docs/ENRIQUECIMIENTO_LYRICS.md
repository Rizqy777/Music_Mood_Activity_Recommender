# Integracion del dataset de lyrics

El dataset `songs_with_attributes_and_lyrics.csv` forma parte del flujo principal del proyecto.

## 1. Pipeline

`scripts/run_pipeline.py` trabaja ahora con tres fuentes:

- `mood_dataset`
- `tracks_dataset`
- `lyrics_dataset`

El tercer dataset se produce en Kafka, se consume a Bronze, se transforma con Spark a Silver y se sube a S3 igual que los demas.

Rutas principales:

```text
data_lake/bronze/lyrics_dataset/
data_lake/silver/lyrics_dataset/
data_lake/gold/lyrics_dataset/
s3://<bucket>/bronze/lyrics_dataset/
s3://<bucket>/silver/lyrics_dataset/
s3://<bucket>/gold/lyrics_dataset/
```

## 2. Preparacion

`notebooks/preparacion_datos.ipynb` descarga `silver/lyrics_dataset/` desde S3, limpia duplicados por `track_id`, escala sus audio features con las mismas estadisticas del dataset etiquetado y genera:

```text
data_lake/tmp_gold/lyrics_prepared/full
s3://<bucket>/gold/lyrics_prepared/full
```

La letra completa no se guarda en `lyrics_prepared/full`; se transforma en senales ligeras:

- `lyrics_length`
- `lyrics_word_count`
- `lyrics_sad_lexicon_rate`
- `lyrics_happy_lexicon_rate`
- `lyrics_energetic_lexicon_rate`
- `lyrics_calm_lexicon_rate`

## 3. Recomendador

`notebooks/build_recommender.ipynb` y `scripts/build_recommender.py` cargan:

```text
gold/tracks_prepared/full
gold/lyrics_prepared/full
```

Despues unen ambos catalogos, clasifican las canciones con el modelo emocional acustico y, cuando existen senales de lyrics, combinan:

```text
0.65 * audio_proba + 0.35 * lyrics_lexicon_proba
```

El catalogo final sigue siendo:

```text
data_lake/recommender/classified_tracks.parquet
```

por lo que la app no necesita un flujo alternativo.
