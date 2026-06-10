# AWS RDS MySQL, MongoDB Atlas y Athena

## Variables de entorno

Ejemplo para `.env` o `aws_credentials.env`:

```text
MONGO_URI=mongodb+srv://<user>:<password>@<cluster-url>/music_recommender?retryWrites=true&w=majority
MONGO_DATABASE=music_recommender

RDS_HOST=<mysql-rds-endpoint>.rds.amazonaws.com
RDS_PORT=3306
RDS_DATABASE=music_recommender
RDS_USER=admin
RDS_PASSWORD=<password>

ATHENA_DATABASE=music_recommender_lake
ATHENA_RESULTS_S3=s3://<bucket>/athena/results/
ATHENA_WORKGROUP=primary

AWS_LAMBDA_FUNCTION_NAME=music-recommender-pipeline-metadata
AWS_LAMBDA_INVOCATION_TYPE=Event
```

## Que se guarda en RDS MySQL

Tabla: `pipeline_runs`

Uso: registrar de forma estructurada cada salida generada por el pipeline.

Columnas:

- `id`
- `dataset_name`
- `layer_name`
- `storage_path`
- `created_at`

Ejemplo:

| dataset_name | layer_name | storage_path |
|---|---|---|
| mood_dataset | silver | `C:\...\data_lake\silver\mood_dataset` |
| tracks_dataset | silver | `C:\...\data_lake\silver\tracks_dataset` |
| lyrics_dataset | silver | `C:\...\data_lake\silver\lyrics_dataset` |

## Que se guarda en MongoDB Atlas

Base de datos: `music_recommender`

Coleccion: `data_lake_layers`

Uso: registrar metadatos flexibles de las capas del data lake.

Ejemplo de documento:

```json
{
  "dataset": "lyrics_dataset",
  "layer": "silver",
  "path": "C:\\Users\\...\\data_lake\\silver\\lyrics_dataset",
  "created_at": "2026-05-09T10:30:00Z"
}
```

MongoDB se usa para metadatos semiestructurados, mientras que RDS MySQL se usa para trazabilidad tabular.

## Que se registra en Glue/Athena

AWS Glue Data Catalog registra tablas externas sobre los Parquet de S3.

Ejemplos:

- `silver_mood_dataset`
- `silver_tracks_dataset`
- `silver_lyrics_dataset`
- `gold_mood_prepared_train`
- `gold_mood_prepared_test`
- `gold_tracks_prepared_full`
- `gold_lyrics_prepared_full`

Athena ejecuta consultas ligeras de validacion, por ejemplo:

```sql
SELECT COUNT(*) AS rows_count
FROM music_recommender_lake.silver_lyrics_dataset;
```

## Que hace AWS Lambda

El pipeline invoca la funcion configurada en `AWS_LAMBDA_FUNCTION_NAME` cada vez que registra una salida de capa del data lake.

Evento enviado:

```json
{
  "event_type": "data_lake_layer_written",
  "layer": "silver",
  "dataset": "lyrics_dataset",
  "location": "C:\\Users\\...\\data_lake\\silver\\lyrics_dataset",
  "created_at": "2026-05-09T10:30:00Z"
}
```

La funcion incluida en `src/aws/lambda_handler.py` guarda una copia de auditoria del evento en S3:

```text
s3://<bucket>/lambda/audit/<layer>/<dataset>/<timestamp>.json
```

Despliegue:

```powershell
.\.venv\Scripts\python.exe scripts\deploy_lambda.py --role-arn arn:aws:iam::<account-id>:role/<lambda-role>
```
