# Data Pipeline - Punto 2

Implementa la fase de obtencion y almacenamiento de datos para el recomendador musical por emociones.

## Ejecucion

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

El script maestro levanta la arquitectura local con Docker Compose, produce eventos Kafka desde muestras/chunks de CSV, consume a Bronze, transforma con Spark a Silver y prepara Gold. Durante la ejecucion emite logs por pantalla y en `logs/`.

Para pruebas locales sin S3 real:

```bash
USE_S3=false python scripts/run_pipeline.py
```

En PowerShell:

```powershell
$env:USE_S3="false"; python scripts/run_pipeline.py
```

Para abrir ventanas de monitorizacion en tiempo real en Windows:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --monitor-terminals
```

Esto abre monitores para Pipeline, Kafka (producer/admin), Bronze (consumer), Silver y Gold.

Para una ejecucion reproducible que no mezcle Bronze de ejecuciones anteriores:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --clean-bronze
```

Para omitir la capa Gold por ahora:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --skip-gold
```

## Capas

- Bronze: `data_lake/bronze/mood_dataset/` y `data_lake/bronze/tracks_dataset/`
- Silver: `data_lake/silver/mood_dataset/` y `data_lake/silver/tracks_dataset/`
- Gold: `data_lake/gold/mood_dataset/` y `data_lake/gold/tracks_dataset/`

Si `USE_S3=true`, las mismas capas se espejan en `s3://$AWS_BUCKET_NAME/`.

## Variables importantes

- `MAX_ROWS_PER_DATASET`: limita filas por dataset para streaming local. Vacio procesa todo.
- `PRODUCER_BATCH_SIZE`: filas por batch/chunk enviado a Kafka.
- `PRODUCER_BATCH_DELAY_SECONDS`: pausa entre batches para modo demo (0 = sin pausa).
- `AWS_BUCKET_NAME`, `AWS_DEFAULT_REGION`: se cargan desde `aws_credentials.env`.
- `MONGO_URI`: MongoDB local o Atlas.
- `RDS_DSN`: PostgreSQL local compatible con RDS o endpoint RDS real.

### Modo demo de streaming (Kafka)

Windows PowerShell:

```powershell
$env:PRODUCER_BATCH_SIZE="200"
$env:PRODUCER_BATCH_DELAY_SECONDS="1.5"
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Linux / Ubuntu:

```bash
PRODUCER_BATCH_SIZE=200 PRODUCER_BATCH_DELAY_SECONDS=1.5 .venv/bin/python scripts/run_pipeline.py
```

No se implementa EDA avanzado, entrenamiento, Gradio, Hugging Face ni recomendaciones finales.
