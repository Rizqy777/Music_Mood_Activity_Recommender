# Guia de Arranque - Pipeline de Datos

Esta guia explica como arrancar la fase 2 del proyecto: obtencion, ingesta, procesamiento y almacenamiento de datos.

El pipeline implementa:

- Docker Compose con Kafka, Spark, MongoDB y PostgreSQL compatible con RDS.
- Producer Kafka desde los CSV disponibles en `datasets/`.
- Consumer Kafka hacia Bronze.
- Transformaciones Spark Bronze -> Silver.
- Preparacion Spark Silver -> Gold.
- Escritura local en `data_lake/`.
- Escritura en S3 si `USE_S3=true`.
- Registro de metadatos en MongoDB y PostgreSQL local.

No ejecuta EDA avanzado, entrenamiento, Gradio, Hugging Face ni recomendaciones finales.

## 1. Requisitos Previos

### Windows

Instalar:

- Python 3.11 o superior.
- Docker Desktop.
- Git Bash opcional, aunque PowerShell es suficiente.

Comprobar:

```powershell
python --version
docker --version
docker compose version
```

### Linux / Ubuntu

Instalar:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip docker.io docker-compose-plugin
```

Comprobar:

```bash
python3 --version
docker --version
docker compose version
```

Si tu usuario no puede usar Docker sin `sudo`:

```bash
sudo usermod -aG docker $USER
```

Cierra sesion y vuelve a entrar.

## 2. Credenciales AWS

Las credenciales deben estar en:

```text
aws_credentials.env
```

Formato admitido:

```env
aws_access_key_id=...
aws_secret_access_key=...
aws_session_token=...
```

Tambien se admiten nombres en mayusculas:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_DEFAULT_REGION=us-east-1
```

Si no defines `AWS_DEFAULT_REGION`, el pipeline usa `us-east-1`, que es la region donde el laboratorio ha permitido crear buckets.

Si no defines `AWS_BUCKET_NAME`, se genera automaticamente:

```text
music-recommender-data-lake-<account_id>
```

## 3. Crear y Activar Entorno Python

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux / Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. Arrancar Pipeline Completo con S3

### Windows PowerShell

```powershell
Remove-Item Env:\USE_S3 -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Linux / Ubuntu

```bash
unset USE_S3
.venv/bin/python scripts/run_pipeline.py
```

El script:

1. Levanta Docker Compose.
2. Espera a Kafka, MongoDB y PostgreSQL y verifica que Kafka este realmente en ejecucion.
3. Prepara el bucket S3 automaticamente.
4. Produce eventos Kafka desde los CSV.
5. Consume eventos hacia Bronze.
6. Ejecuta Spark Bronze -> Silver.
7. Ejecuta Spark Silver -> Gold.
8. Registra metadatos en MongoDB y PostgreSQL.

Durante la ejecucion se escriben logs en:

```text
logs/pipeline.log
logs/00_docker.log
logs/01_storage_s3.log
logs/02_kafka.log
logs/03_bronze.log
logs/04_silver.log
logs/05_gold.log
logs/06_metadata.log
logs/07_s3.log
```

Si quieres monitorizar por ventanas separadas en Windows:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --monitor-terminals
```

Esto abre monitores para Pipeline, Kafka (producer/admin), Bronze (consumer), Silver y Gold.

Si quieres que cada ejecucion procese solo los eventos nuevos de esa ejecucion y no archivos Bronze anteriores:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --clean-bronze
```

Si quieres omitir la capa Gold por ahora:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --skip-gold
```

## 5. Arrancar Pipeline Solo Local, Sin S3

Usalo para pruebas sin AWS.

### Windows PowerShell

```powershell
$env:USE_S3="false"
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Linux / Ubuntu

```bash
USE_S3=false .venv/bin/python scripts/run_pipeline.py
```

## 6. Limitar o Ampliar Filas Procesadas

Por defecto se procesan 1000 filas por dataset para pruebas rapidas.

### Windows PowerShell

```powershell
$env:MAX_ROWS_PER_DATASET="5000"
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Linux / Ubuntu

```bash
MAX_ROWS_PER_DATASET=5000 .venv/bin/python scripts/run_pipeline.py
```

Para procesar todo el CSV, deja la variable vacia:

### Windows PowerShell

```powershell
$env:MAX_ROWS_PER_DATASET=""
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Linux / Ubuntu

```bash
MAX_ROWS_PER_DATASET= .venv/bin/python scripts/run_pipeline.py
```

## 6.1. Modo Demo de Streaming (Kafka)

Usa `PRODUCER_BATCH_SIZE` para definir el tamano del chunk y
`PRODUCER_BATCH_DELAY_SECONDS` para introducir una pausa entre batches.

### Windows PowerShell

```powershell
$env:PRODUCER_BATCH_SIZE="200"
$env:PRODUCER_BATCH_DELAY_SECONDS="1.5"
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Linux / Ubuntu

```bash
PRODUCER_BATCH_SIZE=200 PRODUCER_BATCH_DELAY_SECONDS=1.5 .venv/bin/python scripts/run_pipeline.py
```

## 7. Verificar Resultados Locales

### Windows PowerShell

```powershell
Get-ChildItem -Recurse data_lake
```

### Linux / Ubuntu

```bash
find data_lake -type f
```

Estructura esperada:

```text
data_lake/
  bronze/
    mood_dataset/
    tracks_dataset/
  silver/
    mood_dataset/
    tracks_dataset/
  gold/
    mood_dataset/
    tracks_dataset/
```

## 8. Verificar Resultados en S3

### Windows PowerShell

```powershell
@'
import boto3
from src.config import load_settings

settings = load_settings()
s3 = boto3.client("s3", region_name=settings.aws_region)

for prefix in [
    "bronze/mood_dataset/",
    "bronze/tracks_dataset/",
    "silver/mood_dataset/",
    "silver/tracks_dataset/",
    "gold/mood_dataset/",
    "gold/tracks_dataset/",
]:
    response = s3.list_objects_v2(Bucket=settings.bucket_name, Prefix=prefix)
    print(prefix, response.get("KeyCount", 0))
'@ | .\.venv\Scripts\python.exe -
```

### Linux / Ubuntu

```bash
.venv/bin/python - <<'PY'
import boto3
from src.config import load_settings

settings = load_settings()
s3 = boto3.client("s3", region_name=settings.aws_region)

for prefix in [
    "bronze/mood_dataset/",
    "bronze/tracks_dataset/",
    "silver/mood_dataset/",
    "silver/tracks_dataset/",
    "gold/mood_dataset/",
    "gold/tracks_dataset/",
]:
    response = s3.list_objects_v2(Bucket=settings.bucket_name, Prefix=prefix)
    print(prefix, response.get("KeyCount", 0))
PY
```

## 9. Parar la Arquitectura

### Windows PowerShell

```powershell
docker compose -f docker\docker-compose.yml down
```

### Linux / Ubuntu

```bash
docker compose -f docker/docker-compose.yml down
```

Si quieres borrar volumenes locales de MongoDB y PostgreSQL:

### Windows PowerShell

```powershell
docker compose -f docker\docker-compose.yml down -v
```

### Linux / Ubuntu

```bash
docker compose -f docker/docker-compose.yml down -v
```

## 10. Es un Proceso en Streaming?

Si, pero con una precision importante:

- La ingesta esta implementada como streaming simulado mediante Kafka.
- El producer lee los CSV por chunks y envia cada fila como evento Kafka.
- El consumer consume esos eventos desde Kafka y los persiste en Bronze.
- Por tanto, la entrada Bronze se construye mediante flujo de eventos.

No es todavia un streaming continuo infinito ni Spark Structured Streaming.

En esta fase, el flujo es bounded streaming: parte de CSV estaticos, simula eventos fila a fila, consume desde Kafka y despues ejecuta transformaciones Spark por lotes sobre Bronze para generar Silver y Gold.

Esto cumple el objetivo de la fase de datos: incorporar Kafka como capa de ingesta en tiempo real/simulada sin avanzar a modelado ni despliegue final.
