# Flujo basico de ejecucion (paso a paso)

Este documento deja solo el flujo esencial para ejecutar el proyecto de principio a fin.

## 1. Preparar entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Pipeline (Bronze y Silver)

```powershell
Remove-Item Env:\USE_S3 -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Este paso ingiere y transforma los tres datasets principales:

- `mood_dataset`
- `tracks_dataset`
- `lyrics_dataset`

Tambien registra las tablas Silver generadas en AWS Glue Data Catalog y lanza consultas de validacion en Amazon Athena.

## 3. EDA (visualizacion y analisis)

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\eda.ipynb
```
Ejecuta todas las celdas del notebook.

## 4. Preparacion de datos (Gold preparado)

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\preparacion_datos.ipynb
```
Ejecuta todas las celdas. Se generan:

- data_lake/tmp_gold/mood_prepared/train
- data_lake/tmp_gold/mood_prepared/test
- data_lake/tmp_gold/tracks_prepared/full
- data_lake/tmp_gold/lyrics_prepared/full

Al final del notebook tambien se registran las tablas Gold en AWS Glue Data Catalog y se validan con Amazon Athena.

## 5. Entrenamiento de modelos (notebook)

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\train_models.ipynb
```
Ejecuta las celdas para ver metricas y tiempos de entrenamiento.

## 6. Interprete de actividad (notebook)

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\train_activity_text_model.ipynb
```

## 7. Recomendador por actividad (notebook)

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\build_recommender.ipynb
```

## 8. Pruebas rapidas

```powershell
.\.venv\Scripts\python.exe scripts\check_recommender_cases.py
```

## 9. Lanzar interfaz web unificada en local

```powershell
.\.venv\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000`. La pagina de login guarda el usuario en el navegador y permite acceder al recomendador (`app.py`) y al clasificador realtime (`app_realtime_predictions.py`) desde una misma web.

Si necesitas lanzar las apps Gradio de forma independiente para pruebas:

```powershell
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe app_realtime_predictions.py
```

## 10. Despliegue automatico en AWS

Con las credenciales y `AWS_BUCKET_NAME` ya definidos en `.env`, ejecutar:

```powershell
.\.venv\Scripts\python.exe scripts\deploy_webapp_aws.py
```

El script empaqueta el proyecto, sube el paquete a S3, crea o reutiliza un security group HTTP, lanza una instancia EC2 con Amazon Linux 2023, instala las dependencias web desde `requirements-web.txt` y arranca `web_app.py` con Uvicorn en el puerto 80. Al terminar muestra la URL publica.
