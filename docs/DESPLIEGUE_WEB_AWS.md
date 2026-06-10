# Web unificada y despliegue AWS

## Objetivo

Se ha anadido una web unica para alojar las dos interfaces Gradio del proyecto:

- `app.py`: recomendador por estado de animo y actividad.
- `app_realtime_predictions.py`: clasificador realtime de tracks y datasets.

La web esta implementada en `web_app.py` con FastAPI. La pagina principal muestra un login/registro minimalista. Las credenciales se guardan en `localStorage` del navegador y una cookie local permite navegar por la sesion sin crear una base de datos de usuarios.

## Rutas principales

- `/`: login y registro local.
- `/app`: pagina del recomendador con la app Gradio montada.
- `/realtime`: pagina del clasificador realtime con la app Gradio montada.
- `/logout`: salida de sesion.
- `/health`: comprobacion simple del servicio.

Internamente, Gradio queda montado en:

- `/gradio/recommender/`
- `/gradio/realtime/`

## Ejecucion local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Despues abrir:

```text
http://127.0.0.1:8000
```

## Despliegue en AWS

El despliegue se realiza con un unico script:

```powershell
.\.venv\Scripts\python.exe scripts\deploy_webapp_aws.py
```

El script:

1. Carga las credenciales y configuracion desde `.env`.
2. Empaqueta el proyecto sin `.venv`, notebooks, logs ni datasets pesados.
3. Conserva los artefactos necesarios para servir la app, especialmente `models/` y `data_lake/recommender/`.
4. Sube el paquete a `s3://$AWS_BUCKET_NAME/deployments/webapp/`.
5. Genera una URL prefirmada temporal para que la instancia descargue el paquete sin configurar IAM manualmente.
6. Crea o reutiliza un security group con HTTP abierto en el puerto 80.
7. Lanza una instancia EC2 Amazon Linux 2023.
8. Instala las dependencias web desde `requirements-web.txt`, descarga el paquete con Python, crea el servicio systemd `music-mood-web` y arranca Uvicorn.
9. Muestra la URL publica de acceso.

Por defecto se usa `t3.medium`, configurable con:

```powershell
$env:AWS_WEB_INSTANCE_TYPE="t3.medium"
.\.venv\Scripts\python.exe scripts\deploy_webapp_aws.py
```

Tambien se puede indicar region, bucket o AMI:

```powershell
.\.venv\Scripts\python.exe scripts\deploy_webapp_aws.py --region us-east-1 --bucket nombre-bucket --ami-id ami-xxxxxxxx
```

## Notas de seguridad

El login es deliberadamente simple para el laboratorio: no sustituye a un sistema real de autenticacion. Las credenciales quedan guardadas en el navegador y se usa una cookie local para mantener la sesion. Para un entorno productivo convendria sustituirlo por autenticacion real en servidor, HTTPS y gestion segura de secretos.
