# Reset completo para ejecutar desde cero

Este documento indica que puedes borrar para recomenzar el proyecto sin datos generados. Separamos en dos niveles: minimo y total.

## 1. Reset minimo (recomendado)

Borra solo salidas generadas por ejecuciones. Mantiene codigo, datasets originales y docs.

- data_lake/bronze/
- data_lake/silver/
- data_lake/gold/
- data_lake/tmp_silver/
- data_lake/tmp_gold/
- data_lake/recommender/
- data_lake/model_outputs/
- data_lake/prepared/
- data_lake/s3_cache/
- logs/
- models/
- __pycache__/

## 2. Reset total (desde 0 absoluto)

Incluye todo lo anterior y ademas:

- .venv/ (si quieres reinstalar dependencias)
- .specstory/ (historial local)
- chats/ (si no quieres logs de chat)

## 3. Que NO borrar

- datasets/ (datasets originales)
- docs/ (documentacion)
- scripts/ y src/ (codigo)
- docker/ (infraestructura)
- notebooks/ (cuadernos)
- images/ (banner)
- aws_credentials.env (si quieres conservar credenciales)

## 4. Aviso sobre S3

Si quieres un reset completo en S3, debes borrar manualmente el bucket que se genero o limpiar sus prefijos bronze/silver/gold.
