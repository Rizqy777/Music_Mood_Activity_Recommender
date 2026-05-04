# PROJECT SPECIFICATION — Punto 2 del Proyecto Final ML

## Objetivo de esta fase

Implementar EXCLUSIVAMENTE el apartado 2 del proyecto:

"Obtención y almacenamiento de los datos"

NO avanzar a modelado, entrenamiento, EDA ni despliegue final salvo que sea estrictamente necesario para preparar la arquitectura de datos.

---

## Contexto del Proyecto

Se está construyendo un sistema de recomendación musical basado en emociones.

Arquitectura global futura:

Mood Dataset → Entrenamiento modelo emocional  
Spotify Tracks Dataset → Catálogo de inferencia / recomendación  

Pero en esta fase SOLO debe implementarse la pipeline de datos.

---

## Datasets Disponibles

Ubicados en carpeta:

/datasets

Archivos:

- mood_dataset.csv
- spotify_tracks_dataset.csv

Documentación descriptiva de datasets:

- README_278k_labelled_uri.md
- README_spotify_tracks_dataset.MD

---

## REGLA CRÍTICA SOBRE DATASETS

NO abrir ni cargar completamente los CSV directamente para inspección manual inicial si no es necesario.

Son datasets grandes y pueden congelar el sistema.

Para entender estructura:

1. Leer primero los archivos README .md
2. Si se necesita inspección adicional:
   - usar scripts Python / notebooks que carguen muestras parciales
   - usar head(), sample(), nrows limitados, etc.

---

## Tecnologías Obligatorias

La implementación debe usar:

- Apache Kafka (Docker, local)
- Apache Spark / PySpark (Docker, local)
- AWS S3 (boto3)
- AWS Lambda (simulado / preparado para trigger)
- AWS RDS
- MongoDB Atlas o local Mongo
- Todo orquestado desde scripts Python / shell

---

## Restricción Arquitectónica

TODO el pipeline debe poder ejecutarse desde UN ÚNICO SCRIPT maestro.

Ese script debe:

1. Levantar arquitectura necesaria
2. Ingerir datasets
3. Procesarlos
4. Guardarlos en almacenamiento final

---

## Credenciales AWS

Las credenciales AWS estarán en:

/aws_credentials.env

El código debe cargarlas desde ahí.

NO hardcodear credenciales.

---

## Pipeline a Implementar

### 1. Producer Kafka

Leer datasets por filas simulando streaming.

Enviar eventos a Kafka.

---

### 2. Consumer Kafka

Consumir eventos.

Persistir datos RAW en S3 Bronze Layer.

Rutas recomendadas:

s3://bucket/bronze/mood_dataset/
s3://bucket/bronze/tracks_dataset/

---

### 3. Spark Bronze → Silver

Procesar datasets Bronze y generar Silver.

---

## Limpiezas Silver — Mood Dataset

- Eliminar:
  - Unnamed: 0
  - Unnamed: 0.1

- Renombrar:
  - duration (ms) → duration_ms
  - labels → mood_label

- Convertir tipos correctos:
  - Audio features → float
  - duration_ms → int

- Tratar nulos
- Eliminar duplicados por uri
- Validar rangos de audio features

Guardar en:

s3://bucket/silver/mood_dataset/

---

## Limpiezas Silver — Tracks Dataset

- Eliminar:
  - Unnamed: 0

- Convertir tipos correctos:
  - popularity → int
  - explicit → bool
  - duration_ms → int
  - Audio features → float

- Tratar nulos
- Eliminar duplicados por track_id
- Normalizar strings textuales

Guardar en:

s3://bucket/silver/tracks_dataset/

---

## Gold Layer

Preparar lógica / estructura para Gold aunque el modelo aún no esté entrenado.

Gold final se generará más adelante tras inferencia.

Debe dejar preparada la función / pipeline para:

### Gold Mood Dataset

- Selección final de features entrenamiento
- Encoding de labels
- Dataset listo para ML

### Gold Tracks Dataset

Placeholder / pipeline preparada para:

- predicted_mood
- context_tag heurístico
- recommendation_score

Guardar en:

s3://bucket/gold/

---

## Requisitos de Código

Generar estructura modular y profesional:

/docker
/scripts
/src
    /kafka
    /spark
    /aws
    /pipeline

---

## Entregables Esperados

El agente debe implementar:

1. docker-compose para Kafka/Spark
2. Producer Kafka
3. Consumer Kafka
4. Scripts Spark Bronze→Silver
5. Preparación pipeline Silver→Gold
6. Script maestro único de orquestación
7. Código limpio, modular y comentado

---

## Importante

NO improvisar funcionalidades fuera del alcance.

NO implementar todavía:
- entrenamiento de modelos
- EDA profundo
- Gradio
- Hugging Face
- recomendaciones finales

Centrarse exclusivamente en pipeline de datos del punto 2.