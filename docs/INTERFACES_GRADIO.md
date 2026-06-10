# Interfaces de Usuario Gradio

Este documento describe las dos aplicaciones interactivas construidas con Gradio (`app.py` y `app_realtime_predictions.py`) y como se integran funcionalmente en el proyecto.

## 1. Recomendador Principal (`app.py`)

Esta es la aplicacion principal que expone el recomendador basado en emocion y actividad.

**Caracteristicas principales:**
- Interfaz web estilizada con Vanilla CSS inyectado sobre Gradio (`gr.themes.Base()`).
- Busqueda cruzada: permite indicar emocion ("Como te sientes") y texto libre ("Que actividad o momento tienes ahora").
- Filtros opcionales por artista y genero.
- Generacion de un pool de candidatos que se filtran por rango de recomendacion.
- Tarjetas de resultados con reproductor integrado de Spotify (`iframe` de embed) para escuchar las canciones directamente.

**Flujo interno:**
1. El usuario introduce parametros.
2. `MusicActivityRecommender` genera scores usando el modelo neuronal/RF.
3. Se devuelven las mejores combinaciones que cumplan el threshold de score.
4. Se presentan visualmente mostrando metadatos, tags de la prediccion y enlace/embed de Spotify.

## 2. Clasificador Realtime (`app_realtime_predictions.py`)

Esta aplicacion complementa al recomendador principal permitiendo anadir nuevas canciones al catalogo al vuelo.

**Caracteristicas principales:**
- **Analisis de URL de Spotify:** Si el usuario introduce una URL de track de Spotify, la aplicacion usa `RapidAPI/Soundnet` para extraer las caracteristicas acusticas (audio features como danceability, energy, etc.) en tiempo real, sin depender del dataset local.
- **Integracion OEmbed y Spotify API:** Para rellenar metadatos (nombre, artista, portada) usa la API de Spotify o OEmbed.
- **Clasificacion Manual:** Se pueden ajustar los descriptores acusticos manualmente antes de enviarlos.
- **Clasificacion de Datasets:** Permite subir un CSV/Parquet entero. El modelo emocional (`RealtimeMoodClassifier`) clasifica todos los tracks y los anexa al archivo `classified_tracks.parquet` usado por el recomendador.

**Impacto:**
Al anadir una cancion por aqui, aparecera disponible en `app.py` para futuras recomendaciones si encaja en el score y el mood.

## 3. Integracion en Web App (`web_app.py`)

Ambas interfaces Gradio no necesitan lanzarse por separado en un entorno de produccion. Se integran dentro de una aplicacion FastAPI (`web_app.py`) usando `gr.mount_gradio_app()`. Esto permite:
- Tener un login unificado local para proteger el acceso.
- Compartir el mismo dominio y puerto en AWS.
- Navegar entre Recomendador y Realtime mediante pestanas en la parte superior.