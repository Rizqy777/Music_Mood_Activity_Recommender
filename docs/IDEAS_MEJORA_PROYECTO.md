# Ideas para mejorar el proyecto

Estas ideas estan ordenadas por impacto y facilidad.

## 1. Mejoras de datos

- Anadir un dataset externo con mas variedad de estados (por ejemplo, moods con mas etiquetas o subgeneros).
- Incluir datos con letras o descripciones (lyrics) para afinar emociones.
- Usar dataset de actividades reales (por ejemplo, playlists publicas con titulo de actividad).

## 2. Mejoras del clasificador emocional

- Probar modelos con embeddings o redes neuronales ligeras.
- Balancear clases con mas ejemplos de calm.
- Usar validacion temporal o por artista para medir generalizacion real.

## 3. Mejoras del recomendador

- Guardar feedback en la app (like/dislike) para reentrenar.
- Crear un modelo de ranking con ejemplos reales del usuario.
- Incluir diversidad en el top_k para evitar canciones repetidas.

## 4. Mejoras de interpretacion de actividad

- Usar embeddings semanticos para entender texto libre sin tantas reglas.
- Agrupar actividades por contexto (trabajo, deporte, ocio) y ajustar pesos.

## 5. Mejoras de producto

- Perfil de usuario (edad, idioma, horario) para ajustar el tono.
- Filtros por genero musical o energia.
- Modo demo que muestre pasos del pipeline en tiempo real.

## 6. Mejores datasets sugeridos

- Dataset de playlists publicas de Spotify con nombre de actividad.
- Dataset con letras (lyrics) para detectar emocion real.
- Dataset de actividad fisica (tempo y energia asociados a entrenamiento).
