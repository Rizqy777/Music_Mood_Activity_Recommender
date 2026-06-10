# Recomendador por emocion y actividad

Este documento explica la capa que faltaba antes del despliegue final: usar el catalogo de canciones `tracks_prepared/full` y convertirlo en un recomendador.

## 1. Por que hacia falta este paso

Hasta ahora el proyecto tenia un modelo que clasifica canciones por emocion:

- sad
- happy
- energetic
- calm

Pero eso no basta para recomendar bien. Si una persona dice "estoy triste", no siempre quiere musica triste. Si ademas dice "voy al gym", la recomendacion deberia buscar algo compatible con la tristeza, pero con energia suficiente para entrenar.

Por eso se crea una segunda capa:

```text
cancion + emocion del usuario + actividad -> puntuacion de recomendacion
```

## 2. Uso del dataset de tracks

Ahora si se usa el dataset:

```text
data_lake/tmp_gold/tracks_prepared/full
```

El proceso hace dos cosas:

1. Clasifica cada track del catalogo con el modelo emocional entrenado.
2. Entrena un recomendador neuronal que aprende a ordenar canciones segun emocion y actividad.

El catalogo clasificado se guarda en:

```text
data_lake/recommender/classified_tracks.parquet
data_lake/recommender/classified_tracks.csv
```

## 3. Modelos comparados y seleccion

El recomendador ya no entrena un unico modelo. En `notebooks/build_recommender.ipynb` y `scripts/build_recommender.py` se comparan dos modelos vistos en clase:

- `RandomForestRegressor`: bosque aleatorio para regresion sobre datos tabulares.
- `MLPRegressor`: red neuronal densa sencilla.

En ambos se modifican parametros mediante `GridSearchCV`. La seleccion se hace por menor RMSE en test, usando tambien MAE y R2 como metricas de validacion.

Entrada del modelo:

- Features musicales de la cancion.
- Probabilidades emocionales del clasificador: sad, happy, energetic, calm.
- Estado emocional indicado por el usuario.
- Perfil numerico de la actividad.

Salida del modelo:

```text
recommendation_score_nn
```

Es una puntuacion entre 0 y 1. Cuanto mas alta, mas adecuada es la cancion para esa combinacion de emocion y actividad.

Si la red neuronal resulta ganadora se guarda como mejor modelo; si gana Random Forest, la aplicacion usa ese modelo. Para no romper el flujo anterior, tambien se mantiene el nombre estable:

```text
models/activity_recommender_mlp.joblib
```

Ese archivo contiene el mejor recomendador disponible, aunque internamente pueda ser Random Forest.

## 4. Como se interpreta la actividad

Para actividades conocidas se usa un perfil musical. Por ejemplo:

- gym: mucha energia, mucho movimiento, poca calma.
- correr: mucha energia y ritmo.
- bailar: energia, movimiento y positividad.
- estudiar: foco, calma y poca speechiness.
- dormir: mucha calma, poca energia.
- comer: calma media, energia baja.
- llorar/desahogarme: poca energia, baja positividad, mas calma y acousticness.

Si el usuario escribe una actividad libre, el sistema la interpreta con un modelo linguistico ligero entrenado con ejemplos de familias de actividad y con reglas de palabras clave. Ademas, cuando la prediccion es dudosa, se usa el perfil musical para ajustar la etiqueta. No intenta guardar todas las actividades posibles una a una. Hace dos cosas:

- Predice la familia de actividad, por ejemplo `estudio_trabajo` o `limpieza_domestica`.
- Convierte el texto en dimensiones musicales:

- movimiento
- energia
- positividad
- foco
- calma
- acousticness

Por ejemplo:

```text
"quiero llorar" -> llorar
"voy a entrenar en el gym" -> gym
"voy a estudiar programacion" -> estudiar
"voy a estudiar" -> estudio_trabajo
"quiero limpiar la casa" -> limpiar
"quiero fregar el suelo" -> limpieza_domestica
"voy a comer" -> alimentacion
```

Esto permite que frases nuevas se comporten de forma razonable. Si alguien escribe "fregar el suelo" o "voy a comer", el sistema no necesita tener esa actividad exacta en la interfaz: la interpreta como una actividad con movimiento/energia coherentes.

## 5. Como se entrena si no hay etiquetas reales de actividad

No existe todavia un dataset con feedback real de usuarios, por ejemplo:

```text
usuario triste + gym + cancion X = buena recomendacion
```

Por eso se usa una tecnica practica llamada weak supervision. En lugar de etiquetar todo a mano, se generan puntuaciones iniciales con reglas musicales sencillas:

- Para actividades fisicas se valora mas energia, danceability y tempo.
- Para estudiar se valora mas calma y poca speechiness.
- Para relajarse o dormir se valora mas acousticness y baja energia.
- Para tristeza + gym no se fuerza musica triste pura; se permite subir hacia energetic/happy.
- Para tristeza + llorar/desahogarse se priorizan canciones sad/calm y se penalizan recomendaciones happy.

El modelo elegido aprende estas combinaciones y produce una puntuacion continua.

Importante: las metricas del recomendador miden si aprende bien estas reglas iniciales, no si ya satisface a usuarios reales. La mejora real debera medirse mas adelante con feedback o pruebas manuales.

## 6. Artefactos generados

```text
models/activity_text_interpreter.joblib
models/activity_text_interpreter_metadata.json
models/activity_recommender_random_forest.joblib
models/activity_recommender_mlp.joblib
models/activity_recommender_best.joblib
models/activity_recommender_metadata.json
data_lake/recommender/classified_tracks.parquet
data_lake/recommender/classified_tracks.csv
data_lake/recommender/recommender_metrics.csv
data_lake/recommender/recommender_training_sample.csv
data_lake/recommender/activity_text_training_examples.csv
```

La interfaz tambien muestra un enlace directo a Spotify para cada cancion:

```text
https://open.spotify.com/intl-es/track/<track_id>
```

Metricas de la ejecucion:

```text
MAE  = 0.0062
RMSE = 0.0084
R2   = 0.9924
```

Estas metricas son altas porque el objetivo se genera con reglas. Se documentan como control tecnico, no como prueba final de calidad del recomendador.

## 7. Como ejecutar

Entrenar el recomendador:

```powershell
.\.venv\Scripts\python.exe scripts\build_recommender.py
```

Entrenar el interprete de actividad libre:

```powershell
.\.venv\Scripts\python.exe scripts\train_activity_text_model.py
```

Probar una recomendacion desde Python:

```powershell
@'
from src.recommender import MusicActivityRecommender

r = MusicActivityRecommender()
print(r.recommend("triste", "voy a entrenar en el gym", 5))
'@ | .\.venv\Scripts\python.exe -
```

Lanzar la interfaz Gradio:

```powershell
.\.venv\Scripts\python.exe app.py
```

## 8. Resultado conceptual

El sistema ya no recomienda solo por emocion literal. Ahora hace una recomendacion contextual:

```text
estado emocional + texto libre de actividad + caracteristicas musicales -> canciones ordenadas
```

Esto encaja mejor con la idea final del proyecto, porque permite casos como:

- Estoy triste, pero voy al gym.
- Estoy feliz y voy a estudiar.
- Estoy tranquilo y quiero caminar.
- Estoy energico y voy a limpiar la casa.

La siguiente mejora natural seria guardar feedback del usuario, por ejemplo si acepta o rechaza una recomendacion, y usarlo como fine-tuning real del recomendador.
