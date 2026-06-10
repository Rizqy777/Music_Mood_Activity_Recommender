# Creacion y entrenamiento del modelo

Este documento resume los puntos 5 y 6 del proyecto: entrenar al menos dos modelos vistos en clase, ajustar hiperparametros, comparar su bondad de ajuste y validar el resultado con metricas.

## 1. Objetivo del modelo

El objetivo es predecir `mood_label`, es decir, la emocion asociada a una cancion:

- `0`: sad
- `1`: happy
- `2`: energetic
- `3`: calm

El problema es de clasificacion supervisada multiclase. El modelo aprende a partir de canciones ya etiquetadas y despues podra usarse para clasificar canciones nuevas del catalogo musical.

## 2. Dataset utilizado

Se usa el dataset `mood_prepared` generado en la fase de preparacion de datos.

Origen principal:

```text
s3://<bucket>/gold/mood_prepared/train
s3://<bucket>/gold/mood_prepared/test
```

El script `scripts/train_models.py` intenta leer primero desde S3. Si las credenciales AWS no estan disponibles o han caducado, puede trabajar con la copia local generada por el notebook de preparacion:

```text
data_lake/tmp_gold/mood_prepared/train
data_lake/tmp_gold/mood_prepared/test
```

En la ejecucion corregida se usaron:

- Entrenamiento: 800 canciones.
- Test: 200 canciones.
- Features: `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `loudness`, `tempo`, `duration_ms`.
- Target: `mood_label`.

La distribucion de clases no esta totalmente equilibrada. La clase `happy` tiene muchos mas ejemplos que `calm`, por eso se optimiza `f1_macro` en lugar de usar solo accuracy.

Durante la validacion se detecto que el split preparado inicialmente tenia duplicados exactos de features entre train y test. Eso provocaba resultados artificialmente altos en Random Forest. Para evitar esta fuga de informacion, el script usa por defecto `--split-strategy deduplicated`: une train y test, elimina duplicados exactos y rehace un split estratificado 80/20.

Para que los resultados sean reproducibles se usa semilla fija:

```text
RANDOM_STATE = 42
```

Esta semilla se aplica en la validacion cruzada y en los modelos que tienen parte aleatoria, como Random Forest.

## 3. Modelos elegidos

### Modelo 1: Logistic Regression

Se ha elegido porque es un modelo sencillo, rapido e interpretable. Sirve como buena linea base para saber si las variables acusticas separan razonablemente las emociones.

Arquitectura del pipeline:

```text
SimpleImputer(strategy="median")
LogisticRegression(class_weight="balanced")
```

Aunque los datos ya vienen preparados desde Gold, se mantiene un `SimpleImputer` dentro del pipeline para que el entrenamiento sea mas robusto si aparece algun nulo inesperado.

Parametros probados con `GridSearchCV`:

- `model__C`: `[0.1, 1.0, 10.0]`
- `model__fit_intercept`: `[True, False]`

El parametro `C` controla la regularizacion. Valores bajos hacen el modelo mas conservador; valores altos permiten ajustarse mas a los datos. `fit_intercept` decide si el modelo aprende un termino independiente.

Mejores parametros:

```text
model__C = 10.0
model__fit_intercept = True
```

Resultado obtenido:

- Mejor `f1_macro` en validacion cruzada: `0.6528`.
- `f1_macro` en test: `0.5903`.
- Accuracy en test: `0.7900`.

Este modelo funciona de forma aceptable, pero se queda corto para separar todas las emociones, especialmente cuando las relaciones entre variables no son lineales.

### Modelo 2: Random Forest

Se ha elegido porque suele funcionar muy bien con datos tabulares y puede aprender relaciones no lineales entre variables. Para este proyecto tiene sentido porque emociones como `happy`, `sad` o `energetic` no dependen de una sola variable, sino de combinaciones entre energia, valencia, tempo, loudness, etc.

Arquitectura del pipeline:

```text
SimpleImputer(strategy="median")
RandomForestClassifier(random_state=42)
```

Parametros probados con `GridSearchCV`:

- `model__n_estimators`: `[200, 400]`
- `model__max_depth`: `[None, 10, 20]`
- `model__min_samples_leaf`: `[1, 3]`
- `model__class_weight`: `["balanced", "balanced_subsample"]`

Estos parametros se eligieron por motivos sencillos:

- `n_estimators`: mas arboles suelen dar predicciones mas estables.
- `max_depth`: limita o permite la profundidad de los arboles.
- `min_samples_leaf`: evita hojas demasiado pequenas si se necesita reducir sobreajuste.
- `class_weight`: compensa que algunas emociones tengan menos ejemplos.

Mejores parametros:

```text
model__class_weight = "balanced_subsample"
model__max_depth = 10
model__min_samples_leaf = 3
model__n_estimators = 400
```

Resultado obtenido:

- Mejor `f1_macro` en validacion cruzada: `0.6412`.
- `f1_macro` en test: `0.6331`.
- Accuracy en test: `0.8600`.

El resultado ya es mas realista que la primera prueba. Random Forest mejora la accuracy y el F1 ponderado, aunque el F1 macro baja porque la clase `calm` tiene muy pocos ejemplos y es dificil evaluarla de forma estable.

## 4. Validacion del hito

Aunque el problema principal es de clasificacion multiclase, el hito pide calcular tambien:

- MAE
- RMSE
- R2

Por eso el entrenamiento informa dos grupos de metricas:

- Metricas de clasificacion: accuracy, precision, recall, F1 y ROC-AUC multiclase.
- Metricas numericas del hito: MAE, RMSE y R2 sobre las etiquetas `0-3`.

La seleccion del clasificador emocional se hace con `f1_macro`, porque el dataset esta desbalanceado y conviene dar peso a todas las emociones. MAE/RMSE/R2 quedan como validacion adicional requerida por el hito.

## 5. Comparacion inicial

| Modelo | Mejor CV f1_macro | Test f1_macro | Test accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.6528 | 0.5903 | 0.7900 |
| Random Forest | 0.6412 | 0.6331 | 0.8600 |

El modelo seleccionado como mejor candidato es `RandomForestClassifier`, porque obtiene mejores resultados tanto en validacion cruzada como en test.

## 6. Artefactos generados

El entrenamiento guarda:

```text
models/mood_logistic_regression.joblib
models/mood_random_forest.joblib
models/mood_best_model.joblib
models/mood_training_summary.json
data_lake/model_outputs/mood_classifier/metrics.csv
data_lake/model_outputs/mood_classifier/gridsearch_cv_results.csv
data_lake/model_outputs/mood_classifier/predictions.csv
```

`mood_best_model.joblib` contiene el mejor modelo segun `f1_macro` en test.

El notebook `notebooks/train_models.ipynb` sigue el mismo flujo visible paso a paso y conserva estos mismos nombres de salida, para que `build_recommender.ipynb` pueda cargar el modelo emocional sin cambios.

## 7. Como ejecutar el entrenamiento

Modo automatico, recomendado para trabajar:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

Este modo usa split deduplicado por defecto. Si se quiere reproducir exactamente el split preparado en Gold, se puede usar:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py --split-strategy prepared
```

Forzar lectura desde S3:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py --source s3
```

Forzar lectura local:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py --source local
```

Si se quiere subir tambien los artefactos entrenados a S3:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py --source s3 --upload-to-s3
```

## 8. Decision final de esta fase

Para continuar el proyecto se recomienda usar `Random Forest` como modelo principal del recomendador emocional. Es mas potente que la regresion logistica y se adapta mejor a combinaciones de variables musicales.

La regresion logistica se conserva como modelo base porque ayuda a comparar y explicar que aporta un modelo mas complejo.
