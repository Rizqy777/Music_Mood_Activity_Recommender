# Validacion y representacion grafica del modelo

Este documento cubre los puntos 6 y 7 del proyecto. La validacion se realiza con el mismo script de entrenamiento:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

El script entrena los modelos, ajusta hiperparametros con `GridSearchCV`, evalua en test y genera graficas.

## 1. Reproducibilidad

Se usa una semilla fija:

```text
RANDOM_STATE = 42
```

Esto hace que la validacion cruzada y los modelos con aleatoriedad den resultados estables entre ejecuciones. Puede haber pequenas diferencias si cambia la version de librerias, pero no deberia cambiar el comportamiento general.

Ademas, se rehizo el split de modelado para evitar fuga de informacion. El split inicial de Gold tenia duplicados exactos entre train y test: las 983 filas de test tenian una combinacion de features ya presente en train. Por eso Random Forest obtenia puntuaciones perfectas. El entrenamiento actual usa `--split-strategy deduplicated`, que elimina duplicados exactos y vuelve a crear un split estratificado.

## 2. Metricas utilizadas

Como el problema es de clasificacion multiclase, se usan estas metricas:

- Accuracy: porcentaje total de aciertos.
- Precision: de las canciones que el modelo predice como una emocion, cuantas son correctas.
- Recall: de las canciones reales de una emocion, cuantas consigue encontrar.
- F1-score: equilibrio entre precision y recall.
- ROC-AUC multiclase One-vs-Rest: mide la capacidad del modelo para separar cada clase frente al resto.
- MAE, RMSE y R2: se calculan sobre las etiquetas numericas `0-3` para cumplir la validacion pedida en el hito.

Se da importancia especial a `f1_macro`, porque calcula el rendimiento medio dando el mismo peso a todas las clases. Esto es importante porque `calm` tiene muchos menos ejemplos que `happy`.

MAE/RMSE/R2 se interpretan como metricas auxiliares en este clasificador, no como criterio principal de seleccion, porque las etiquetas representan clases discretas.

## 3. Ajuste de hiperparametros

Los dos modelos se ajustan con `GridSearchCV` usando validacion cruzada estratificada de 3 particiones.

La metrica que se optimiza es:

```text
f1_macro
```

Esto evita elegir un modelo que solo funcione bien para la clase mayoritaria.

## 4. Resultados comparados

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | ROC-AUC macro |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7900 | 0.5766 | 0.6309 | 0.5903 | 0.9452 |
| Random Forest | 0.8600 | 0.6363 | 0.6316 | 0.6331 | 0.9607 |

El mejor modelo es `Random Forest`.

La regresion logistica funciona como modelo base y obtiene resultados razonables, pero Random Forest aprende mejor las relaciones no lineales entre variables acusticas.

## 5. Seleccion del modelo final

Se selecciona:

```text
RandomForestClassifier
```

Mejores parametros:

```text
model__class_weight = "balanced_subsample"
model__max_depth = 10
model__min_samples_leaf = 3
model__n_estimators = 400
```

Motivo de la seleccion:

- Tiene mejor `f1_macro` en test.
- Tiene mejor accuracy.
- Tiene mejor ROC-AUC.
- Maneja mejor combinaciones de variables como energia, acousticness, speechiness y danceability.

La clase `calm` sigue siendo el punto debil porque solo hay 6 ejemplos unicos en todo el dataset deduplicado. Por eso la accuracy puede parecer buena, pero el `f1_macro` es mas bajo: esta metrica penaliza que una clase minoritaria no se aprenda bien.

## 6. Representacion grafica generada

Las graficas se guardan en:

```text
data_lake/model_outputs/mood_classifier/plots/
```

Graficas generadas:

- `metrics_comparison.png`: compara accuracy, precision, recall y F1 entre modelos.
- `confusion_matrix_logistic_regression.png`: matriz de confusion de Logistic Regression.
- `confusion_matrix_random_forest.png`: matriz de confusion de Random Forest.
- `predictions_vs_real_logistic_regression.png`: valores reales frente a predicciones.
- `predictions_vs_real_random_forest.png`: valores reales frente a predicciones.
- `learning_curve_random_forest.png`: curva de aprendizaje del mejor modelo.
- `feature_importance_random_forest.png`: importancia de variables del mejor modelo.

En las graficas de predicciones se diferencia claramente:

- Datos originales: etiqueta real del conjunto test.
- Predicciones del modelo: etiqueta predicha por el modelo.

## 7. Importancia de variables

Para Random Forest, las variables mas importantes han sido:

| Variable | Importancia |
|---|---:|
| energy | 0.2380 |
| danceability | 0.1718 |
| acousticness | 0.1259 |
| loudness | 0.1061 |
| speechiness | 0.1027 |

Esto tiene sentido para un recomendador musical por emociones. La energia ayuda a separar canciones activas de canciones tranquilas, acousticness y speechiness aportan informacion sobre el tipo de sonido, y danceability/loudness ayudan a distinguir canciones mas alegres o energicas.

## 8. Artefactos de validacion

Se generan estos archivos:

```text
data_lake/model_outputs/mood_classifier/metrics.csv
data_lake/model_outputs/mood_classifier/gridsearch_cv_results.csv
data_lake/model_outputs/mood_classifier/predictions.csv
data_lake/model_outputs/mood_classifier/feature_importance_random_forest.csv
```

Tambien se actualiza:

```text
models/mood_best_model.joblib
models/mood_training_summary.json
```
