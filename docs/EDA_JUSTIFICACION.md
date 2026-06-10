# Justificacion del EDA (Silver en S3)

Este documento describe el proposito y utilidad de cada bloque del EDA realizado sobre la capa Silver en S3. Se redacta en tono tecnico para ser reutilizado en la memoria final del proyecto.

## 1. Comprension general del dataset

**Analisis**
- Vista de las primeras filas, shape, tipos de datos y porcentaje de nulos.
- Descripcion funcional de columnas.

**Por que se realiza**
- Permite validar que los datos provienen de Silver y que las transformaciones previas fueron aplicadas correctamente.
- Identifica columnas con problemas de calidad o tipos inconsistentes.

**Utilidad en el proyecto**
- Evita decisiones de modelado basadas en suposiciones incorrectas.
- Ayuda a detectar variables que requieren tratamiento posterior (sin realizarlo en esta fase).

**Informacion relevante**
- Tamano efectivo del dataset.
- Columnas con altos niveles de nulos.

**Impacto posterior**
- Define si se necesitan estrategias de imputacion o exclusion en fases de preparacion.

## 1.1 Estadisticas descriptivas completas

**Analisis**
- Media, mediana, moda, desviacion estandar, minimos, maximos y percentiles.
- Resumen de variables categoricas cuando aplica.

**Por que se realiza**
- Resume la distribucion numerica y detecta valores extremos y dispersion.

**Utilidad en el proyecto**
- Facilita la comparacion entre datasets y la deteccion de sesgos.

**Informacion relevante**
- Rango real de cada feature, dispersion y posicion central.

**Impacto posterior**
- Orienta transformaciones futuras (p. ej., log, normalizacion) sin ejecutarlas en esta fase.

## 1.2 Calidad de datos (nulos y cardinalidad)

**Analisis**
- Porcentaje de nulos por columna.
- Cardinalidad de variables categoricas.

**Por que se realiza**
- Identifica columnas que pueden afectar el modelado por ausencia de datos o alta cardinalidad.

**Utilidad en el proyecto**
- Prioriza acciones de limpieza en fases posteriores.

**Informacion relevante**
- Columnas con nulos relevantes.
- Variables con cardinalidad alta.

**Impacto posterior**
- Determina si se requieren tecnicas de imputacion o codificacion avanzada.

## 2. Distribucion de la variable objetivo (mood)

**Analisis**
- Frecuencias y proporciones por clase.
- Metricas de balance: ratio de desbalance, entropia y Gini.

**Por que se realiza**
- El desbalance condiciona la eleccion de metricas y la estrategia de validacion.

**Utilidad en el proyecto**
- Permite anticipar si se necesitan tecnicas de reequilibrio en fases posteriores.

**Informacion relevante**
- Clase mayoritaria y minoritaria.
- Grado de desbalance cuantificado.

**Impacto posterior**
- Define el uso de metricas por clase (precision/recall/F1 por etiqueta).

## 3. Distribucion de features acusticas

**Analisis**
- Histogramas y boxplots por feature.
- Metricas de asimetria (skew) y outliers (IQR).

**Por que se realiza**
- Permite identificar distribuciones con sesgo y presencia de valores extremos.

**Utilidad en el proyecto**
- Ayuda a seleccionar transformaciones futuras y a entender la calidad de los datos acusticos.

**Informacion relevante**
- Features con skew alto y outliers relevantes.

**Impacto posterior**
- Indica si conviene aplicar transformaciones de escala o manejo de outliers.

## 4. Relacion feature vs mood

**Analisis**
- Boxplots por clase.
- Ranking de features por varianza explicada ($\eta^2$).

**Por que se realiza**
- Evalua que variables discriminan mejor entre emociones.

**Utilidad en el proyecto**
- Orienta la seleccion de features y la interpretabilidad del modelo.

**Informacion relevante**
- Features con mayor separacion entre clases.

**Impacto posterior**
- Permite priorizar variables para modelado y reducir ruido.

## 5. Correlacion entre features

**Analisis**
- Heatmap correlacional.
- Lista de correlaciones altas para identificar redundancias.

**Por que se realiza**
- Evita multicolinealidad que pueda afectar modelos lineales o interpretabilidad.

**Utilidad en el proyecto**
- Identifica variables potencialmente redundantes.

**Informacion relevante**
- Pares con correlacion alta y posible solapamiento informativo.

**Impacto posterior**
- Ayuda a decidir reduccion de variables o regularizacion.

## 6. Comparativa Mood vs Tracks (dataset shift)

**Analisis**
- Comparativa visual con KDE.
- Metricas cuantitativas con Cohen d.

**Por que se realiza**
- Detecta diferencias de distribucion entre dataset etiquetado y catalogo completo.

**Utilidad en el proyecto**
- Anticipa riesgo de generalizacion del modelo a datos reales de recomendacion.

**Informacion relevante**
- Features con mayor shift entre datasets.

**Impacto posterior**
- Puede motivar ajustes de entrenamiento o calibracion.

## 7. Conclusiones

**Analisis**
- Sintesis de hallazgos clave a partir de las metricas anteriores.

**Por que se realiza**
- Deja trazabilidad entre resultados del EDA y decisiones futuras del pipeline.

**Utilidad en el proyecto**
- Justifica tecnicamente las siguientes fases sin adelantarse a ellas.

**Informacion relevante**
- Features discriminativas, riesgos de desbalance, outliers y dataset shift.

**Impacto posterior**
- Base para planificar la preparacion del dataset y el modelado.
