
## Objetivo

Realizar un Análisis Exploratorio de Datos profundo y útil orientado a comprender la viabilidad del problema de clasificación emocional musical.

El EDA debe generar hallazgos relevantes para justificar decisiones posteriores de preparación de datos y modelado.

NO se aceptan gráficos o estadísticas puramente decorativos sin interpretación útil.

---

## Fuente de Datos Obligatoria

El EDA debe leer los datasets exclusivamente desde la capa Silver almacenada en S3.

NO leer directamente desde los CSV originales de `datasets/`.
NO leer desde Bronze.
NO usar copias locales intermedias salvo que sean generadas automáticamente desde la lectura de S3 por necesidades técnicas.

El objetivo es analizar los datos procesados reales producidos por el pipeline de datos.

## Enfoque Analítico Obligatorio

Todo análisis debe responder preguntas relevantes para el problema de negocio/modelado.

Cada gráfico o tabla debe acompañarse de interpretación práctica.

---

## Análisis Obligatorios

### 1. Comprensión General del Dataset

- Primeras 5 filas
- Shape
- Tipos de datos
- Nulos
- Descripción funcional de columnas

---

### 2. Distribución de la Variable Objetivo

Analizar el mood target:

- Frecuencia por clase
- Balance / desbalance de clases
- Implicaciones para modelado

---

### 3. Distribución de Features Acústicas

Para variables clave:

- Histogramas
- Boxplots
- Análisis de skew / outliers

Interpretar:
- Qué variables parecen informativas
- Qué distribuciones podrían requerir transformación

---

### 4. Relación Feature vs Mood

Analizar cómo cambia cada feature relevante según mood:

- Boxplots agrupados por mood
- Violinplots opcionales
- Comparativas visuales entre clases

Objetivo:
Determinar qué variables discriminan mejor entre emociones.

---

### 5. Correlación Entre Features

- Heatmap correlacional
- Detectar redundancias / multicolinealidad

Interpretar:
- Features potencialmente redundantes
- Impacto esperado en modelos posteriores

---

### 6. Comparativa Mood Dataset vs Tracks Dataset

Comparar distribuciones entre ambos datasets para features compartidas:

Objetivo:
Detectar dataset shift / diferencias de distribución.

Interpretar:
- Riesgo de generalización del modelo al catálogo completo

---

### 7. Conclusiones del EDA

Generar conclusiones accionables:

- Features más prometedoras
- Problemas detectados
- Transformaciones potencialmente necesarias en preparación
- Riesgos de modelado identificados

---

## Restricciones

NO realizar todavía:

- Feature engineering
- Escalado
- Encoding
- Modelado
- Train/test split

---

## Entregable

Notebook:

`notebooks/eda.ipynb`

Debe parecer un análisis académico argumentado, no un dashboard automático.