# Ajuste del interprete de actividad

Este documento resume el afinado del interprete de actividad y la interfaz, con lenguaje sencillo.

## 1. Problema detectado

Habia frases que se clasificaban mal. Por ejemplo:

- "voy a comer" se interpretaba como entrenamiento intenso.
- Algunas actividades nuevas no caian en la familia correcta.

## 2. Que se cambio

Se aplico un enfoque hibrido:

- Se mantiene el modelo de texto para entender frases libres.
- Se anaden reglas por palabras clave para corregir casos claros.
- Si el modelo tiene baja confianza, se elige la etiqueta mas cercana al perfil musical.

Tambien se anadio una nueva familia:

- alimentacion (comer, almorzar, cenar, desayunar, merendar)

## 3. Resultado esperado

La actividad se interpreta de forma mas estable y coherente con frases reales:

- "voy a comer" -> alimentacion
- "tengo que estudiar" -> estudio_trabajo
- "quiero fregar el suelo" -> limpieza_domestica

Esto evita tener que escribir todas las actividades posibles a mano.

## 4. Cambios en la interfaz

Se incorporo el banner de la carpeta images y un estilo mas visual:

- Cabecera con imagen.
- Tarjetas mas grandes y legibles.
- Boton directo a Spotify.

## 5. Comprobaciones

Se entreno de nuevo el interprete y se pasaron pruebas basicas con casos clave.

Si quieres repetirlo:

```powershell
.\.venv\Scripts\python.exe scripts\train_activity_text_model.py
.\.venv\Scripts\python.exe scripts\check_recommender_cases.py
.\.venv\Scripts\python.exe app.py
```

## 6. Limitacion actual

Aunque el interprete ya es mas fino, el mejor ajuste real se consigue con feedback de usuarios (aceptar o rechazar canciones). Ese seria el siguiente paso natural.
