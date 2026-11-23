# prediccion_red

Práctica del curso **Introducción a Deep Learning**: una red neuronal escrita desde cero en numpy
que clasifica coordenadas en dos clases.

| Clase | Etiqueta | Radio |
|---|---|---|
| 0 | Orizaba, México | R = 2 |
| 1 | París, Francia | R = 0.5 |

Las ciudades son solo etiquetas: los puntos se generan alrededor del origen con la función
`circulo` del código de referencia (no hay coordenadas geográficas ni normalización).

## Configuración

| Parámetro | Valor |
|---|---|
| Semilla | `np.random.seed(42)` |
| Puntos por clase | 150 (300 en total) |
| lr | 0.001 |
| Arquitectura | [2, 4, 8, 1] |
| Activaciones | relu, relu, sigmoid |
| Inicialización | W = 0.75, b = 0.25 |
| Épocas | 1 (`range(0, 1)`, igual que el código de referencia) |

### Diferencias con el Excel de referencia

El Excel de referencia (`neural_network_numpy.xlsx`) usa **20 puntos fijos (10 por clase) y lr = 0.01**.
Este proyecto entrena con **150 + 150 puntos generados y lr = 0.001**. La lógica de la red es la misma;
el modo `--validar` reproduce exactamente el caso del Excel para comprobarlo.

## Entorno virtual

Todo se ejecuta dentro de `.venv` (no se versiona; está en `.gitignore`).

```bash
cd prediccion_red
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correrlo

```bash
# 1. Validación: 20 puntos fijos del Excel, lr = 0.01
python red_neuronal.py --validar
#    La última capa debe dar W = 0.74906805 y b = 0.24981331 (igual que el Excel).

# 2. Entrenamiento real: 150 + 150 puntos, lr = 0.001
python red_neuronal.py
#    Escribe datos_entrenamiento.csv (300 filas: x, y, clase).

# 3. Entregables
python generar_entregables.py
#    Escribe resultados.xlsx y reporte.pdf (y vuelve a escribir datos_entrenamiento.csv).
```

`generar_entregables.py` no reimplementa la red: ejecuta `red_neuronal.py` (con y sin `--validar`),
usa sus propias funciones y repite la secuencia de `entrenamiento` guardando cada paso intermedio.
Antes de escribir nada comprueba que la validación coincida con el Excel y que los pesos, sesgos,
predicciones y MSE de la traza sean idénticos a los de `red_neuronal.py`; si no, se detiene.

## Archivos

| Archivo | Contenido |
|---|---|
| `red_neuronal.py` | La red (lógica del código de referencia, sin cambios). |
| `generar_entregables.py` | Genera `resultados.xlsx` y `reporte.pdf` desde la corrida real. |
| `datos_entrenamiento.csv` | Los 300 puntos de entrenamiento (generado). |
| `resultados.xlsx` | Hoja por época con la misma estructura del Excel de referencia (forward; backprop con derivadas, Delta, Wtemp, W/b nuevos), más hojas de predicciones y MSE, pesos finales y validación N=20. Los valores salen de la corrida, no están escritos a mano. |
| `reporte.pdf` | Los mismos pasos que el Excel (las matrices de 300 filas se muestran con las 5 primeras y 5 últimas filas), gráfica de dispersión de los 300 puntos por ciudad, MSE, las 300 predicciones y pesos finales. |
| `requirements.txt` | Versiones exactas (`pip freeze`). |

## Resultado de la corrida real

Con una sola época, lr = 0.001 y todos los pesos iniciales iguales, la red todavía no separa las clases:
la salida sigmoide queda entre ≈0.85 y 1.0 para todos los puntos, así que todos se redondean a la
clase 1 (París). Exactitud = 0.5 y MSE (predicción redondeada) = 0.5. Esto es lo que produce el
código tal como está; no se cambió la semilla, el lr ni el número de épocas.
