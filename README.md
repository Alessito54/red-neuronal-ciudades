# red-neuronal-ciudades

Práctica del curso **Introducción a Deep Learning**: una red neuronal escrita desde cero en numpy
que clasifica puntos GPS (lat, lon) en dos ciudades.

## Ciudades

| Clase | Ciudad | Centro (lat, lon) | Radio |
|---|---|---|---|
| 0 | Orizaba, México | (18.85, −97.10) | R = 2 |
| 1 | París, Francia | (48.86, 2.35) | R = 0.5 |

Los puntos son simulados alrededor de las coordenadas reales de cada ciudad. La función `circulo`
genera el desplazamiento igual que el código de referencia, lo divide entre **50** (escala /50,
como el ejemplo del PDF) y se lo suma al centro: `lat = centro_lat + y/50`, `lon = centro_lon + x/50`.

## Configuración

| Parámetro | Valor |
|---|---|
| Semilla | `np.random.seed(42)` |
| Puntos por clase | 150 (300 en total) |
| lr | 0.01 |
| Épocas | 2 |
| Arquitectura | [2, 4, 8, 1] |
| Activaciones | relu, relu, sigmoid |
| Inicialización | W = 0.75, b = 0.25 |
| Reinicio por época | Sí: cada época arranca otra vez con W = 0.75 y b = 0.25 (`construir_red`), como en el Excel de referencia |
| Error | MSE |

Como los pesos se reinician en cada época, las dos épocas dan exactamente el mismo resultado.

## Validación con el Excel de referencia

El modo `--validar` usa los 20 puntos fijos del Excel de referencia (10 por clase) con lr = 0.01
y 2 épocas, para comprobar que la lógica de la red coincide con el Excel. En las dos épocas la
última capa debe dar **W = 0.74906805** y **b = 0.24981331**.

## Entorno virtual

Todo se ejecuta dentro de `.venv` (no se versiona; está en `.gitignore`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correrlo

```bash
# 1. Validación: 20 puntos fijos del Excel, lr = 0.01, 2 épocas
python red_neuronal.py --validar

# 2. Entrenamiento: 150 + 150 puntos GPS, lr = 0.01, 2 épocas
python red_neuronal.py
#    Escribe datos_entrenamiento.csv (300 filas: lat, lon, clase).
#    SHA-256 esperado: 020789e3e7569bee61a15a6570e92e5c630d70346b6b83b221b3c12209802775
```

## Archivos

| Archivo | Contenido |
|---|---|
| `red_neuronal.py` | La red y la generación de los puntos GPS. |
| `datos_entrenamiento.csv` | Los 300 puntos de entrenamiento (generado por `red_neuronal.py`). |
| `resultados_corrida.md` | Parámetros, caso A (validación) y caso B (entrenamiento) por época: forward, backprop, pesos y sesgos nuevos, MSE y predicciones. |
| `requirements.txt` | Versiones exactas (`pip freeze`). |

`generar_entregables.py`, `resultados.xlsx` y `reporte.pdf` corresponden a la versión anterior
(puntos alrededor del origen, lr = 0.001, 1 época) y están desactualizados.

## Resultado del entrenamiento (caso B)

| Época | W[0] última capa | b última capa | MSE | Exactitud |
|---|---|---|---|---|
| 0 | 0.70970394 | 0.24946272 | 0.5 | 0.5 |
| 1 | 0.70970394 | 0.24946272 | 0.5 | 0.5 |

Con lr = 0.01, pesos iniciales iguales y reinicio en cada época, la red todavía no separa las
clases: todos los puntos se redondean a la clase 1 (París).
