# red-neuronal-ciudades

Práctica del curso **Introducción a Deep Learning**: una red neuronal escrita desde cero en numpy
que clasifica puntos GPS (lat, lon) en dos ciudades, la misma red hecha con TensorFlow/Keras, y
su despliegue: Flask + Docker + Kubernetes (kind) para numpy, y TensorFlow Serving para Keras.

> Guía rápida de comandos para probar todo después de prender la PC:
> [`entrenamiento/COMO_PROBAR.md`](entrenamiento/COMO_PROBAR.md).

## Ciudades

| Clase | Ciudad | Centro (lat, lon) | Datos |
|---|---|---|---|
| 0 | Orizaba, México | (18.85, −97.10) | `dataset/orizaba.csv` (300 puntos) |
| 1 | París, Francia | (48.86, 2.35) | `dataset/paris.csv` (300 puntos) |

Los dos CSV tienen las columnas `lat,lon,clase`. Los scripts de `entrenamiento/` toman la clase de
la columna `clase`.

## Configuración (red en numpy)

| Parámetro | Valor |
|---|---|
| Puntos por clase | 300 (600 en total), de los CSV |
| lr | 0.01 |
| Épocas | 2 |
| Arquitectura | [2, 4, 8, 1] |
| Activaciones | relu, relu, sigmoid |
| Inicialización | W = 0.75, b = 0.25 |
| Reinicio por época | Sí: cada época arranca otra vez con W = 0.75 y b = 0.25 (`construir_red`), como en el Excel de referencia |
| Error | MSE |

Como los pesos se reinician en cada época, las dos épocas dan exactamente el mismo resultado.

## Archivos

| Archivo | Contenido |
|---|---|
| `dataset/orizaba.csv`, `dataset/paris.csv` | Puntos de entrenamiento (lat, lon, clase). |
| `entrenamiento/neural_network_numpy.py` | Red en numpy desde cero. Lee los CSV, entrena y escribe `entrenamiento/modelo.npz`. Con `streamlit run` muestra el mapa. |
| `entrenamiento/gps_tensorflow_keras.py` | La misma arquitectura con Keras (Adam, 300 épocas). Guarda `entrenamiento/gps_classifier.keras` y exporta `entrenamiento/gps-classifier/1/` para TF Serving. |
| `entrenamiento/tensorflow-20.py` | La red del Excel (20 puntos fijos) con TensorFlow y `GradientTape`. Salida legible; `--json` la da en JSON. |
| `entrenamiento/Dockerfile.serving` | Imagen de TF Serving (Bitnami, arm64) con el modelo Keras de Orizaba/París. |
| `entrenamiento/COMO_PROBAR.md` | Guía de comandos para correr y probar todo. |
| `modelo.npz` | Modelo numpy que usa el servidor Flask (`W0`, `b0`, `W1`, `b1`, `W2`, `b2` y `neuronas`). Para servir el nuevo, copia `entrenamiento/modelo.npz` aquí. |
| `gps_tensorflow_keras.py`, `gps_classifier.keras`, `gps-classifier/`, `Dockerfile.serving` | Versión anterior del clasificador Keras, entrenada con `dataset/city1.csv` y `dataset/city2.csv`. |
| `app/server.py` | Servidor Flask: `GET /health` y `POST /predict`. |
| `app/requirements.txt` | Dependencias del servidor (Flask, gunicorn, numpy). Las usa el Dockerfile. |
| `linear_model.py` | Cliente: manda pares (lat, lon) a `/predict`. Solo biblioteca estándar. |
| `Dockerfile` | Imagen `python:3.12-slim` con el servidor y `modelo.npz`, servida con gunicorn en el puerto 8080. |
| `k8s/kind-config.yaml` | Clúster kind: 1 control-plane y 2 workers; publica el puerto 30080 en el host. |
| `k8s/deployment.yaml` | Deployment `red-gps`: 2 réplicas, readinessProbe en `/health`. |
| `k8s/service.yaml` | Service NodePort en el puerto 30080. |
| `requirements.txt` | Versiones exactas para el entorno local (red, servidor y gráficas). |

### Resultados (no se versionan)

Los resultados de las corridas están en la carpeta `resultados/` en la raíz del repo. Esa carpeta
está en `.gitignore`, así que **no se sube a git**; cada quien la genera en su máquina.

| Archivo | Contenido |
|---|---|
| `resultados/resultados_corrida.md` | Parámetros, caso A (validación) y caso B (entrenamiento) por época: forward, backprop, pesos y sesgos nuevos, MSE y predicciones. |
| `resultados/prueba_cliente.txt` | Salida de la prueba local del servidor y el cliente. |

## 1. Local

### Entorno virtual

Todo se ejecuta dentro de `.venv` (no se versiona; está en `.gitignore`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Validar y entrenar

Desde la raíz del repo:

```bash
# Validación: 20 puntos fijos del Excel, lr = 0.01, 2 épocas (no escribe archivos)
python entrenamiento/neural_network_numpy.py --validar
python entrenamiento/tensorflow-20.py           # lo mismo con TensorFlow
#    En las dos épocas la última capa debe dar W = 0.74906805 y b = 0.24981331.

# Entrenamiento numpy: 300 + 300 puntos de los CSV, lr = 0.01, 2 épocas
python entrenamiento/neural_network_numpy.py
#    Escribe entrenamiento/modelo.npz.

# Mapa de los puntos en Streamlit (puerto 8502 para no chocar con TF Serving)
streamlit run entrenamiento/neural_network_numpy.py --server.port 8502

# Entrenamiento Keras: 300 + 300 puntos de los CSV, Adam, 300 épocas
python entrenamiento/gps_tensorflow_keras.py
#    Escribe entrenamiento/gps_classifier.keras y entrenamiento/gps-classifier/1/.
```

Los modelos entrenados ya vienen en el repo; solo hace falta volver a entrenar si cambia la red.

### TensorFlow Serving (modelo Keras)

```bash
docker build -f entrenamiento/Dockerfile.serving -t alessandro117/gps-orizaba-paris .
docker run -d --name my_serving_base -p 8501:8501 alessandro117/gps-orizaba-paris

curl http://localhost:8501/v1/models/gps-classifier          # "state": "AVAILABLE"
curl -X POST http://localhost:8501/v1/models/gps-classifier:predict \
     -H 'Content-Type: application/json' \
     -d '{"instances": [[18.85, -97.10], [48.86, 2.35]]}'
#    ~0 = Orizaba (clase 0), ~1 = París (clase 1)
```

### Servidor

```bash
python app/server.py                                          # Flask, puerto 8080
# o igual que en Docker:
gunicorn --bind 0.0.0.0:8080 --workers 2 --chdir app server:app
```

Endpoints:

- `GET /health`: estado, arquitectura, activaciones, clases y umbral.
- `POST /predict`: body `{"puntos": [[lat, lon], ...]}`. Por cada punto responde
  `probabilidad_clase_1` (la salida de la sigmoide), `clase` (umbral 0.5: 0 = Orizaba, 1 = París)
  y `ciudad`. Las entradas **no** se normalizan, igual que en el entrenamiento.

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
     -H 'Content-Type: application/json' \
     -d '{"puntos": [[18.85, -97.10], [48.86, 2.35]]}'
```

## 2. Cliente

```bash
python linear_model.py 0.1 1.0 2.71 3.14
```

Los números se leen como pares (lat, lon): `(0.1, 1.0)` y `(2.71, 3.14)`. El cliente llama a
`POST /predict` en la URL de la variable de entorno `MODEL_URL`
(por defecto `http://localhost:8080/predict`) e imprime, por par, la probabilidad de la clase 1 y la
clase predicha. Para otro servidor:

```bash
MODEL_URL=http://localhost:30080/predict python linear_model.py 0.1 1.0 2.71 3.14
```

## 3. Docker

```bash
docker build -t red-gps:latest .
docker run --rm -p 8080:8080 red-gps:latest
python linear_model.py 0.1 1.0 2.71 3.14
```

## 4. Kubernetes con kind

```bash
kind create cluster --config k8s/kind-config.yaml       # clúster "red-gps": 1 control-plane + 2 workers
docker build -t red-gps:latest .
kind load docker-image red-gps:latest --name red-gps
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/red-gps
kubectl get pods -o wide                                  # 2 réplicas listas

MODEL_URL=http://localhost:30080/predict python linear_model.py 0.1 1.0 2.71 3.14
```

El puerto 30080 del NodePort queda publicado en el host por `extraPortMappings` en
`k8s/kind-config.yaml`. Para borrar el clúster: `kind delete cluster --name red-gps`.

## Resultado del entrenamiento

### Red en numpy (`dataset/orizaba.csv` + `dataset/paris.csv`)

| Época | W[0] última capa | b última capa | MSE | Exactitud |
|---|---|---|---|---|
| 0 | 0.66940789 | 0.24946272 | 0.5 | 0.5 |
| 1 | 0.66940789 | 0.24946272 | 0.5 | 0.5 |

Con lr = 0.01, pesos iniciales iguales y reinicio en cada época, la red todavía no separa las
clases: todos los puntos se clasifican como clase 1 (París). El servidor Flask (con el
`modelo.npz` de la raíz, de la versión anterior) también responde clase 1 para los centros de las
dos ciudades (Orizaba da probabilidad 0.84).

### Red en Keras (mismos CSV)

Adam (lr = 0.01), 300 épocas, pesos aleatorios: **exactitud = 1.0**. Orizaba da probabilidad
~1.6e-08 y París ~0.9999999, tanto en local como en TF Serving.
