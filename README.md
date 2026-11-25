# red-neuronal-ciudades

Práctica del curso **Introducción a Deep Learning**: una red neuronal escrita desde cero en numpy
que clasifica puntos GPS (lat, lon) en dos ciudades. La red se sirve con Flask, se empaqueta en
Docker y se despliega en Kubernetes (kind) y en Okteto. Un cliente en Python manda puntos al servidor.

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

## Archivos

| Archivo | Contenido |
|---|---|
| `red_neuronal.py` | La red y la generación de los puntos GPS. Al entrenar escribe `datos_entrenamiento.csv` y `modelo.npz`. |
| `datos_entrenamiento.csv` | Los 300 puntos de entrenamiento (lat, lon, clase). |
| `modelo.npz` | W y b de las tres capas después de la última época (`W0`, `b0`, `W1`, `b1`, `W2`, `b2`) y la arquitectura (`neuronas`). |
| `app/server.py` | Servidor Flask: `GET /health` y `POST /predict`. |
| `app/requirements.txt` | Dependencias del servidor (Flask, gunicorn, numpy). Las usa el Dockerfile. |
| `linear_model.py` | Cliente: manda pares (lat, lon) a `/predict`. Solo biblioteca estándar. |
| `Dockerfile` | Imagen `python:3.12-slim` con el servidor y `modelo.npz`, servida con gunicorn en el puerto 8080. |
| `k8s/kind-config.yaml` | Clúster kind: 1 control-plane y 2 workers; publica el puerto 30080 en el host. |
| `k8s/deployment.yaml` | Deployment `red-gps`: 2 réplicas, readinessProbe en `/health`. |
| `k8s/service.yaml` | Service NodePort en el puerto 30080. |
| `okteto.yml`, `k8s/okteto/ingress.yaml` | Despliegue en Okteto; el Ingress lleva `dev.okteto.com/generate-host: "true"` para que Okteto genere la URL pública. |
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

```bash
# Validación: 20 puntos fijos del Excel, lr = 0.01, 2 épocas (no escribe archivos)
python red_neuronal.py --validar
#    En las dos épocas la última capa debe dar W = 0.74906805 y b = 0.24981331.

# Entrenamiento: 150 + 150 puntos GPS, lr = 0.01, 2 épocas
python red_neuronal.py
#    Escribe datos_entrenamiento.csv (300 filas: lat, lon, clase) y modelo.npz.
#    SHA-256 esperado del CSV: 020789e3e7569bee61a15a6570e92e5c630d70346b6b83b221b3c12209802775
```

`modelo.npz` ya viene en el repo; solo hace falta volver a entrenar si cambia la red.

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

## 5. Okteto

```bash
okteto context use https://cloud.okteto.com    # o la URL de tu instancia
okteto deploy
```

`okteto deploy` construye la imagen con el `Dockerfile`, aplica `k8s/deployment.yaml`,
`k8s/service.yaml` y `k8s/okteto/ingress.yaml`, cambia la imagen del Deployment por la que
construyó Okteto (`OKTETO_BUILD_RED_GPS_IMAGE`) y espera a que esté listo. La URL pública aparece en
el dashboard de Okteto (o con `kubectl get ingress red-gps`):

```bash
MODEL_URL=https://<host-generado-por-okteto>/predict python linear_model.py 0.1 1.0 2.71 3.14
```

## Resultado del entrenamiento (caso B)

| Época | W[0] última capa | b última capa | MSE | Exactitud |
|---|---|---|---|---|
| 0 | 0.70970394 | 0.24946272 | 0.5 | 0.5 |
| 1 | 0.70970394 | 0.24946272 | 0.5 | 0.5 |

Con lr = 0.01, pesos iniciales iguales y reinicio en cada época, la red todavía no separa las
clases: todos los puntos se clasifican como clase 1 (París). Por eso el servidor también responde
clase 1 para los centros de las dos ciudades (Orizaba da probabilidad 0.84).
