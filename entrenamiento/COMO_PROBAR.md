# Cómo probar cada cosa (después de prender la PC)

## 0. Siempre primero

```bash
source ~/.zshrc && red              # activa el entorno (te manda a ~/nn_scratch)
cd ~/Documents/prediccion_red       # regresa al proyecto
```

## 1. Red en numpy (datos en la terminal)

```bash
python entrenamiento/neural_network_numpy.py             # 300 + 300 puntos de los CSV
python entrenamiento/neural_network_numpy.py --validar   # 20 puntos del Excel
```

Con `--validar` la última capa debe dar `W = 0.74906805` y `b = 0.24981331`.

## 2. Mapa en Streamlit

```bash
streamlit run entrenamiento/neural_network_numpy.py --server.port 8502
```

Abre http://localhost:8502. Usa el puerto **8502** para no chocar con TF Serving (8501).
Se detiene con `Ctrl+C`.

## 3. Red en TensorFlow/Keras (entrenar y exportar)

```bash
python entrenamiento/gps_tensorflow_keras.py
```

Guarda `entrenamiento/gps_classifier.keras` y `entrenamiento/gps-classifier/1/` (para TF Serving).

## 4. TF Serving en Docker

1. Abre **Docker Desktop** y espera a que arranque.
2. Prende el contenedor (ya existe, solo está apagado):

```bash
docker start my_serving_base
docker ps                            # debe salir my_serving_base con 0.0.0.0:8501->8501
```

Si no existe (`No such container`), créalo con la imagen local:

```bash
docker run -d --name my_serving_base -p 8501:8501 alessandro117/gps-orizaba-paris
```

Si reentrenaste el modelo, vuelve a construir la imagen antes:

```bash
docker build -f entrenamiento/Dockerfile.serving -t alessandro117/gps-orizaba-paris .
docker rm -f my_serving_base
docker run -d --name my_serving_base -p 8501:8501 alessandro117/gps-orizaba-paris
```

## 5. Probar en Postman (o curl)

**Estado del modelo** (sin body):

- `GET http://localhost:8501/v1/models/gps-classifier`
- Debe decir `"state": "AVAILABLE"`.

**Predicción**:

- `POST http://localhost:8501/v1/models/gps-classifier:predict`
- Body → raw → JSON:

```json
{"instances": [[18.85, -97.10], [48.86, 2.35]]}
```

- Respuesta: cerca de 0 = Orizaba (clase 0), cerca de 1 = París (clase 1).

Lo mismo con curl:

```bash
curl http://localhost:8501/v1/models/gps-classifier
curl -X POST http://localhost:8501/v1/models/gps-classifier:predict \
     -H 'Content-Type: application/json' \
     -d '{"instances": [[18.85, -97.10], [48.86, 2.35]]}'
```

## Si algo falla

| Error | Causa | Arreglo |
|---|---|---|
| `can't open file .../nn_scratch/...` | Estás en otra carpeta | `cd ~/Documents/prediccion_red` |
| `Method Not Allowed` o sale HTML de Streamlit | Streamlit ocupa el 8501 | Detén Streamlit (`Ctrl+C`) y `docker restart my_serving_base` |
| `ECONNREFUSED 127.0.0.1:8501` | El contenedor está apagado o no tomó el puerto | `docker start my_serving_base` o `docker restart my_serving_base` |
| `Cannot connect to the Docker daemon` | Docker Desktop apagado | Abre Docker Desktop |
| Ver qué ocupa el 8501 | | `lsof -nP -iTCP:8501 -sTCP:LISTEN` |
