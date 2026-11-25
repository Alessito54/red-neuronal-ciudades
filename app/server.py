# Servidor Flask que sirve la red GPS entrenada por red_neuronal.py.
# Clase 0 = Orizaba, México. Clase 1 = París, Francia.
#
# Local:   python app/server.py              (puerto 8080, o PORT)
# Docker:  gunicorn --bind 0.0.0.0:8080 server:app

import math
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

RUTA_MODELO = Path(os.environ.get("MODELO_PATH", Path(__file__).resolve().parent.parent / "modelo.npz"))
CLASES = {0: "Orizaba", 1: "París"}
UMBRAL = 0.5


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cargar_modelo(ruta):
    m = np.load(ruta)
    capas = [(m[f"W{i}"], m[f"b{i}"]) for i in range(3)]
    neuronas = [int(n) for n in m["neuronas"]]
    return capas, neuronas


CAPAS, NEURONAS = cargar_modelo(RUTA_MODELO)
ACTIVACIONES = [relu, relu, sigmoid]

app = Flask(__name__)
app.json.ensure_ascii = False


def forward(X):
    # Mismo forward que entrenamiento() en red_neuronal.py; las entradas van sin normalizar.
    a = X
    for (W, b), f in zip(CAPAS, ACTIVACIONES):
        a = f(a @ W + b)
    return a


@app.get("/health")
def health():
    return jsonify({
        "estado": "ok",
        "modelo": RUTA_MODELO.name,
        "arquitectura": NEURONAS,
        "activaciones": ["relu", "relu", "sigmoid"],
        "clases": {str(k): v for k, v in CLASES.items()},
        "umbral": UMBRAL,
    })


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    puntos = body.get("puntos") if isinstance(body, dict) else None
    if not isinstance(puntos, list) or not puntos:
        return jsonify({"error": 'Se espera JSON {"puntos": [[lat, lon], ...]} con al menos un punto'}), 400
    for i, p in enumerate(puntos):
        if (not isinstance(p, list) or len(p) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p)):
            return jsonify({"error": f"El punto {i} debe ser [lat, lon] con dos números"}), 400

    X = np.array(puntos, dtype=float)
    probs = forward(X).ravel()
    resultados = []
    for (lat, lon), p in zip(puntos, probs):
        clase = int(p >= UMBRAL)
        resultados.append({
            "lat": lat,
            "lon": lon,
            "probabilidad_clase_1": float(p),
            "clase": clase,
            "ciudad": CLASES[clase],
        })
    return jsonify({"predicciones": resultados})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
