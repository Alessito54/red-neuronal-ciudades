# Cliente del servidor de la red GPS (solo biblioteca estándar).
#
# Uso:
#   python linear_model.py 0.1 1.0 2.71 3.14
# Los números se leen como pares (lat, lon): (0.1, 1.0) y (2.71, 3.14).
# URL del servidor: variable de entorno MODEL_URL (por defecto http://localhost:8080/predict).

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8080/predict")


def leer_pares(args):
    if not args or len(args) % 2 != 0:
        sys.exit("Uso: python linear_model.py lat1 lon1 [lat2 lon2 ...] (cantidad par de números)")
    try:
        numeros = [float(a) for a in args]
    except ValueError as e:
        sys.exit(f"Argumento no numérico: {e}")
    return [numeros[i:i + 2] for i in range(0, len(numeros), 2)]


def predecir(puntos):
    datos = json.dumps({"puntos": puntos}).encode("utf-8")
    req = urllib.request.Request(MODEL_URL, data=datos, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"Error {e.code} del servidor: {e.read().decode('utf-8', 'replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"No se pudo conectar a {MODEL_URL}: {e.reason}")


def main():
    puntos = leer_pares(sys.argv[1:])
    respuesta = predecir(puntos)
    print(f"Servidor: {MODEL_URL}")
    for p in respuesta["predicciones"]:
        print(f"(lat={p['lat']}, lon={p['lon']}) -> probabilidad clase 1 = {p['probabilidad_clase_1']:.8f}, "
              f"clase = {p['clase']} ({p['ciudad']})")


if __name__ == "__main__":
    main()
