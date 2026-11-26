# Red [2, 4, 8, 1] con TensorFlow sobre los 20 puntos fijos del Excel de referencia.
# Misma lógica que entrenamiento/neural_network_numpy.py --validar.
#
# Uso:
#   python entrenamiento/tensorflow-20.py          -> salida legible
#   python entrenamiento/tensorflow-20.py --json   -> salida en JSON (una línea)

import json
import sys

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

X = np.array([
    [-2.848, 5.975], [-10.429, -8.024], [-1.284, 2.872], [4.758, 5.955],
    [-3.938, -5.78], [8.525, 17.866], [-13.715, 4.789], [-3.19, 13.309],
    [3.227, 18.344], [-5.014, -17.507], [0.491, 1.897], [-0.54, 1.12],
    [-2.032, 1.092], [-1.476, -1.606], [-0.084, 3.386], [-1.618, 1.16],
    [2.238, -1.671], [-2.102, 1.924], [-1.059, 4.297], [2.642, 2.981],
])
Y = np.array([0.0] * 10 + [1.0] * 10).reshape(20, 1)
LR, EPOCAS, N = 0.01, 2, 20

X_t = tf.constant(X, dtype=tf.float64)
Y_t = tf.constant(Y, dtype=tf.float64)


def construir_red():
    return [
        [tf.Variable(np.full((2, 4), 0.75)), tf.Variable(np.full((1, 4), 0.25))],
        [tf.Variable(np.full((4, 8), 0.75)), tf.Variable(np.full((1, 8), 0.25))],
        [tf.Variable(np.full((8, 1), 0.75)), tf.Variable(np.full((1, 1), 0.25))],
    ]


def forward(capas, x):
    a = x
    for i, (W, b) in enumerate(capas):
        z = tf.matmul(a, W) + b
        a = tf.nn.relu(z) if i < 2 else tf.sigmoid(z)
    return a


COMO_JSON = "--json" in sys.argv
NOMBRES_CAPAS = ["Capa 0 (2 -> 4, relu)", "Capa 1 (4 -> 8, relu)", "Capa 2 (8 -> 1, sigmoid)"]

if not COMO_JSON:
    print("=" * 60)
    print("Red [2, 4, 8, 1] con TensorFlow (puntos del Excel)")
    print(f"lr = {LR}   épocas = {EPOCAS}   puntos = {N} (10 por clase)")
    print("Inicialización: W = 0.75, b = 0.25 (se reinicia en cada época)")
    print("=" * 60)

for epoca in range(EPOCAS):
    capas = construir_red()
    with tf.GradientTape() as tape:
        pred = forward(capas, X_t)
        perdida = 0.5 * tf.reduce_sum((pred - Y_t) ** 2)
    variables = [v for par in capas for v in par]
    grads = tape.gradient(perdida, variables)
    for k, (W, b) in enumerate(capas):
        W.assign_sub(LR * grads[2 * k])
        b.assign_sub(LR * grads[2 * k + 1] / N)
    if not COMO_JSON:
        W_ultima = capas[-1][0].numpy().ravel()[0]
        b_ultima = capas[-1][1].numpy().ravel()[0]
        print(f"Época {epoca}: pérdida = {float(perdida):.8f}   "
              f"última capa W = {W_ultima:.8f}, b = {b_ultima:.8f}")

pred_final = forward(capas, X_t).numpy()

if COMO_JSON:
    salida = {
        "pesos": {f"W{i}": W.numpy().tolist() for i, (W, _) in enumerate(capas)},
        "sesgos": {f"b{i}": b.numpy().tolist() for i, (_, b) in enumerate(capas)},
        "predicciones": pred_final.ravel().tolist(),
    }
    print(json.dumps(salida))
    sys.exit()

clases = np.round(pred_final)
mse = float(np.mean((clases - Y) ** 2))
exactitud = float(np.mean(clases == Y))

print("\n--- Predicciones ---")
print(f"{'#':>3}  {'x1':>8}  {'x2':>8}  {'real':>4}  {'salida':>10}  {'clase':>5}")
for i, ((x1, x2), real, p, c) in enumerate(zip(X, Y.ravel(), pred_final.ravel(), clases.ravel())):
    print(f"{i:>3}  {x1:>8.3f}  {x2:>8.3f}  {int(real):>4}  {p:>10.6f}  {int(c):>5}")

print("\n--- Resultado ---")
print(f"MSE (predicciones redondeadas) = {mse}")
print(f"Exactitud = {exactitud}")

print("\n--- Pesos y sesgos finales ---")
for nombre, (W, b) in zip(NOMBRES_CAPAS, capas):
    print(f"\n{nombre}")
    print("W =")
    print(np.array2string(W.numpy(), precision=8, suppress_small=True, max_line_width=120))
    print("b =")
    print(np.array2string(b.numpy(), precision=8, suppress_small=True, max_line_width=120))