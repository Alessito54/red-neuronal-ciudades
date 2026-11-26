import json
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

pred_final = forward(capas, X_t).numpy()
salida = {
    "pesos": {f"W{i}": W.numpy().tolist() for i, (W, _) in enumerate(capas)},
    "sesgos": {f"b{i}": b.numpy().tolist() for i, (_, b) in enumerate(capas)},
    "predicciones": pred_final.ravel().tolist(),
}
print(json.dumps(salida))