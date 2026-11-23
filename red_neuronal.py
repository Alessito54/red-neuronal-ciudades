# Red neuronal en numpy desde cero, entrenada con coordenadas.
# Clase 0 = Orizaba, México (R = 2). Clase 1 = París, Francia (R = 0.5).
# Basado en el código de referencia "excel demo"; la lógica de la red no cambia.
#
# Uso:
#   python red_neuronal.py            -> 150 + 150 puntos, lr = 0.001
#   python red_neuronal.py --validar  -> 20 puntos fijos del Excel, lr = 0.01 (prueba de validación)

import math
import sys

import numpy as np
from scipy import stats

VALIDAR = "--validar" in sys.argv

np.random.seed(42)

sigmoid = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x * (1 - x),
)


def derivada_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


relu = (
    lambda x: x * (x > 0),
    lambda x: derivada_relu(x),
)


def circulo(num_datos=100, R=1, minimo=0, maximo=1):
    pi = math.pi
    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10
    x = np.cos(theta) * r
    y = np.sin(theta) * r

    y = y.reshape((num_datos, 1))
    x = x.reshape((num_datos, 1))

    # Se reduce la precisión para no causar overflow
    x = np.round(x, 3)
    y = np.round(y, 3)

    df = np.column_stack([x, y])
    return df


def mse(Ypredich, Yreal):
    x = (np.array(Ypredich) - np.array(Yreal)) ** 2
    x = np.mean(x)
    y = np.array(Ypredich) - np.array(Yreal)
    return (x, y)


def entrenamiento(epoch, X, Y, red_neuronal, lr=0.01):
    # output guarda el resultado de cada capa
    output = [X]
    for num_capa in range(len(red_neuronal)):
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
        a = red_neuronal[num_capa].funcion_act[0](z)
        output.append(a)

    # Backpropagation
    back = list(range(len(output) - 1))
    back.reverse()

    delta = []

    for capa in back:
        a = output[capa + 1]

        if capa == back[0]:
            x = mse(a, Y)[1] * red_neuronal[capa].funcion_act[1](a)
            delta.append(x)
        else:
            x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
            delta.append(x)

        W_temp = red_neuronal[capa].W.transpose()

        # Descenso de gradiente
        red_neuronal[capa].b = red_neuronal[capa].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        red_neuronal[capa].W = red_neuronal[capa].W - output[capa].transpose() @ delta[-1] * lr

    return output[-1]


class capa:
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.full((1, n_neuronas), 0.25).reshape(1, n_neuronas)
        self.W = np.full((1, n_neuronas * n_neuronas_capa_anterior), 0.75).reshape(n_neuronas_capa_anterior, n_neuronas)


if VALIDAR:
    # 20 puntos fijos del Excel de referencia (10 por clase)
    N = 10
    LR = 0.01
    X = np.array([
        [-2.848, 5.975], [-10.429, -8.024], [-1.284, 2.872], [4.758, 5.955],
        [-3.938, -5.78], [8.525, 17.866], [-13.715, 4.789], [-3.19, 13.309],
        [3.227, 18.344], [-5.014, -17.507], [0.491, 1.897], [-0.54, 1.12],
        [-2.032, 1.092], [-1.476, -1.606], [-0.084, 3.386], [-1.618, 1.16],
        [2.238, -1.671], [-2.102, 1.924], [-1.059, 4.297], [2.642, 2.981],
    ])
else:
    # 150 puntos por ciudad
    N = 150
    LR = 0.001
    datos_1 = circulo(num_datos=N, R=2)     # clase 0: Orizaba
    datos_2 = circulo(num_datos=N, R=0.5)   # clase 1: París
    X = np.concatenate([datos_1, datos_2])
    X = np.round(X, 3)

Y = [0] * N + [1] * N
Y = np.array(Y).reshape(len(Y), 1)

neuronas = [2, 4, 8, 1]
funciones_activacion = [relu, relu, sigmoid]
red_neuronal = []

for paso in range(len(neuronas) - 1):
    red_neuronal.append(capa(neuronas[paso], neuronas[paso + 1], funciones_activacion[paso]))

error = []
predicciones = []

for epoch in range(0, 1):
    ronda = entrenamiento(epoch, X=X, Y=Y, red_neuronal=red_neuronal, lr=LR)
    predicciones.append(ronda)
    temp = mse(np.round(predicciones[-1]), Y)[0]
    error.append(temp)

print(f"lr = {LR}, puntos por clase = {N}")
print("Error MSE por época:", error)
print("Predicciones clase 0:", predicciones[-1][:N].ravel().round(4))
print("Predicciones clase 1:", predicciones[-1][N:].ravel().round(4))
print("Exactitud:", np.mean(np.round(predicciones[-1]) == Y))

for i, c in enumerate(red_neuronal):
    print(f"Capa {i} W:\n{c.W}\nCapa {i} b:\n{c.b}")

print("Última capa W (primeros 4):", red_neuronal[-1].W.ravel()[:4])
print("Última capa b:", red_neuronal[-1].b.ravel())

if not VALIDAR:
    np.savetxt("datos_entrenamiento.csv", np.column_stack([X, Y]), delimiter=",",
               header="x,y,clase", comments="", fmt="%.3f")
