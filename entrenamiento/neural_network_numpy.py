# Red neuronal en numpy desde cero, entrenada con puntos GPS (lat, lon).
# Clase 0 = Orizaba, México. Clase 1 = París, Francia.
# Los puntos se leen de dataset/orizaba.csv y dataset/paris.csv (columnas lat,lon,clase).
# La lógica de la red es la del código de referencia "excel demo".
#
# Uso (desde la raíz del repo o desde esta carpeta):
#   python entrenamiento/neural_network_numpy.py              -> puntos de los CSV, lr = 0.01, 2 épocas
#   python entrenamiento/neural_network_numpy.py --validar    -> 20 puntos fijos del Excel, lr = 0.01, 2 épocas
#   streamlit run entrenamiento/neural_network_numpy.py       -> entrena y muestra el mapa en Streamlit

import sys
from pathlib import Path

import numpy as np

VALIDAR = "--validar" in sys.argv

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
ORIZABA_PATH = DATASET_DIR / "orizaba.csv"   # clase 0
PARIS_PATH = DATASET_DIR / "paris.csv"       # clase 1
MODELO_PATH = SCRIPT_DIR / "modelo.npz"

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


def cargar_csv(path):
    # Devuelve (X, Y): X con columnas lat, lon; Y con la columna clase
    datos = np.loadtxt(path, delimiter=",", skiprows=1)
    if datos.ndim != 2 or datos.shape[1] != 3:
        raise ValueError(f"{path.name} debe tener las columnas lat,lon,clase")
    return datos[:, :2], datos[:, 2:3]


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
    # 20 puntos fijos del Excel de referencia (10 por clase), lr = 0.01, 2 épocas
    LR = 0.01
    EPOCAS = 2
    X = np.array([
        [-2.848, 5.975], [-10.429, -8.024], [-1.284, 2.872], [4.758, 5.955],
        [-3.938, -5.78], [8.525, 17.866], [-13.715, 4.789], [-3.19, 13.309],
        [3.227, 18.344], [-5.014, -17.507], [0.491, 1.897], [-0.54, 1.12],
        [-2.032, 1.092], [-1.476, -1.606], [-0.084, 3.386], [-1.618, 1.16],
        [2.238, -1.671], [-2.102, 1.924], [-1.059, 4.297], [2.642, 2.981],
    ])
    Y = np.array([0] * 10 + [1] * 10).reshape(-1, 1)
else:
    # Puntos GPS de los CSV del dataset
    LR = 0.01
    EPOCAS = 2  # igual que la validación del PDF y el Excel de referencia
    X_orizaba, Y_orizaba = cargar_csv(ORIZABA_PATH)
    X_paris, Y_paris = cargar_csv(PARIS_PATH)
    X = np.concatenate([X_orizaba, X_paris])
    Y = np.concatenate([Y_orizaba, Y_paris])

N0 = int(np.sum(Y == 0))  # puntos de la clase 0 (van primero)

neuronas = [2, 4, 8, 1]
funciones_activacion = [relu, relu, sigmoid]


def construir_red():
    red = []
    for paso in range(len(neuronas) - 1):
        red.append(capa(neuronas[paso], neuronas[paso + 1], funciones_activacion[paso]))
    return red


error = []
predicciones = []

for epoch in range(0, EPOCAS):
    # Cada época arranca con pesos 0.75 y sesgos 0.25, como en el Excel de referencia
    red_neuronal = construir_red()
    ronda = entrenamiento(epoch, X=X, Y=Y, red_neuronal=red_neuronal, lr=LR)
    predicciones.append(ronda)
    temp = mse(np.round(predicciones[-1]), Y)[0]
    error.append(temp)
    print(f"Época {epoch}: última capa W = {red_neuronal[-1].W.ravel()[0]:.8f}, b = {red_neuronal[-1].b.ravel()[0]:.8f}")

print(f"lr = {LR}, épocas = {EPOCAS}, puntos clase 0 = {N0}, puntos clase 1 = {len(Y) - N0}")
print("Error MSE por época:", [float(e) for e in error])
print("Predicciones clase 0:", predicciones[-1][:N0].ravel().round(4))
print("Predicciones clase 1:", predicciones[-1][N0:].ravel().round(4))
print("Exactitud:", np.mean(np.round(predicciones[-1]) == Y))

for i, c in enumerate(red_neuronal):
    print(f"Capa {i} W:\n{c.W}\nCapa {i} b:\n{c.b}")

if not VALIDAR:
    # Modelo: W y b de las tres capas después de la última época
    np.savez(MODELO_PATH,
             **{f"W{i}": c.W for i, c in enumerate(red_neuronal)},
             **{f"b{i}": c.b for i, c in enumerate(red_neuronal)},
             neuronas=np.array(neuronas))
    print(f"Modelo guardado en {MODELO_PATH}")


# ---------------------------------------------------------------------------
# Mapa con Streamlit (solo se muestra con: streamlit run entrenamiento/neural_network_numpy.py)
# ---------------------------------------------------------------------------
def corriendo_en_streamlit():
    try:
        from streamlit.runtime import exists
    except ImportError:
        return False
    return exists()


if not VALIDAR and corriendo_en_streamlit():
    import pandas as pd
    import streamlit as st

    map_data = pd.DataFrame(X, columns=["lat", "lon"])

    st.title("Orizaba y París: puntos GPS")
    st.header("Using Streamlit and numpy")
    st.write(f"Exactitud de la red: {np.mean(np.round(predicciones[-1]) == Y):.2f}")

    # Mostrar el conjunto de datos GPS en el mapa
    st.map(map_data)
