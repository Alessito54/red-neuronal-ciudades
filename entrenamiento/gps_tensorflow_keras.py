# Clasificador GPS con TensorFlow/Keras, entrenado con dataset/orizaba.csv y dataset/paris.csv.
# Clase 0 = Orizaba, México. Clase 1 = París, Francia.
#   Python 3.13.18 (entorno virtual "red": source ~/.zshrc && red)
#   tensorflow==2.21.0
#
# Uso (desde la raíz del repo o desde esta carpeta):
#   python entrenamiento/gps_tensorflow_keras.py
# Guarda el modelo en entrenamiento/gps_classifier.keras y la versión para
# TensorFlow Serving en entrenamiento/gps-classifier/1.

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
ORIZABA_PATH = DATASET_DIR / "orizaba.csv"   # clase 0
PARIS_PATH = DATASET_DIR / "paris.csv"       # clase 1
KERAS_PATH = SCRIPT_DIR / "gps_classifier.keras"
EXPORT_PATH = SCRIPT_DIR / "gps-classifier" / "1"


def load_gps(path):
    # Devuelve (X, Y): X con columnas lat, lon; Y con la columna clase
    datos = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if datos.ndim != 2 or datos.shape[1] != 3:
        raise ValueError(f"{path.name} debe tener las columnas lat,lon,clase")
    return datos[:, :2], datos[:, 2:3]


def main():
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)

    X_orizaba, Y_orizaba = load_gps(ORIZABA_PATH)
    X_paris, Y_paris = load_gps(PARIS_PATH)

    X = np.concatenate([X_orizaba, X_paris], axis=0)
    Y = np.concatenate([Y_orizaba, Y_paris], axis=0)
    n0 = len(X_orizaba)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,), dtype=tf.float32, name="coordinates"),
            tf.keras.layers.Dense(4, activation="relu", name="hidden_1"),
            tf.keras.layers.Dense(8, activation="relu", name="hidden_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="gps_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)
        ],
    )

    model.summary()

    model.fit(
        X,
        Y,
        epochs=300,
        batch_size=32,
        shuffle=True,
        verbose=1,
    )

    probabilidades = model.predict(X, verbose=0)
    predicciones = np.round(probabilidades)

    mse_final = np.mean((predicciones - Y) ** 2)
    accuracy_final = np.mean(predicciones == Y)

    print("\n=== RESULTADOS FINALES ===")
    print("MSE sobre predicciones redondeadas:", mse_final)
    print("Accuracy:", accuracy_final)

    print("\n=== Predicciones Orizaba, clase 0 (primeros 10) ===")
    print(probabilidades[:n0][:10])

    print("\n=== Predicciones París, clase 1 (primeros 10) ===")
    print(probabilidades[n0:][:10])

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, bias = layer.get_weights()
            print(f"\nCapa: {layer.name}")
            print("W:")
            print(weights)
            print("b:")
            print(bias)

    model.save(KERAS_PATH)
    model.export(str(EXPORT_PATH))
    print(f"\nModelo guardado en {KERAS_PATH} y exportado en {EXPORT_PATH}")


if __name__ == "__main__":
    main()
