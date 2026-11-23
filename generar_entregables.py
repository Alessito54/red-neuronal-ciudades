# Genera resultados.xlsx y reporte.pdf a partir de la corrida real de red_neuronal.py.
#
# No reimplementa la red: ejecuta red_neuronal.py (con y sin --validar) y usa sus
# propias funciones (capa, relu, sigmoid, mse). Para mostrar los pasos intermedios
# repite la misma secuencia de operaciones de `entrenamiento` guardando cada matriz,
# y comprueba que los pesos, sesgos y predicciones finales sean idénticos a los
# que produjo red_neuronal.py. Si no coinciden, se detiene.
#
# Uso (dentro del .venv):
#   python generar_entregables.py

import contextlib
import io
import os
import runpy
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(DIR, "red_neuronal.py")
EXCEL_REF = {"W": 0.74906805, "b": 0.24981331}  # última capa, Excel de referencia
CIUDADES = {0: "Orizaba, México", 1: "París, Francia"}
COLOR = {0: "#2a78d6", 1: "#eb6834"}  # paleta categórica validada (azul / naranja)
MARCADOR = {0: "o", 1: "^"}
NOMBRE_CAPA = {0: "Capa 0 (2 -> 4, ReLU)", 1: "Capa 1 (4 -> 8, ReLU)", 2: "Capa 2 (8 -> 1, Sigmoide)"}


# --------------------------------------------------------------------------- corrida
def ejecutar(validar):
    """Ejecuta red_neuronal.py tal cual y devuelve su espacio de nombres."""
    argv = sys.argv
    sys.argv = [SCRIPT] + (["--validar"] if validar else [])
    salida = io.StringIO()
    try:
        with contextlib.redirect_stdout(salida):
            ns = runpy.run_path(SCRIPT, run_name="__main__")
    finally:
        sys.argv = argv
    ns["_stdout"] = salida.getvalue()
    return ns


def trazar(ns):
    """Repite `entrenamiento` época por época guardando cada paso intermedio."""
    X, Y, lr = ns["X"], ns["Y"], ns["LR"]
    neuronas, acts = ns["neuronas"], ns["funciones_activacion"]
    capas = [ns["capa"](neuronas[i], neuronas[i + 1], acts[i]) for i in range(len(neuronas) - 1)]
    epocas = []
    for epoch in range(len(ns["error"])):
        ep = {"epoch": epoch, "forward": [], "back": []}
        output = [X]
        for i, c in enumerate(capas):
            W, b = c.W.copy(), c.b.copy()
            z = output[-1] @ c.W + c.b
            a = c.funcion_act[0](z)
            ep["forward"].append({"W": W, "b": b, "z": z, "a": a.copy()})
            output.append(a)
        back = list(range(len(output) - 1))
        back.reverse()
        delta = []
        for cp in back:
            c = capas[cp]
            a_pre = output[cp + 1].copy()
            paso = {"capa": cp, "a": a_pre}
            if cp == back[0]:
                err = ns["mse"](a_pre, Y)[1]
                der = c.funcion_act[1](a_pre.copy())
                paso.update(Y=Y, error=err, error2=err ** 2, mse=float(np.mean(err ** 2)), derivada=der)
                x = err * der
            else:
                prod = delta[-1] @ W_temp
                der = c.funcion_act[1](a_pre.copy())
                paso.update(delta_prev=delta[-1], W_temp=W_temp, delta_W=prod, derivada=der)
                x = prod * der
            delta.append(x)
            W_temp = c.W.transpose()
            entrada = output[cp]
            grad_b = np.mean(delta[-1], axis=0, keepdims=True)
            grad_W = entrada.transpose() @ delta[-1]
            paso.update(delta=x, W=c.W.copy(), b=c.b.copy(), lr=lr, delta_mean=grad_b,
                        entrada_T=entrada.transpose(), grad_W=grad_W)
            c.b = c.b - grad_b * lr
            c.W = c.W - grad_W * lr
            paso.update(W_new=c.W.copy(), b_new=c.b.copy())
            ep["back"].append(paso)
        ep["prediccion"] = output[-1]
        ep["mse_redondeado"] = ns["mse"](np.round(output[-1]), Y)[0]
        epocas.append(ep)

    # Comprobación contra la corrida real de red_neuronal.py
    for i, (mio, real) in enumerate(zip(capas, ns["red_neuronal"])):
        if not (np.array_equal(mio.W, real.W) and np.array_equal(mio.b, real.b)):
            raise SystemExit(f"La traza no coincide con red_neuronal.py en la capa {i}")
    if not np.array_equal(epocas[-1]["prediccion"], ns["predicciones"][-1]):
        raise SystemExit("Las predicciones de la traza no coinciden con red_neuronal.py")
    if not np.allclose([e["mse_redondeado"] for e in epocas], ns["error"]):
        raise SystemExit("El MSE de la traza no coincide con red_neuronal.py")
    return epocas


# --------------------------------------------------------------------------- Excel
TITULO = Font(bold=True, size=12)
NEGRITA = Font(bold=True)
GRIS = PatternFill("solid", fgColor="E8E8E8")
AZUL = PatternFill("solid", fgColor="DCE9F9")


def bloque(ws, fila, col, M, titulo, subtitulo=None):
    """Escribe una matriz con título; devuelve el ancho usado (columnas)."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    ws.cell(fila, col, titulo).font = NEGRITA
    ws.cell(fila, col).fill = AZUL
    if subtitulo:
        ws.cell(fila + 1, col, subtitulo).font = Font(italic=True, size=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ws.cell(fila + 2 + i, col + j, float(M[i, j])).number_format = "0.00000000"
    return M.shape[1]


def fila_bloques(ws, fila, bloques, col=2):
    """Coloca bloques lado a lado con una columna de separación; devuelve alto máximo."""
    alto = 0
    for titulo, M, sub in bloques:
        ancho = bloque(ws, fila, col, M, titulo, sub)
        col += ancho + 1
        alto = max(alto, np.atleast_2d(M).shape[0])
    return alto + 3


def forma(M):
    M = np.atleast_2d(M)
    return f"{M.shape[0]} x {M.shape[1]}"


def hoja_epoca(wb, ep, ns):
    N2 = len(ns["Y"])
    ws = wb.create_sheet(f"epoch {ep['epoch']} N={N2}")
    ws.cell(1, 1, f"Epoch {ep['epoch']}  ·  {N2} puntos  ·  lr = {ns['LR']}  ·  semilla 42").font = TITULO
    ws.cell(2, 1, "FORWARD").font = TITULO
    ws.cell(2, 1).fill = GRIS

    # Columna de índice y clase
    ws.cell(3, 1, "#").font = NEGRITA
    for i in range(N2):
        ws.cell(5 + i, 1, i + 1)
    f = ep["forward"]
    bloques = [("X", ns["X"], f"INPUT = 2  ({forma(ns['X'])})")]
    for i, paso in enumerate(f):
        act = "SIGMOIDE" if i == len(f) - 1 else "RELU"
        bloques += [
            (f"W capa {i}", paso["W"], forma(paso["W"])),
            (f"b capa {i}", paso["b"], forma(paso["b"])),
            (f"z capa {i}", paso["z"], "z = a_prev @ W + b"),
            (f"a capa {i}", paso["a"], act),
        ]
    fila = 3 + fila_bloques(ws, 3, bloques) + 1

    ws.cell(fila, 1, "BACK PROPAGATION").font = TITULO
    ws.cell(fila, 1).fill = GRIS
    ws.cell(fila + 1, 1, "NOTA: se lee al revés en esta etapa (capa 2 -> capa 0)").font = Font(italic=True)
    fila += 3
    for paso in ep["back"]:
        cp = paso["capa"]
        ws.cell(fila, 1, NOMBRE_CAPA[cp]).font = TITULO
        fila += 1
        if "Y" in paso:
            ws.cell(fila, 2, "MSE (sobre a continua)").font = NEGRITA
            ws.cell(fila, 4, paso["mse"]).number_format = "0.00000000"
            fila += 1
            bloques = [
                ("a", paso["a"], "salida sigmoide"),
                ("Y", paso["Y"], "0 = Orizaba, 1 = París"),
                ("a - Y", paso["error"], "mse(a, Y)[1]"),
                ("(a - Y)^2", paso["error2"], "error cuadrático"),
                ("derivada", paso["derivada"], "a * (1 - a)"),
                ("Delta", paso["delta"], "(a - Y) * derivada"),
            ]
        else:
            bloques = [
                ("Delta (capa siguiente)", paso["delta_prev"], forma(paso["delta_prev"])),
                ("Wtemp", paso["W_temp"], "W transpuesta (antes de actualizar)"),
                ("Delta @ Wtemp", paso["delta_W"], forma(paso["delta_W"])),
                ("a", paso["a"], "salida ReLU"),
                ("derivada relu (a)", paso["derivada"], "1 si a > 0, si no 0"),
                ("Delta", paso["delta"], "(Delta @ Wtemp) * derivada"),
            ]
        nombre_in = "X" if cp == 0 else f"a capa {cp - 1}"
        bloques += [
            ("Delta promedio", paso["delta_mean"], "mean(Delta, axis=0)"),
            ("b", paso["b"], forma(paso["b"])),
            ("lr", np.array([[paso["lr"]]]), ""),
            ("b new", paso["b_new"], "b - Delta promedio * lr"),
            ("W", paso["W"], forma(paso["W"])),
            (f"{nombre_in} transpuesta @ Delta", paso["grad_W"], forma(paso["grad_W"])),
            ("W new", paso["W_new"], "W - (aᵀ @ Delta) * lr"),
        ]
        fila += fila_bloques(ws, fila, bloques) + 1
        ws.cell(fila, 2, f"{nombre_in} transpuesta ({forma(paso['entrada_T'])})").font = NEGRITA
        fila += fila_bloques(ws, fila, [("", paso["entrada_T"], None)]) + 1
    ws.column_dimensions["A"].width = 22
    ws.freeze_panes = "B5"


def hoja_resumen(wb, epocas, ns):
    ws = wb.create_sheet("predicciones y MSE", 0)
    Y = ns["Y"].ravel()
    pred = epocas[-1]["prediccion"].ravel()
    ws.cell(1, 1, "Resumen de la corrida real").font = TITULO
    info = [
        ("Semilla", 42), ("Puntos por clase", ns["N"]), ("Total de puntos", len(Y)),
        ("lr", ns["LR"]), ("Arquitectura", "[2, 4, 8, 1]"), ("Activaciones", "relu, relu, sigmoid"),
        ("Épocas", len(epocas)),
    ]
    for e in epocas:
        info.append((f"MSE época {e['epoch']} (predicción redondeada)", float(e["mse_redondeado"])))
        info.append((f"MSE época {e['epoch']} (a continua)", e["back"][0]["mse"]))
    info.append(("Exactitud", float(np.mean(np.round(pred) == Y))))
    for i, (k, v) in enumerate(info):
        ws.cell(3 + i, 1, k).font = NEGRITA
        ws.cell(3 + i, 2, v)
    fila = 4 + len(info)
    enc = ["#", "x", "y", "clase", "ciudad", "predicción (a)", "redondeo", "¿acierto?"]
    for j, h in enumerate(enc):
        ws.cell(fila, 1 + j, h).font = NEGRITA
        ws.cell(fila, 1 + j).fill = AZUL
    X = ns["X"]
    for i in range(len(Y)):
        r = fila + 1 + i
        vals = [i + 1, float(X[i, 0]), float(X[i, 1]), int(Y[i]), CIUDADES[int(Y[i])],
                float(pred[i]), int(np.round(pred[i])), "sí" if np.round(pred[i]) == Y[i] else "no"]
        for j, v in enumerate(vals):
            ws.cell(r, 1 + j, v)
        ws.cell(r, 6).number_format = "0.00000000"
    for col, w in zip("ABCDEFGH", [38, 10, 10, 8, 18, 16, 10, 10]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = ws.cell(fila + 1, 1)


def hoja_pesos(wb, ns):
    ws = wb.create_sheet("pesos finales")
    ws.cell(1, 1, "Pesos y sesgos finales (después del entrenamiento)").font = TITULO
    fila = 3
    for i, c in enumerate(ns["red_neuronal"]):
        ws.cell(fila, 1, NOMBRE_CAPA[i]).font = TITULO
        fila += 1 + fila_bloques(ws, fila + 1, [("W", c.W, forma(c.W)), ("b", c.b, forma(c.b))], col=1)


def hoja_validacion(wb, ns_val):
    ws = wb.create_sheet("validacion N=20")
    ws.cell(1, 1, "Validación: 20 puntos fijos del Excel de referencia, lr = 0.01").font = TITULO
    ultima = ns_val["red_neuronal"][-1]
    filas = [("", "Excel de referencia", "red_neuronal.py --validar", "diferencia")]
    for k, v in [("W última capa", float(ultima.W.ravel()[0])), ("b última capa", float(ultima.b.ravel()[0]))]:
        ref = EXCEL_REF[k[0]]
        filas.append((k, ref, v, v - ref))
    for i, f in enumerate(filas):
        for j, v in enumerate(f):
            c = ws.cell(3 + i, 1 + j, v)
            if i == 0:
                c.font = NEGRITA
            elif j > 0:
                c.number_format = "0.00000000"
    ws.cell(7, 1, "Pesos W de la última capa (8 x 1):").font = NEGRITA
    for i, w in enumerate(ultima.W.ravel()):
        ws.cell(8 + i, 1, float(w)).number_format = "0.00000000"
    for col, w in zip("ABCD", [18, 22, 28, 14]):
        ws.column_dimensions[col].width = w


def crear_excel(epocas, ns, ns_val, ruta):
    wb = Workbook()
    wb.remove(wb.active)
    for ep in epocas:
        hoja_epoca(wb, ep, ns)
    hoja_resumen(wb, epocas, ns)
    hoja_pesos(wb, ns)
    hoja_validacion(wb, ns_val)
    for ws in wb.worksheets:
        ws.sheet_view.zoomScale = 90
        for row in ws.iter_rows(min_row=1, max_row=2):
            for c in row:
                c.alignment = Alignment(vertical="center")
    wb.save(ruta)


# --------------------------------------------------------------------------- PDF
def grafica_dispersion(ns):
    X, Y = ns["X"], ns["Y"].ravel()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for k in (0, 1):
        m = Y == k
        ax.scatter(X[m, 0], X[m, 1], s=22, c=COLOR[k], marker=MARCADOR[k],
                   edgecolors="#fcfcfb", linewidths=0.6, label=f"{CIUDADES[k]} (clase {k}, n = {m.sum()})")
    ax.set_aspect("equal")
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.set_xlabel("x", color="#55554f")
    ax.set_ylabel("y", color="#55554f")
    ax.set_title("Datos de entrenamiento: 300 puntos (150 por ciudad)", fontsize=11, color="#0b0b0b")
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="#fcfcfb", edgecolor="#e1e0d9")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def tabla(M, encabezado=None, filas_idx=None, dec=6, ancho=None):
    M = np.atleast_2d(np.asarray(M, dtype=float))
    datos = []
    if encabezado:
        datos.append(encabezado)
    for i in range(M.shape[0]):
        fila = [f"{v:.{dec}f}" for v in M[i]]
        if filas_idx is not None:
            fila = [filas_idx[i]] + fila
        datos.append(fila)
    t = Table(datos, colWidths=ancho, repeatRows=1 if encabezado else 0)
    estilo = [
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]
    if encabezado:
        estilo += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f9")),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(estilo))
    return t


def muestra(M, n=5):
    """Primeras n filas (Orizaba) y últimas n filas (París) de una matriz por punto."""
    M = np.atleast_2d(M)
    total = M.shape[0]
    idx = list(range(n)) + list(range(total - n, total))
    return M[idx], [str(i + 1) for i in idx]


def crear_pdf(epocas, ns, ns_val, ruta):
    est = getSampleStyleSheet()
    h1, h2, h3, p = est["Title"], est["Heading2"], est["Heading3"], est["BodyText"]
    nota = ParagraphStyle("nota", parent=p, fontSize=8, textColor=colors.HexColor("#55554f"))
    hist = []
    N2 = len(ns["Y"])
    Y = ns["Y"].ravel()
    pred = epocas[-1]["prediccion"].ravel()
    exactitud = float(np.mean(np.round(pred) == Y))

    hist += [Paragraph("Red neuronal en numpy: Orizaba vs. París", h1),
             Paragraph("Práctica del curso <i>Introducción a Deep Learning</i>. Red neuronal escrita "
                       "desde cero en numpy y entrenada con coordenadas generadas alrededor del origen. "
                       "Las ciudades son solo etiquetas de clase.", p),
             Spacer(1, 6)]
    conf = [["Parámetro", "Valor"],
            ["Clase 0", "Orizaba, México (R = 2)"], ["Clase 1", "París, Francia (R = 0.5)"],
            ["Puntos por clase", str(ns["N"])], ["Total de puntos", str(N2)],
            ["Semilla", "42"], ["lr", str(ns["LR"])], ["Arquitectura", "[2, 4, 8, 1]"],
            ["Activaciones", "relu, relu, sigmoid"], ["Inicialización", "W = 0.75, b = 0.25"],
            ["Épocas", str(len(epocas))]]
    t = Table(conf, colWidths=[5 * cm, 8 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f9")),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    hist += [t, Spacer(1, 10)]

    # Validación
    ult = ns_val["red_neuronal"][-1]
    hist.append(Paragraph("1. Validación contra el Excel de referencia", h2))
    hist.append(Paragraph("Con los 20 puntos fijos del Excel y lr = 0.01 "
                          "(<font face='Courier'>python red_neuronal.py --validar</font>), la última capa "
                          "debe coincidir con el Excel de referencia.", p))
    val = [["", "Excel", "numpy", "diferencia"],
           ["W última capa", f"{EXCEL_REF['W']:.8f}", f"{ult.W.ravel()[0]:.8f}", f"{ult.W.ravel()[0] - EXCEL_REF['W']:.1e}"],
           ["b última capa", f"{EXCEL_REF['b']:.8f}", f"{ult.b.ravel()[0]:.8f}", f"{ult.b.ravel()[0] - EXCEL_REF['b']:.1e}"]]
    t = Table(val, colWidths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f9")),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    hist += [Spacer(1, 4), t]

    # Datos
    hist += [PageBreak(), Paragraph("2. Datos de entrenamiento", h2),
             Image(grafica_dispersion(ns), width=15 * cm, height=13.1 * cm),
             Paragraph("Azul / círculos: Orizaba (clase 0, R = 2). Naranja / triángulos: París (clase 1, R = 0.5). "
                       "Los datos también se guardan en <font face='Courier'>datos_entrenamiento.csv</font>.", nota)]

    for ep in epocas:
        hist += [PageBreak(), Paragraph(f"3. Época {ep['epoch']}: forward", h2),
                 Paragraph(f"z = a_prev @ W + b, a = f(z). Las matrices por punto ({N2} filas) se muestran "
                           "con las 5 primeras filas (Orizaba) y las 5 últimas (París); las "
                           f"{N2} filas completas están en <font face='Courier'>resultados.xlsx</font>.", nota)]
        for i, paso in enumerate(ep["forward"]):
            act = "sigmoide" if i == len(ep["forward"]) - 1 else "ReLU"
            hist.append(Paragraph(f"{NOMBRE_CAPA[i]}", h3))
            hist += [Paragraph(f"W ({forma(paso['W'])}) y b ({forma(paso['b'])}) antes de actualizar", nota),
                     tabla(paso["W"]), Spacer(1, 2), tabla(paso["b"]), Spacer(1, 4)]
            zm, idx = muestra(paso["z"])
            am, _ = muestra(paso["a"])
            ncol = paso["z"].shape[1]
            hist += [Paragraph(f"z = a_prev @ W + b ({forma(paso['z'])})", nota),
                     tabla(zm, ["#"] + [f"z{j}" for j in range(ncol)], idx, dec=4), Spacer(1, 3),
                     Paragraph(f"a = {act}(z)", nota),
                     tabla(am, ["#"] + [f"a{j}" for j in range(ncol)], idx, dec=4)]

        hist += [PageBreak(), Paragraph(f"4. Época {ep['epoch']}: backpropagation", h2),
                 Paragraph("Se recorre al revés: capa 2 -> capa 1 -> capa 0. "
                           "b new = b - mean(Delta)·lr ; W new = W - (a_prev^T @ Delta)·lr.", nota)]
        for paso in ep["back"]:
            cp = paso["capa"]
            hist.append(Paragraph(NOMBRE_CAPA[cp], h3))
            if "Y" in paso:
                hist.append(Paragraph(f"MSE sobre la salida continua: {paso['mse']:.8f}. "
                                      "Delta = (a - Y) · a(1 - a).", nota))
                cols = [paso["a"], paso["Y"], paso["error"], paso["error2"], paso["derivada"], paso["delta"]]
                M, idx = muestra(np.hstack(cols))
                hist.append(tabla(M, ["#", "a", "Y", "a - Y", "(a - Y)²", "derivada", "Delta"], idx, dec=6))
            else:
                hist.append(Paragraph("Wtemp = W de la capa siguiente transpuesta (antes de actualizarla). "
                                      "Delta = (Delta_sig @ Wtemp) · relu'(a).", nota))
                hist += [Paragraph(f"Wtemp ({forma(paso['W_temp'])})", nota), tabla(paso["W_temp"]), Spacer(1, 3)]
                n = paso["a"].shape[1]
                for titulo, clave, pref in [("Delta_sig @ Wtemp", "delta_W", "DWt"),
                                            ("derivada relu (a)", "derivada", "relu'"),
                                            ("Delta", "delta", "D")]:
                    M, idx = muestra(paso[clave])
                    hist += [Paragraph(f"{titulo} ({forma(paso[clave])})", nota),
                             tabla(M, ["#"] + [f"{pref}{j}" for j in range(n)], idx, dec=4), Spacer(1, 3)]
            hist += [Spacer(1, 3),
                     Paragraph("Delta promedio, b y b new", nota),
                     tabla(np.vstack([paso["delta_mean"], paso["b"], paso["b_new"]]), None,
                           ["mean Delta", "b", "b new"], dec=8),
                     Spacer(1, 3),
                     Paragraph(f"a_prev^T @ Delta ({forma(paso['grad_W'])})", nota), tabla(paso["grad_W"], dec=6),
                     Spacer(1, 3),
                     Paragraph("W new", nota), tabla(paso["W_new"], dec=8), Spacer(1, 6)]

    # Resultados
    hist += [PageBreak(), Paragraph("5. MSE y predicciones", h2)]
    res = [["Época", "MSE (predicción redondeada)", "MSE (a continua)"]]
    for e in epocas:
        res.append([str(e["epoch"]), f"{e['mse_redondeado']:.6f}", f"{e['back'][0]['mse']:.6f}"])
    t = Table(res)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f9")),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    pred0, pred1 = np.round(pred[Y == 0]), np.round(pred[Y == 1])
    hist += [t, Spacer(1, 6),
             Paragraph(f"Exactitud: {exactitud:.4f}. Orizaba: {int((pred0 == 0).sum())}/{len(pred0)} aciertos; "
                       f"París: {int((pred1 == 1).sum())}/{len(pred1)} aciertos. "
                       f"Salida a: mín {pred.min():.4f}, máx {pred.max():.4f}.", p),
             Paragraph("Las predicciones corresponden a la salida del forward de la última época "
                       "(antes de aplicar la actualización de esa época), igual que en red_neuronal.py.", nota),
             Spacer(1, 6)]
    filas = [["#", "x", "y", "ciudad", "a", "redondeo"]]
    for i in range(N2):
        filas.append([str(i + 1), f"{ns['X'][i, 0]:.3f}", f"{ns['X'][i, 1]:.3f}", CIUDADES[int(Y[i])],
                      f"{pred[i]:.6f}", str(int(np.round(pred[i])))])
    t = Table(filas, repeatRows=1, colWidths=[1.2 * cm, 2 * cm, 2 * cm, 3.6 * cm, 2.4 * cm, 1.8 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f9")),
                           ("FONTSIZE", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 1),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    hist.append(t)

    # Pesos finales
    hist += [PageBreak(), Paragraph("6. Pesos y sesgos finales", h2)]
    for i, c in enumerate(ns["red_neuronal"]):
        hist += [Paragraph(NOMBRE_CAPA[i], h3), Paragraph(f"W ({forma(c.W)})", nota), tabla(c.W, dec=8),
                 Spacer(1, 2), Paragraph("b", nota), tabla(c.b, dec=8), Spacer(1, 6)]

    doc = SimpleDocTemplate(ruta, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Red neuronal numpy: Orizaba vs. París")
    doc.build(hist)


# --------------------------------------------------------------------------- main
def main():
    os.chdir(DIR)
    ns_val = ejecutar(validar=True)
    ult = ns_val["red_neuronal"][-1]
    if not (np.allclose(ult.W, EXCEL_REF["W"], atol=5e-9) and np.allclose(ult.b, EXCEL_REF["b"], atol=5e-9)):
        raise SystemExit(f"La validación no coincide con el Excel: W={ult.W.ravel()}, b={ult.b.ravel()}")
    print(f"Validación OK: W = {ult.W.ravel()[0]:.8f}, b = {ult.b.ravel()[0]:.8f}")

    ns = ejecutar(validar=False)  # también escribe datos_entrenamiento.csv
    epocas = trazar(ns)
    print(f"Corrida real: {len(ns['Y'])} puntos, lr = {ns['LR']}, épocas = {len(epocas)}; traza idéntica a red_neuronal.py")

    crear_excel(epocas, ns, ns_val, os.path.join(DIR, "resultados.xlsx"))
    print("Escrito resultados.xlsx")
    crear_pdf(epocas, ns, ns_val, os.path.join(DIR, "reporte.pdf"))
    print("Escrito reporte.pdf")


if __name__ == "__main__":
    main()
