"""Rutas, constantes y lectura de los datos del proyecto.

Los ficheros de ``data/raw/`` son agregados de los microdatos oficiales de
matriculaciones de la DGT (ver ``pipeline/fase0_descarga_dgt.py``). Tienen un
formato compacto pensado para versionarse en Git: una línea por serie,
``clave:valor_mes1,valor_mes2,...`` con un hueco vacío cuando el valor es 0.
Las columnas siguen el orden de ``data/raw/months.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed"
GOLD = DATA / "gold"
RESULTS = RAIZ / "results"
MODELS = RAIZ / "models"
FIGURES = RAIZ / "reports" / "figures"

# Segmentos según el permiso de conducción necesario. Se calculan a partir del
# tipo de vehículo, la cilindrada y la potencia (kW) que figuran en la ficha ITV.
SEGMENTOS = ["AM", "A1", "A2", "A"]
NOMBRE_SEGMENTO = {
    "AM": "Ciclomotor (AM)",
    "A1": "125 cc (A1 / B+3)",
    "A2": "Media (A2)",
    "A": "Alta (A)",
}
DESCRIPCION_SEGMENTO = {
    "AM": "Ciclomotores de hasta 50 cc o 4 kW. Carnet AM (desde 15 años).",
    "A1": "Motos de hasta 125 cc y 11 kW. Carnet A1, o B con 3 años de antigüedad.",
    "A2": "Motos de hasta 35 kW. Carnet A2 (desde 18 años).",
    "A": "Motos de más de 35 kW. Carnet A (2 años de A2).",
}

# Meses con el confinamiento de 2020: la actividad administrativa se paralizó y
# no reflejan demanda. Se marcan para no aprender de ellos.
MESES_COVID = pd.period_range("2020-03", "2020-05", freq="M")


def meses() -> pd.PeriodIndex:
    texto = (RAW / "months.txt").read_text().strip()
    return pd.PeriodIndex([pd.Period(f"{m[:4]}-{m[4:]}", freq="M") for m in texto.split(",")])


def _leer_series(fichero: Path) -> dict[str, np.ndarray]:
    series = {}
    for linea in fichero.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        clave, valores = linea.split(":", 1)
        series[clave] = np.array([int(v) if v else 0 for v in valores.split(",")], dtype=int)
    return series


def leer_provincia_segmento() -> pd.DataFrame:
    """Matriculaciones de motos nuevas por mes, provincia y segmento (formato largo)."""
    m = meses()
    filas = []
    for clave, valores in _leer_series(RAW / "dgt_provincia_segmento.txt").items():
        provincia, segmento = clave.split("|")
        if len(valores) != len(m):
            raise ValueError(f"La serie {clave} tiene {len(valores)} meses y se esperaban {len(m)}")
        filas.append(pd.DataFrame({"mes": m, "provincia": provincia, "segmento": segmento,
                                   "matriculaciones": valores}))
    df = pd.concat(filas, ignore_index=True)
    df["segmento"] = pd.Categorical(df["segmento"], categories=SEGMENTOS, ordered=True)
    return df.sort_values(["provincia", "segmento", "mes"], ignore_index=True)


def leer_nacional_detalle() -> pd.DataFrame:
    """Total nacional por segmento, nuevo/usado importado, propulsión y tipo de titular."""
    m = meses()
    filas = []
    for clave, valores in _leer_series(RAW / "dgt_nacional_segmento.txt").items():
        segmento, nu, prop, titular = clave.split("|")
        filas.append(pd.DataFrame({
            "mes": m, "segmento": segmento,
            "estado": "Nueva" if nu == "N" else "Usada importada",
            "propulsion": "Eléctrica" if prop == "E" else "Combustión",
            "titular": "Particular" if titular == "F" else "Empresa",
            "matriculaciones": valores}))
    df = pd.concat(filas, ignore_index=True)
    df["segmento"] = pd.Categorical(df["segmento"], categories=SEGMENTOS, ordered=True)
    return df


def leer_marcas_mensual() -> pd.DataFrame:
    m = meses()
    filas = [pd.DataFrame({"mes": m, "marca": marca, "matriculaciones": v})
             for marca, v in _leer_series(RAW / "dgt_marca_mensual.txt").items()]
    return pd.concat(filas, ignore_index=True)


def leer_marcas_segmento_12m() -> pd.DataFrame:
    filas = []
    for marca, v in _leer_series(RAW / "dgt_marca_segmento_12m.txt").items():
        filas.append({"marca": marca, **dict(zip(SEGMENTOS, v))})
    return pd.DataFrame(filas)


def leer_top_modelos() -> pd.DataFrame:
    filas = []
    for linea in (RAW / "dgt_top_modelos_12m.txt").read_text(encoding="utf-8").splitlines():
        marca, modelo, seg, n, kw, cc = linea.split("|")
        filas.append({"marca": marca, "modelo": " ".join(modelo.split()), "segmento": seg,
                      "matriculaciones": int(n), "kw_medio": float(kw), "cc_medio": int(cc)})
    return pd.DataFrame(filas)


def leer_provincias() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL / "provincias.csv", keep_default_na=False)
