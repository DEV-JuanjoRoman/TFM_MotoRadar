"""Fase 3 - Validación de origen móvil (backtest) y comparación de modelos.

Simula cómo habría funcionado el sistema en producción durante los últimos 23
meses: en cada mes de origen (de septiembre de 2024 a julio de 2026) se entrena
solo con lo que se sabía ese mes, se predicen los 3 meses siguientes y se compara
con lo que pasó de verdad.

    python pipeline/fase3_backtest.py            # completo (~10 min con ETS)
    python pipeline/fase3_backtest.py --rapido   # sin ETS (~1 min)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motoradar import datos as D  # noqa: E402
from motoradar import metricas as Me  # noqa: E402
from motoradar import modelos as Mo  # noqa: E402
from motoradar.features import construir_tabla, matriz_series  # noqa: E402

ORIGENES = pd.period_range("2024-09", "2026-07", freq="M")
MODELOS = {"naive_estacional": "Naive estacional (año anterior)",
           "naive_tendencia": "Naive estacional con tendencia",
           "ets": "Holt-Winters (ETS)",
           "lightgbm": "LightGBM global"}


def predicciones_ets(df: pd.DataFrame, origenes) -> pd.DataFrame:
    Y = matriz_series(df)
    filas = []
    for origen in origenes:
        hist = Y.loc[:origen]
        for s, col in enumerate(Y.columns):
            pred = Mo.prever_ets(hist[col].to_numpy(), 3)
            for h in (1, 2, 3):
                filas.append((s, origen, h, pred[h - 1]))
    return pd.DataFrame(filas, columns=["serie", "origen", "h", "ets"])


def main(rapido: bool):
    df = D.leer_provincia_segmento()
    tabla = construir_tabla(df)
    resultados = []
    t0 = time.time()
    for origen in ORIGENES:
        train = tabla[tabla["mes_objetivo"] <= origen]  # solo objetivos ya observados
        test = tabla[tabla["origen"] == origen].copy()
        modelo = Mo.entrenar_lgbm(train)
        test["lightgbm"] = Mo.predecir_lgbm(modelo, test)
        test["naive_estacional"] = Mo.naive_estacional(test)
        test["naive_tendencia"] = Mo.naive_tendencia(test)
        resultados.append(test)
        print(f"  origen {origen}: {len(test)} predicciones ({time.time() - t0:.0f} s)")
    R = pd.concat(resultados, ignore_index=True)

    if not rapido:
        print("Ajustando Holt-Winters serie a serie...")
        ets = predicciones_ets(df, ORIGENES)
        R = R.merge(ets, on=["serie", "origen", "h"], how="left")
    modelos = [m for m in MODELOS if m in R.columns]

    cols = ["origen", "mes_objetivo", "h", "provincia", "segmento", "y"] + modelos
    salida = R[cols].assign(origen=R["origen"].astype(str), mes_objetivo=R["mes_objetivo"].astype(str))
    D.RESULTS.mkdir(exist_ok=True)
    salida.to_csv(D.RESULTS / "backtest_predicciones.csv", index=False)

    # ---------------- Resumen ----------------
    nac = R.groupby(["origen", "h", "segmento"], observed=True)[["y"] + modelos].sum().reset_index()
    filas = []
    for m in modelos:
        filas.append({
            "modelo": MODELOS[m],
            "wape_provincia": Me.wape(R["y"], R[m]),
            "wape_nacional": Me.wape(nac["y"], nac[m]),
            "mae_provincia": Me.mae(R["y"], R[m]),
            "sesgo": Me.sesgo(R["y"], R[m]),
        })
    resumen = pd.DataFrame(filas)
    resumen.to_csv(D.RESULTS / "comparativa_modelos.csv", index=False)
    print("\n" + resumen.round(4).to_string(index=False))

    por_h = R.groupby("h").apply(lambda g: pd.Series({MODELOS[m]: Me.wape(g["y"], g[m]) for m in modelos}))
    por_h.to_csv(D.RESULTS / "wape_por_horizonte.csv")
    por_seg = R.groupby("segmento", observed=True).apply(
        lambda g: pd.Series({MODELOS[m]: Me.wape(g["y"], g[m]) for m in modelos}))
    por_seg.to_csv(D.RESULTS / "wape_por_segmento.csv")
    print("\nWAPE por horizonte\n" + por_h.round(4).to_string())
    print("\nWAPE por segmento\n" + por_seg.round(4).to_string())

    # Mejora frente al mejor baseline, con intervalo bootstrap por meses de origen
    filas = []
    for ref in [m for m in modelos if m != "lightgbm"]:
        p, lo, hi = Me.mejora_bootstrap(R, "lightgbm", ref)
        filas.append({"referencia": MODELOS[ref], "mejora": p, "ic95_inf": lo, "ic95_sup": hi})
    boot = pd.DataFrame(filas)
    boot.to_csv(D.RESULTS / "mejora_bootstrap.csv", index=False)
    print("\nMejora del WAPE de LightGBM (IC 95 % bootstrap por meses de origen)\n"
          + boot.round(4).to_string(index=False))

    # Meses de origen en los que gana LightGBM a cada baseline
    gana = R.groupby("origen").apply(lambda g: pd.Series(
        {MODELOS[m]: Me.wape(g["y"], g["lightgbm"]) < Me.wape(g["y"], g[m])
         for m in modelos if m != "lightgbm"}))
    gana.sum().rename("origenes_ganados").to_frame().assign(total=len(gana)).to_csv(
        D.RESULTS / "origenes_ganados.csv")
    print("\nOrígenes en los que LightGBM gana:\n" + gana.sum().to_string() + f"  (de {len(gana)})")
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="omite Holt-Winters")
    main(ap.parse_args().rapido)
