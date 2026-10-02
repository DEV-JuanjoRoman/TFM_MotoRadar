"""Fase 4 - Análisis de errores e intervalos de predicción.

A partir de las predicciones del backtest (fase 3):

* errores del modelo por tamaño de provincia, mes del año y serie;
* intervalos de predicción del 80 % por segmento y horizonte, calibrados con la
  primera mitad del backtest y evaluados (cobertura real) en la segunda mitad;
* gráficos para la memoria.

    python pipeline/fase4_analisis_errores.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motoradar import datos as D  # noqa: E402
from motoradar import metricas as Me  # noqa: E402
from motoradar.estilo import AZUL, COLOR_SEGMENTO, GRIS, NARANJA, TEXTO_2  # noqa: E402
from pipeline.fase2_eda import guardar  # noqa: E402  (aplica también el estilo)

NIVEL = 0.80


def cuantiles_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Cuantiles del cociente real/previsto por segmento y horizonte.

    Se usa log((y+1)/(p+1)) para que las series pequeñas no den cocientes infinitos.
    """
    r = np.log1p(df["y"]) - np.log1p(df["lightgbm"])
    q = (1 - NIVEL) / 2
    return (df.assign(r=r).groupby(["segmento", "h"], observed=True)["r"]
            .quantile([q, 1 - q]).unstack().rename(columns={q: "q_inf", 1 - q: "q_sup"}).reset_index())


def aplicar_intervalo(df: pd.DataFrame, cuant: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cuant, on=["segmento", "h"], how="left")
    out["inf"] = np.clip(np.expm1(np.log1p(out["lightgbm"]) + out["q_inf"]), 0, None)
    out["sup"] = np.expm1(np.log1p(out["lightgbm"]) + out["q_sup"])
    return out


def main():
    R = pd.read_csv(D.RESULTS / "backtest_predicciones.csv", keep_default_na=False)
    R["segmento"] = pd.Categorical(R["segmento"], categories=D.SEGMENTOS, ordered=True)
    provincias = D.leer_provincias().set_index("codigo_dgt")

    # ---- Error por tamaño de provincia ----
    tam = R.groupby("provincia")["y"].sum()
    grupos = pd.qcut(tam.rank(method="first"), 4, labels=["Pequeñas", "Medianas", "Grandes", "Muy grandes"])
    R["tamano"] = R["provincia"].map(grupos)
    por_tam = R.groupby("tamano", observed=True).apply(lambda g: pd.Series({
        "WAPE naive": Me.wape(g["y"], g["naive_estacional"]),
        "WAPE ETS": Me.wape(g["y"], g["ets"]) if "ets" in g else np.nan,
        "WAPE LightGBM": Me.wape(g["y"], g["lightgbm"]),
        "provincias": g["provincia"].nunique()}))
    por_tam.to_csv(D.RESULTS / "wape_por_tamano_provincia.csv")
    print("WAPE por tamaño de provincia\n" + por_tam.round(3).to_string())

    # ---- Error por mes objetivo ----
    R["mes_num"] = pd.PeriodIndex(R["mes_objetivo"], freq="M").month
    por_mes = R.groupby("mes_num").apply(lambda g: Me.wape(g["y"], g["lightgbm"]))
    por_mes.rename("wape").to_csv(D.RESULTS / "wape_por_mes.csv")

    # ---- Series con más error absoluto (dónde fallar cuesta más) ----
    err = R.assign(abs_err=(R["y"] - R["lightgbm"]).abs()).groupby(["provincia", "segmento"], observed=True).agg(
        error_abs=("abs_err", "sum"), real=("y", "sum"))
    err["wape"] = err["error_abs"] / err["real"]
    err["cuota_error"] = err["error_abs"] / err["error_abs"].sum()
    peores = err.sort_values("error_abs", ascending=False).head(10)
    peores = peores.reset_index().assign(provincia=lambda d: d["provincia"].map(provincias["provincia"]))
    peores.to_csv(D.RESULTS / "series_mayor_error.csv", index=False)
    print("\nSeries que concentran más error\n" + peores.round(3).to_string(index=False))

    # ---- Intervalos: calibrar en la 1.ª mitad, evaluar en la 2.ª ----
    origenes = sorted(R["origen"].unique())
    corte = origenes[len(origenes) // 2]
    cal, val = R[R["origen"] < corte], R[R["origen"] >= corte]
    cuant = cuantiles_ratio(cal)
    v = aplicar_intervalo(val, cuant)
    v["dentro"] = (v["y"] >= v["inf"]) & (v["y"] <= v["sup"])
    cobertura = v.groupby("segmento", observed=True)["dentro"].mean()
    cobertura["Total"] = v["dentro"].mean()
    cobertura.rename("cobertura_real").to_csv(D.RESULTS / "cobertura_intervalos.csv")
    print(f"\nCobertura real del intervalo del {NIVEL:.0%} (orígenes >= {corte}):\n" + cobertura.round(3).to_string())
    # Cuantiles definitivos con todo el backtest, para la previsión final
    cuantiles_ratio(R).to_csv(D.RESULTS / "cuantiles_intervalo.csv", index=False)

    # ---- Experimento: corregir el sesgo con el error reciente ----
    # El modelo se queda corto en un mercado que acelera (sesgo ~ +5 %). Se prueba
    # multiplicar cada previsión por real/previsto de los 12 meses anteriores
    # (solo con información disponible en el origen).
    filas = []
    for o, g in R.groupby("origen"):
        pasado = R[(R["mes_objetivo"] <= o) & (R["origen"] < o) &
                   (R["origen"] >= str(pd.Period(o, "M") - 12))]
        f = float(np.clip(pasado["y"].sum() / pasado["lightgbm"].sum(), 0.9, 1.15)) if len(pasado) > 500 else 1.0
        filas.append(g.assign(corregido=g["lightgbm"] * f))
    X = pd.concat(filas)
    exp = pd.DataFrame({
        "variante": ["LightGBM", "LightGBM con corrección de sesgo"],
        "wape": [Me.wape(X["y"], X["lightgbm"]), Me.wape(X["y"], X["corregido"])],
        "sesgo": [Me.sesgo(X["y"], X["lightgbm"]), Me.sesgo(X["y"], X["corregido"])]})
    exp.to_csv(D.RESULTS / "experimento_correccion_sesgo.csv", index=False)
    print("\nExperimento de corrección de sesgo (se descarta: empeora el WAPE)\n" + exp.round(4).to_string(index=False))

    # ---- Figuras ----
    comp = pd.read_csv(D.RESULTS / "comparativa_modelos.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colores = [GRIS, GRIS, "#86b6ef", AZUL][: len(comp)]
    ax.barh(comp["modelo"], comp["wape_provincia"] * 100, color=colores, height=0.6)
    for i, v_ in enumerate(comp["wape_provincia"] * 100):
        ax.text(v_, i, f" {v_:.1f} %", va="center", color=TEXTO_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WAPE provincia x segmento (menos es mejor)")
    ax.set_title("Error de previsión a 1-3 meses (backtest: oct 2024 - ago 2026)")
    guardar(fig, "08_comparativa_modelos.png")

    # Real frente a previsto a nivel nacional (h=1)
    h1 = R[R["h"] == 1].groupby("mes_objetivo")[["y", "lightgbm", "naive_estacional"]].sum()
    x = pd.PeriodIndex(h1.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(x, h1["y"], color="#0b0b0b", lw=2.2, marker="o", ms=5, label="Real")
    ax.plot(x, h1["lightgbm"], color=AZUL, lw=2.2, marker="o", ms=5, label="LightGBM (a 1 mes)")
    ax.plot(x, h1["naive_estacional"], color=GRIS, lw=1.6, ls="--", label="Mismo mes año anterior")
    ax.yaxis.set_major_formatter(lambda v_, _: f"{v_/1000:.0f} mil")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Total nacional: real frente a previsto un mes antes")
    guardar(fig, "09_real_vs_previsto.png")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ws = pd.read_csv(D.RESULTS / "wape_por_segmento.csv", index_col=0)
    xs = np.arange(len(ws))
    ax.bar(xs - 0.2, ws.iloc[:, 0] * 100, width=0.38, color=GRIS, label=ws.columns[0])
    ax.bar(xs + 0.2, ws["LightGBM global"] * 100, width=0.38,
           color=[COLOR_SEGMENTO[s] for s in ws.index], label="LightGBM global")
    ax.set_xticks(xs, [D.NOMBRE_SEGMENTO[s] for s in ws.index])
    ax.set_ylabel("WAPE (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Error por segmento: naive (gris) frente a LightGBM (color)")
    guardar(fig, "10_error_segmento.png")


if __name__ == "__main__":
    main()
