"""Fase 5 - Modelo final y previsión que consume la app.

Entrena LightGBM con toda la historia disponible (hasta agosto de 2026), predice
los 3 meses siguientes para las 208 series provincia x segmento y añade el
intervalo del 80 % calibrado en la fase 4.

Salidas:
    models/lgbm_final.txt              modelo en formato nativo de LightGBM
    models/metadata.json               fecha de corte, variables, métricas
    data/gold/previsiones.csv          previsión por provincia, segmento y mes
    results/importancia_variables.csv  importancia (ganancia) de cada variable

    python pipeline/fase5_modelo_final.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motoradar import datos as D  # noqa: E402
from motoradar import modelos as Mo  # noqa: E402
from motoradar.estilo import AZUL, TEXTO_2  # noqa: E402
from motoradar.features import VARIABLES, construir_tabla  # noqa: E402
from pipeline.fase2_eda import guardar  # noqa: E402
from pipeline.fase4_analisis_errores import aplicar_intervalo  # noqa: E402

NOMBRES = {
    "mismo_mes_anio_ant_rel": "Mismo mes del año anterior", "lag0_rel": "Último mes conocido",
    "media_3m_rel": "Media de los 3 últimos meses", "mes_num": "Mes del año", "mes_cos": "Mes (coseno)",
    "mes_sin": "Mes (seno)", "log_nivel": "Tamaño de la serie", "tendencia_anual": "Tendencia anual",
    "yoy_3m": "Ritmo 3 meses vs año anterior", "dias_laborables": "Días laborables del mes",
    "dias_lab_anio_ant": "Días laborables año anterior", "mismo_mes_hace2_rel": "Mismo mes hace 2 años",
    "segmento_cod": "Segmento", "provincia_cod": "Provincia", "h": "Horizonte", "origen_covid": "Historia con COVID",
}


def main():
    df = D.leer_provincia_segmento()
    ultimo = D.meses()[-1]
    tabla = construir_tabla(df, con_objetivo=False)
    train = tabla[tabla["mes_objetivo"] <= ultimo]
    futuro = tabla[(tabla["origen"] == ultimo)].copy()

    modelo = Mo.entrenar_lgbm(train)
    D.MODELS.mkdir(exist_ok=True)
    modelo.booster_.save_model(str(D.MODELS / "lgbm_final.txt"))

    futuro["lightgbm"] = Mo.predecir_lgbm(modelo, futuro)
    futuro["naive_estacional"] = Mo.naive_estacional(futuro)
    cuant = pd.read_csv(D.RESULTS / "cuantiles_intervalo.csv")
    cuant["segmento"] = pd.Categorical(cuant["segmento"], categories=D.SEGMENTOS, ordered=True)
    futuro = aplicar_intervalo(futuro, cuant)

    # Mismo mes del año anterior real, para dar contexto en la app
    real = df.set_index(["provincia", "segmento", "mes"])["matriculaciones"]
    futuro["real_anio_anterior"] = [real.get((p, s, m - 12), np.nan) for p, s, m in
                                    zip(futuro["provincia"], futuro["segmento"], futuro["mes_objetivo"])]
    prov = D.leer_provincias().set_index("codigo_dgt")
    salida = pd.DataFrame({
        "mes": futuro["mes_objetivo"].astype(str), "h": futuro["h"], "provincia": futuro["provincia"],
        "nombre_provincia": futuro["provincia"].map(prov["provincia"]), "segmento": futuro["segmento"].astype(str),
        "prevision": futuro["lightgbm"].round(1), "inf_80": futuro["inf"].round(1), "sup_80": futuro["sup"].round(1),
        "naive_estacional": futuro["naive_estacional"].round(1),
        "real_anio_anterior": futuro["real_anio_anterior"],
    }).sort_values(["provincia", "segmento", "mes"])
    D.GOLD.mkdir(exist_ok=True)
    salida.to_csv(D.GOLD / "previsiones.csv", index=False)

    imp = pd.DataFrame({"variable": VARIABLES,
                        "ganancia": modelo.booster_.feature_importance("gain")}).sort_values("ganancia", ascending=False)
    imp["ganancia_rel"] = imp["ganancia"] / imp["ganancia"].sum()
    imp.to_csv(D.RESULTS / "importancia_variables.csv", index=False)
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh([NOMBRES.get(v, v.replace("_rel", "").replace("lag", "Hace ") + (" meses" if v.startswith("lag") else ""))
             for v in top["variable"]], top["ganancia_rel"] * 100, color=AZUL, height=0.65)
    for i, v in enumerate(top["ganancia_rel"] * 100):
        ax.text(v, i, f" {v:.1f} %", va="center", color=TEXTO_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% de la ganancia total")
    ax.set_title("Qué mira el modelo: importancia de las variables")
    guardar(fig, "11_importancia_variables.png")

    comp = pd.read_csv(D.RESULTS / "comparativa_modelos.csv").set_index("modelo")
    boot = pd.read_csv(D.RESULTS / "mejora_bootstrap.csv").set_index("referencia")
    cob = pd.read_csv(D.RESULTS / "cobertura_intervalos.csv", index_col=0)
    meta = {
        "fecha_entrenamiento": date.today().isoformat(),
        "datos_hasta": str(ultimo),
        "meses_previstos": sorted(salida["mes"].unique().tolist()),
        "filas_entrenamiento": int(len(train.dropna(subset=["objetivo_rel"]))),
        "series": int(futuro["serie"].nunique()),
        "variables": VARIABLES,
        "parametros": Mo.PARAMS_LGBM,
        "backtest": {
            "origenes": "2024-09 a 2026-07",
            "wape_provincia_lightgbm": round(float(comp.loc["LightGBM global", "wape_provincia"]), 4),
            "wape_provincia_naive": round(float(comp.loc["Naive estacional (año anterior)", "wape_provincia"]), 4),
            "wape_nacional_lightgbm": round(float(comp.loc["LightGBM global", "wape_nacional"]), 4),
            "mejora_vs_naive": round(float(boot.loc["Naive estacional (año anterior)", "mejora"]), 4),
            "mejora_vs_naive_ic95": [round(float(boot.loc["Naive estacional (año anterior)", c]), 4) for c in ("ic95_inf", "ic95_sup")],
            "cobertura_intervalo_80": round(float(cob.loc["Total"].iloc[0]), 3),
        },
    }
    if "Holt-Winters (ETS)" in comp.index:
        meta["backtest"]["wape_provincia_ets"] = round(float(comp.loc["Holt-Winters (ETS)", "wape_provincia"]), 4)
        meta["backtest"]["mejora_vs_ets"] = round(float(boot.loc["Holt-Winters (ETS)", "mejora"]), 4)
    (D.MODELS / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    nac = salida.groupby(["mes", "segmento"])[["prevision", "real_anio_anterior"]].sum()
    print("Previsión nacional por segmento\n" + nac.round(0).to_string())
    print("\n" + json.dumps(meta["backtest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
