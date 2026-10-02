"""Fase 2 - Análisis exploratorio. Genera las figuras de la memoria y la defensa
en ``reports/figures`` y una tabla de hallazgos en ``results/hallazgos_eda.csv``.

    python pipeline/fase2_eda.py
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
from motoradar.estilo import AZUL, COLOR_SEGMENTO, GRIS, NARANJA, SUPERFICIE, TEXTO, TEXTO_2  # noqa: E402

MARCAS_CHINAS = ["ZONTES", "VOGE", "QJMOTOR", "CFMOTO", "KOVE", "CYCLONE"]
JAPONESAS = ["HONDA", "YAMAHA", "KAWASAKI", "SUZUKI"]

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": "#d6d5d0", "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2,
    "ytick.color": TEXTO_2, "text.color": TEXTO, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def miles(x, _=None):
    return f"{x/1000:.0f} mil" if x >= 1000 else f"{x:.0f}"


def guardar(fig, nombre):
    D.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(D.FIGURES / nombre)
    plt.close(fig)


def main():
    nac = D.leer_nacional_detalle()
    nuevas = nac[nac["estado"] == "Nueva"]
    prov = D.leer_provincia_segmento()
    marcas = D.leer_marcas_mensual()
    provincias = D.leer_provincias().set_index("codigo_dgt")
    hallazgos = []

    # 1. Evolución del mercado
    total = nuevas.groupby("mes")["matriculaciones"].sum()
    ts = total.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(ts, total.values, color="#b7d3f6", lw=1.5, label="Mensual")
    ax.plot(ts, total.rolling(12).mean().values, color=AZUL, lw=2.5, label="Media móvil 12 meses")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-31"), color="#ecebe7", zorder=0)
    ax.text(pd.Timestamp("2020-04-15"), total.max() * 0.95, "Confinamiento", ha="center", color=TEXTO_2, fontsize=9)
    ax.yaxis.set_major_formatter(miles)
    ax.set_title("Matriculaciones de motos nuevas en España, 2015-2026")
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, "01_evolucion_mercado.png")
    anual = nuevas.groupby(nuevas["mes"].dt.year)["matriculaciones"].sum()
    hallazgos.append(("Crecimiento 2015-2025", f"{anual[2015]:,} -> {anual[2025]:,} (+{anual[2025]/anual[2015]-1:.0%})"))
    ene_ago = nuevas[nuevas["mes"].dt.month <= 8].groupby(nuevas["mes"].dt.year)["matriculaciones"].sum()
    hallazgos.append(("Enero-agosto 2026 frente a 2025", f"{ene_ago[2026]:,} frente a {ene_ago[2025]:,} ({ene_ago[2026]/ene_ago[2025]-1:+.1%})"))

    # 2. Mix por segmento
    mix = nuevas.pivot_table(index=nuevas["mes"].dt.year, columns="segmento", values="matriculaciones",
                             aggfunc="sum", observed=True)
    cuota = mix.div(mix.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for seg in D.SEGMENTOS:
        ax.plot(cuota.index, cuota[seg], color=COLOR_SEGMENTO[seg], lw=2.5, marker="o", ms=5)
        ax.annotate(D.NOMBRE_SEGMENTO[seg], (cuota.index[-1], cuota[seg].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=TEXTO_2, fontsize=10)
    ax.set_ylabel("% de las matriculaciones")
    ax.set_xlim(cuota.index[0] - 0.3, cuota.index[-1] + 2.2)
    ax.set_title("Cuota de cada segmento de carnet (2026: enero-agosto)")
    guardar(fig, "02_mix_segmentos.png")
    hallazgos.append(("Cuota ciclomotores (AM)", f"{cuota.loc[2015,'AM']:.1f} % en 2015 -> {cuota.loc[2026,'AM']:.1f} % en 2026"))
    hallazgos.append(("Cuota A2", f"{cuota.loc[2015,'A2']:.1f} % en 2015 -> {cuota.loc[2026,'A2']:.1f} % en 2026"))

    # 3. Estacionalidad
    base = nuevas[(nuevas["mes"] >= pd.Period("2022-01", "M")) & (nuevas["mes"] <= pd.Period("2025-12", "M"))]
    est = base.pivot_table(index=base["mes"].dt.month, columns="segmento", values="matriculaciones",
                           aggfunc="sum", observed=True)
    indice = est / est.mean() * 100
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for seg in D.SEGMENTOS:
        ax.plot(indice.index, indice[seg], color=COLOR_SEGMENTO[seg], lw=2.2, marker="o", ms=5,
                label=D.NOMBRE_SEGMENTO[seg])
    ax.axhline(100, color=GRIS, lw=1, ls="--")
    ax.set_xticks(range(1, 13), ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
    ax.set_ylabel("Índice (media del año = 100)")
    ax.set_title("Estacionalidad por segmento (2022-2025)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    guardar(fig, "03_estacionalidad.png")
    pico = indice.idxmax()
    hallazgos.append(("Mes pico por segmento", ", ".join(f"{s}: {pico[s]}" for s in D.SEGMENTOS)))

    # 4. Marcas chinas frente a japonesas
    m = marcas.assign(anio=marcas["mes"].dt.year)
    tot = m.groupby("anio")["matriculaciones"].sum()
    china = m[m["marca"].isin(MARCAS_CHINAS)].groupby("anio")["matriculaciones"].sum().reindex(tot.index, fill_value=0) / tot * 100
    japon = m[m["marca"].isin(JAPONESAS)].groupby("anio")["matriculaciones"].sum() / tot * 100
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(japon.index, japon, color=AZUL, lw=2.5, marker="o", ms=5)
    ax.plot(china.index, china, color=NARANJA, lw=2.5, marker="o", ms=5)
    ax.annotate("Japonesas (Honda, Yamaha,\nKawasaki, Suzuki)", (japon.index[-1], japon.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=TEXTO_2, fontsize=10)
    ax.annotate("Nuevas chinas (Zontes, Voge, QJMotor,\nCFMoto, Kove, Cyclone)", (china.index[-1], china.iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center", color=TEXTO_2, fontsize=10)
    ax.set_xlim(china.index[0] - 0.3, china.index[-1] + 3.2)
    ax.set_ylabel("% de las matriculaciones")
    ax.set_title("La irrupción de las marcas chinas (cuota anual; 2026: enero-agosto)")
    guardar(fig, "04_marcas_chinas.png")
    hallazgos.append(("Cuota nuevas marcas chinas", f"{china.loc[2020]:.1f} % en 2020 -> {china.loc[2026]:.1f} % en 2026"))

    # 5. Ranking de marcas últimos 12 meses
    ult = marcas[marcas["mes"] > marcas["mes"].max() - 12]
    rk = ult[ult["marca"] != "OTRAS"].groupby("marca")["matriculaciones"].sum().sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colores = [NARANJA if b in MARCAS_CHINAS else AZUL for b in rk.index]
    ax.barh(rk.index, rk.values, color=colores, height=0.7)
    for i, v in enumerate(rk.values):
        ax.text(v, i, f" {v:,.0f}".replace(",", "."), va="center", color=TEXTO_2, fontsize=9)
    ax.xaxis.set_major_formatter(miles)
    ax.grid(axis="y", visible=False)
    ax.set_title("Marcas más matriculadas (sep 2025 - ago 2026)")
    ax.text(0, -1.4, "Naranja: nuevas marcas chinas", color=TEXTO_2, fontsize=9)
    guardar(fig, "05_ranking_marcas.png")

    # 6. Provincias: volumen y crecimiento
    p12 = prov[prov["mes"] > prov["mes"].max() - 12].groupby("provincia")["matriculaciones"].sum()
    p12a = prov[(prov["mes"] > prov["mes"].max() - 24) & (prov["mes"] <= prov["mes"].max() - 12)].groupby("provincia")["matriculaciones"].sum()
    top = p12.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh([provincias.loc[c, "provincia"] for c in top.index], top.values, color=AZUL, height=0.7)
    for i, c in enumerate(top.index):
        ax.text(top[c], i, f" {top[c]:,.0f}  ({p12[c]/p12a[c]-1:+.0%})".replace(",", "."), va="center",
                color=TEXTO_2, fontsize=9)
    ax.xaxis.set_major_formatter(miles)
    ax.grid(axis="y", visible=False)
    ax.set_title("Provincias con más matriculaciones (12 meses) y variación anual")
    guardar(fig, "06_provincias.png")
    conc = p12.sort_values(ascending=False)
    hallazgos.append(("Concentración geográfica", f"Barcelona + Madrid = {(conc.iloc[:2].sum()/conc.sum()):.0%} del mercado; top 10 = {(conc.iloc[:10].sum()/conc.sum()):.0%}"))

    # 7. Eléctricas y automatriculaciones
    tipo = nuevas.assign(anio=nuevas["mes"].dt.year)
    elec = tipo.pivot_table(index="anio", columns="propulsion", values="matriculaciones", aggfunc="sum")
    emp = tipo.pivot_table(index="anio", columns="titular", values="matriculaciones", aggfunc="sum")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    s = elec["Eléctrica"] / elec.sum(axis=1) * 100
    axes[0].bar(s.index, s.values, color=AZUL, width=0.7)
    axes[0].set_title("Cuota de motos eléctricas")
    axes[0].set_ylabel("%")
    s2 = emp["Empresa"] / emp.sum(axis=1) * 100
    axes[1].bar(s2.index, s2.values, color=NARANJA, width=0.7)
    axes[1].set_title("Matriculadas a nombre de empresa")
    axes[1].set_ylabel("%")
    for a in axes:
        a.grid(axis="x", visible=False)
    guardar(fig, "07_electricas_empresas.png")
    hallazgos.append(("Cuota eléctricas", f"máximo {s.max():.1f} % en {s.idxmax()}; {s.loc[2026]:.1f} % en 2026"))
    hallazgos.append(("Matriculadas por empresas", f"{s2.loc[2020]:.1f} % en 2020 -> {s2.loc[2026]:.1f} % en 2026 (renting, flotas y automatriculaciones)"))

    # 8. Picos de automatriculación: diciembre 2016 y 2020 (cambios de norma Euro)
    emp_mes = nuevas[nuevas["titular"] == "Empresa"].groupby("mes")["matriculaciones"].sum()
    picos = emp_mes.sort_values(ascending=False).head(5)
    hallazgos.append(("Meses con más motos a nombre de empresa",
                      ", ".join(f"{p} ({v:,})" for p, v in picos.items())))

    pd.DataFrame(hallazgos, columns=["hallazgo", "dato"]).to_csv(D.RESULTS / "hallazgos_eda.csv", index=False)
    for h, d in hallazgos:
        print(f"- {h}: {d}")


if __name__ == "__main__":
    main()
