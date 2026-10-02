"""
MotoRadar - Previsión de demanda de motos para concesionarios
TFM Máster en Data Science & IA (Evolve Academy) - Juanjo

Cinco vistas:
  Previsión  - demanda prevista en la provincia para los próximos 3 meses.
  Pedido     - cuántas motos pedir por segmento según tu cuota y tu stock.
  Mercado    - cómo está cambiando el mercado (segmentos, marcas, modelos).
  Fiabilidad - cuánto se equivoca el modelo y frente a qué se compara.
  Carnets    - qué motos puede llevar un cliente y qué carnet le falta.

Datos: microdatos de matriculaciones de la DGT (enero 2015 - agosto 2026).
Ejecutar:  python -m streamlit run app.py
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from motoradar import datos as D
from motoradar.estilo import AZUL, COLOR_SEGMENTO, GRIS, NARANJA
from motoradar.negocio import ESCENARIOS, Conductor, cuota_mercado, plan_pedido, segmentos_permitidos, siguiente_paso

st.set_page_config(page_title="MotoRadar", page_icon="🏍️", layout="wide")

MESES_ES = {1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun", 7: "jul", 8: "ago",
            9: "sep", 10: "oct", 11: "nov", 12: "dic"}
MARCAS_CHINAS = ["ZONTES", "VOGE", "QJMOTOR", "CFMOTO", "KOVE", "CYCLONE"]


def nombre_mes(p: pd.Period, largo=False) -> str:
    return f"{MESES_ES[p.month]} {p.year}" if largo else f"{MESES_ES[p.month]} {str(p.year)[2:]}"


def fmt(x) -> str:
    return f"{x:,.0f}".replace(",", ".")


# ------------------------------------------------------------------ datos
@st.cache_data
def cargar():
    prov = D.leer_provincia_segmento()
    nac = D.leer_nacional_detalle()
    marcas = D.leer_marcas_mensual()
    marcas_seg = D.leer_marcas_segmento_12m()
    modelos = D.leer_top_modelos()
    provincias = D.leer_provincias()
    prev = pd.read_csv(D.GOLD / "previsiones.csv", keep_default_na=False)
    prev["mes"] = pd.PeriodIndex(prev["mes"], freq="M")
    bt = pd.read_csv(D.RESULTS / "backtest_predicciones.csv", keep_default_na=False)
    bt["mes_objetivo"] = pd.PeriodIndex(bt["mes_objetivo"], freq="M")
    meta = json.loads((D.MODELS / "metadata.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(D.RESULTS / "comparativa_modelos.csv")
    seg_err = pd.read_csv(D.RESULTS / "wape_por_segmento.csv", index_col=0)
    return prov, nac, marcas, marcas_seg, modelos, provincias, prev, bt, meta, comp, seg_err


prov, nac, marcas, marcas_seg, modelos, provincias, prev, bt, meta, comp, seg_err = cargar()
ULTIMO = prov["mes"].max()
NOMBRE_PROV = dict(zip(provincias["codigo_dgt"], provincias["provincia"]))
ORDEN_PROV = sorted(NOMBRE_PROV, key=lambda c: NOMBRE_PROV[c])


def estilo(fig, alto=360, titulo=None, fechas=False):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
                      title=dict(text=titulo, x=0, font=dict(size=16)) if titulo else None,
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif"))
    fig.update_xaxes(showgrid=False)
    if fechas:
        fig.update_xaxes(tickformat="%m/%Y", hoverformat="%m/%Y")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", separatethousands=True)
    return fig


# ------------------------------------------------------------------ cabecera
st.title("🏍️ MotoRadar")
st.caption(f"Previsión de demanda de motos por provincia y carnet · Datos oficiales de la DGT hasta "
           f"{nombre_mes(ULTIMO, True)} · Previsión de {nombre_mes(ULTIMO + 1, True)} a {nombre_mes(ULTIMO + 3, True)}")

with st.sidebar:
    st.header("Tu concesionario")
    provincia = st.selectbox("Provincia", ORDEN_PROV, index=ORDEN_PROV.index("M"),
                             format_func=lambda c: NOMBRE_PROV[c])
    st.divider()
    st.markdown("**Segmentos por carnet**")
    for s in D.SEGMENTOS:
        st.markdown(f"<span style='color:{COLOR_SEGMENTO[s]}'>●</span> **{D.NOMBRE_SEGMENTO[s]}**  \n"
                    f"<small>{D.DESCRIPCION_SEGMENTO[s]}</small>", unsafe_allow_html=True)
    st.divider()
    st.caption(f"Modelo LightGBM entrenado con {meta['filas_entrenamiento']:,} ejemplos. "
               f"Error medio en provincia: {meta['backtest']['wape_provincia_lightgbm']:.1%}.".replace(",", "."))

p_prov = prev[prev["provincia"] == provincia]
hist_prov = prov[prov["provincia"] == provincia]

tab_prev, tab_pedido, tab_mercado, tab_fiab, tab_carnet = st.tabs(
    ["📈 Previsión", "📦 Pedido", "🌍 Mercado", "🎯 Fiabilidad", "🪪 Carnets"])

# ================================================================== PREVISIÓN
with tab_prev:
    st.subheader(f"Demanda prevista en {NOMBRE_PROV[provincia]}")
    tot = p_prov.groupby("mes")[["prevision", "inf_80", "sup_80", "real_anio_anterior"]].sum()
    cols = st.columns(3)
    for col, (mes, fila) in zip(cols, tot.iterrows()):
        var = fila["prevision"] / fila["real_anio_anterior"] - 1 if fila["real_anio_anterior"] else np.nan
        col.metric(f"Motos nuevas en {nombre_mes(mes, True)}", fmt(fila["prevision"]),
                   f"{var:+.0%} frente a {nombre_mes(mes - 12)}" if pd.notna(var) else None)
        col.caption(f"Rango probable (80 %): {fmt(fila['inf_80'])} - {fmt(fila['sup_80'])}")

    seg_sel = st.segmented_control("Segmento", ["Todos"] + D.SEGMENTOS, default="Todos",
                                   format_func=lambda s: "Todos" if s == "Todos" else D.NOMBRE_SEGMENTO[s])
    seg_sel = seg_sel or "Todos"
    desde = ULTIMO - 35
    fig = go.Figure()
    segs = D.SEGMENTOS if seg_sel == "Todos" else [seg_sel]
    for s in segs:
        h = hist_prov[(hist_prov["segmento"] == s) & (hist_prov["mes"] >= desde)]
        f = p_prov[p_prov["segmento"] == s].sort_values("mes")
        x_h = h["mes"].dt.to_timestamp()
        x_f = f["mes"].dt.to_timestamp()
        color = COLOR_SEGMENTO[s]
        if len(segs) == 1:
            fig.add_trace(go.Scatter(x=list(x_f) + list(x_f[::-1]), y=list(f["sup_80"]) + list(f["inf_80"][::-1]),
                                     fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                                     hoverinfo="skip", name="Rango 80 %"))
        fig.add_trace(go.Scatter(x=x_h, y=h["matriculaciones"], name=D.NOMBRE_SEGMENTO[s], mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate="%{y:,.0f} motos<extra>" + s + " real</extra>"))
        ult = h.iloc[-1:]
        fig.add_trace(go.Scatter(x=list(ult["mes"].dt.to_timestamp()) + list(x_f),
                                 y=list(ult["matriculaciones"]) + list(f["prevision"]),
                                 mode="lines+markers", line=dict(color=color, width=2, dash="dot"),
                                 marker=dict(size=8), showlegend=False,
                                 hovertemplate="%{y:,.0f} motos<extra>" + s + " previsión</extra>"))
    fig.add_vline(x=ULTIMO.to_timestamp(how="end"), line=dict(color=GRIS, dash="dash", width=1))
    st.plotly_chart(estilo(fig, 400, "Matriculaciones mensuales: real (línea) y previsión (puntos)", fechas=True),
                    width="stretch")

    tabla = p_prov.assign(Mes=p_prov["mes"].map(lambda m: nombre_mes(m, True)),
                          Segmento=p_prov["segmento"].map(D.NOMBRE_SEGMENTO))
    tabla = tabla[["Mes", "Segmento", "prevision", "inf_80", "sup_80", "real_anio_anterior"]].rename(columns={
        "prevision": "Previsión", "inf_80": "Mínimo probable", "sup_80": "Máximo probable",
        "real_anio_anterior": "Mismo mes año anterior"})
    with st.expander("Ver la tabla de previsiones"):
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tabla.columns[2:]})
    st.info("La previsión es de motos **nuevas matriculadas en la provincia** (todas las marcas). "
            "El rango cubre el 80 % de los casos: en la validación, el valor real cayó dentro "
            f"el {meta['backtest']['cobertura_intervalo_80']:.0%} de las veces.")

# ================================================================== PEDIDO
with tab_pedido:
    st.subheader("Planificador de pedido")
    st.write("Introduce tus ventas de los últimos 12 meses y tu stock. MotoRadar calcula tu cuota en "
             f"{NOMBRE_PROV[provincia]}, aplica la previsión del mercado y sugiere cuántas motos pedir.")
    mercado12 = (hist_prov[hist_prov["mes"] > ULTIMO - 12].groupby("segmento", observed=True)["matriculaciones"]
                 .sum().to_dict())
    c1, c2 = st.columns([3, 2])
    with c1:
        base = pd.DataFrame({
            "Segmento": [D.NOMBRE_SEGMENTO[s] for s in D.SEGMENTOS],
            "Mercado provincial 12 meses": [mercado12.get(s, 0) for s in D.SEGMENTOS],
            "Tus ventas 12 meses": [round(mercado12.get(s, 0) * 0.03) for s in D.SEGMENTOS],
            "Tu stock actual": [round(mercado12.get(s, 0) * 0.03 / 12) for s in D.SEGMENTOS],
        })
        editado = st.data_editor(base, hide_index=True, width="stretch", disabled=["Segmento", "Mercado provincial 12 meses"],
                                 column_config={"Mercado provincial 12 meses": st.column_config.NumberColumn(format="%d"),
                                                "Tus ventas 12 meses": st.column_config.NumberColumn(min_value=0, step=1),
                                                "Tu stock actual": st.column_config.NumberColumn(min_value=0, step=1)})
        st.caption("Valores de ejemplo: un concesionario con el 3 % del mercado y un mes de stock.")
    with c2:
        cobertura = st.slider("Meses que quieres cubrir", 1, 3, 2)
        escenario = st.radio("Escenario", list(ESCENARIOS), index=1, horizontal=True,
                             help="Prudente y Optimista usan los extremos del rango del 80 %.")
    ventas = dict(zip(D.SEGMENTOS, editado["Tus ventas 12 meses"].fillna(0)))
    stock = dict(zip(D.SEGMENTOS, editado["Tu stock actual"].fillna(0).astype(int)))
    plan = plan_pedido(p_prov, cuota_mercado(ventas, mercado12), stock, cobertura, escenario)
    cols = st.columns(4)
    for col, (_, f) in zip(cols, plan.set_index("segmento").reindex(D.SEGMENTOS).reset_index().iterrows()):
        col.metric(D.NOMBRE_SEGMENTO[f["segmento"]], f"{int(f['pedido_sugerido'])} motos",
                   f"cuota {f['cuota']:.1%} · vendes ~{f['ventas_esperadas']:.0f}", delta_color="off", delta_arrow="off")
    st.success(f"**Pedido total sugerido: {int(plan['pedido_sugerido'].sum())} motos** para cubrir "
               f"{cobertura} mes(es) en escenario {escenario.lower()}.")
    st.caption("Cálculo: mercado previsto × tu cuota − tu stock, redondeando hacia arriba. "
               "Supone que mantienes tu cuota; si vas a lanzar una marca nueva o una campaña, ajústala.")

# ================================================================== MERCADO
with tab_mercado:
    st.subheader("Cómo está cambiando el mercado")
    nuevas = nac[nac["estado"] == "Nueva"]
    tot_mes = nuevas.groupby("mes")["matriculaciones"].sum()
    ult12, ant12 = tot_mes.iloc[-12:].sum(), tot_mes.iloc[-24:-12].sum()
    cuota_china = (marcas[(marcas["mes"] > ULTIMO - 12) & marcas["marca"].isin(MARCAS_CHINAS)]["matriculaciones"].sum()
                   / marcas[marcas["mes"] > ULTIMO - 12]["matriculaciones"].sum())
    elec = nuevas[nuevas["mes"] > ULTIMO - 12]
    cuota_elec = elec[elec["propulsion"] == "Eléctrica"]["matriculaciones"].sum() / elec["matriculaciones"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Motos nuevas, últimos 12 meses", fmt(ult12), f"{ult12 / ant12 - 1:+.1%} interanual")
    k2.metric("Récord mensual", fmt(tot_mes.max()), nombre_mes(tot_mes.idxmax(), True), delta_color="off", delta_arrow="off")
    k3.metric("Cuota nuevas marcas chinas", f"{cuota_china:.1%}", "Zontes, Voge, QJMotor, CFMoto...", delta_color="off", delta_arrow="off")
    k4.metric("Cuota eléctricas", f"{cuota_elec:.1%}", "últimos 12 meses", delta_color="off", delta_arrow="off")

    c1, c2 = st.columns(2)
    with c1:
        anual = nuevas.assign(anio=nuevas["mes"].dt.year).pivot_table(
            index="anio", columns="segmento", values="matriculaciones", aggfunc="sum", observed=True)
        fig = go.Figure()
        for s in D.SEGMENTOS:
            fig.add_trace(go.Bar(x=anual.index, y=anual[s], name=D.NOMBRE_SEGMENTO[s], marker_color=COLOR_SEGMENTO[s],
                                 hovertemplate="%{y:,.0f}<extra>" + s + "</extra>"))
        fig.update_layout(barmode="stack", bargap=0.25)
        fig.update_traces(marker_line=dict(width=1, color="white"))
        st.plotly_chart(estilo(fig, 380, "Motos nuevas por año y segmento (2026: ene-ago)"), width="stretch")
    with c2:
        m12 = marcas[(marcas["mes"] > ULTIMO - 12) & (marcas["marca"] != "OTRAS")]
        seg_f = st.selectbox("Ranking de marcas: segmento", ["Todos"] + D.SEGMENTOS,
                             format_func=lambda s: "Todos" if s == "Todos" else D.NOMBRE_SEGMENTO[s])
        if seg_f == "Todos":
            rk = m12.groupby("marca")["matriculaciones"].sum()
        else:
            rk = marcas_seg[marcas_seg["marca"] != "OTRAS"].set_index("marca")[seg_f]
        rk = rk.sort_values().tail(12)
        fig = go.Figure(go.Bar(x=rk.values, y=rk.index, orientation="h",
                               marker_color=[NARANJA if b in MARCAS_CHINAS else AZUL for b in rk.index],
                               hovertemplate="%{x:,.0f} motos<extra></extra>"))
        st.plotly_chart(estilo(fig, 380, "Marcas más vendidas, 12 meses (naranja: nuevas chinas)"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        p12 = prov[prov["mes"] > ULTIMO - 12].groupby("provincia")["matriculaciones"].sum()
        pa = prov[(prov["mes"] > ULTIMO - 24) & (prov["mes"] <= ULTIMO - 12)].groupby("provincia")["matriculaciones"].sum()
        mapa = provincias.set_index("codigo_dgt").assign(motos=p12, var=(p12 / pa - 1) * 100).reset_index()
        fig = go.Figure(go.Scattergeo(
            lon=mapa["lon"], lat=mapa["lat"], text=mapa["provincia"], customdata=mapa[["motos", "var"]],
            marker=dict(size=np.sqrt(mapa["motos"]) / 4.5, color=mapa["var"], colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, AZUL]],
                        cmid=0, cmin=-30, cmax=30, line=dict(width=1, color="white"),
                        colorbar=dict(title="% anual", thickness=10)),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]:,.0f} motos en 12 meses<br>%{customdata[1]:+.0f} % interanual<extra></extra>"))
        fig.update_geos(scope="europe", lonaxis_range=[-18.5, 4.8], lataxis_range=[27.3, 44.2],
                        showcountries=True, countrycolor="rgba(128,128,128,0.4)", showland=True,
                        landcolor="rgba(128,128,128,0.08)", showocean=False, bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(estilo(fig, 420, "Mercado por provincia: tamaño = volumen, color = crecimiento"), width="stretch")
    with c4:
        seg_m = st.selectbox("Modelos más vendidos: segmento", D.SEGMENTOS, index=1, format_func=D.NOMBRE_SEGMENTO.get)
        tm = modelos[modelos["segmento"] == seg_m].head(10)
        st.dataframe(tm[["marca", "modelo", "matriculaciones", "kw_medio", "cc_medio"]].rename(columns={
            "marca": "Marca", "modelo": "Modelo", "matriculaciones": "Motos (12 m)", "kw_medio": "kW", "cc_medio": "cc"}),
            hide_index=True, width="stretch", height=390)

# ================================================================== FIABILIDAD
with tab_fiab:
    st.subheader("¿Cuánto me puedo fiar de la previsión?")
    b = meta["backtest"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Error medio por provincia y segmento", f"{b['wape_provincia_lightgbm']:.1%}",
              f"{-b['mejora_vs_naive']:.0%} frente a 'como el año pasado'", delta_color="inverse")
    k2.metric("Error en el total nacional", f"{b['wape_nacional_lightgbm']:.1%}")
    k3.metric("Mejora frente a Holt-Winters", f"{b.get('mejora_vs_ets', np.nan):.0%}")
    k4.metric("Aciertos del rango del 80 %", f"{b['cobertura_intervalo_80']:.0%}")
    st.caption("Validación de origen móvil: durante 23 meses (sep 2024 - jul 2026) se reentrenó el modelo cada mes "
               "solo con los datos disponibles en ese momento y se predijeron los 3 meses siguientes. "
               "Error = WAPE: de cada 100 motos matriculadas, cuántas se equivoca la previsión.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=comp["wape_provincia"] * 100, y=comp["modelo"], orientation="h",
                               marker_color=[GRIS, GRIS, "#86b6ef", AZUL][: len(comp)],
                               text=[f"{v:.1f} %" for v in comp["wape_provincia"] * 100], textposition="outside",
                               hovertemplate="%{x:.1f} %<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, comp["wape_provincia"].max() * 100 * 1.2])
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(estilo(fig, 320, "Error de cada modelo (menos es mejor)"), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[D.NOMBRE_SEGMENTO[s] for s in seg_err.index], y=seg_err.iloc[:, 0] * 100,
                             name="Como el año pasado", marker_color=GRIS))
        fig.add_trace(go.Bar(x=[D.NOMBRE_SEGMENTO[s] for s in seg_err.index], y=seg_err["LightGBM global"] * 100,
                             name="MotoRadar", marker_color=[COLOR_SEGMENTO[s] for s in seg_err.index]))
        fig.update_layout(barmode="group")
        st.plotly_chart(estilo(fig, 320, "Error por segmento (%)"), width="stretch")
    b_p = bt[(bt["provincia"] == provincia) & (bt["h"] == 1)].groupby("mes_objetivo")[["y", "lightgbm", "naive_estacional"]].sum()
    x = b_p.index.to_timestamp()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=b_p["y"], name="Real", line=dict(color="#555", width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=b_p["lightgbm"], name="MotoRadar (1 mes antes)", line=dict(color=AZUL, width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=b_p["naive_estacional"], name="Como el año pasado", line=dict(color=GRIS, dash="dash")))
    st.plotly_chart(estilo(fig, 360, f"{NOMBRE_PROV[provincia]}: lo que pasó frente a lo que se predijo", fechas=True), width="stretch")
    st.warning("Limitaciones: el mercado ha crecido más rápido que nunca y el modelo tiende a quedarse algo corto "
               "(≈5 %). No anticipa automatriculaciones masivas por cambios de normativa (p. ej. diciembre de 2024, "
               "antes de la Euro 5+) ni lanzamientos de marcas nuevas.")

# ================================================================== CARNETS
with tab_carnet:
    st.subheader("Asesor de carnet para el vendedor")
    st.write("Qué motos puede llevar un cliente con sus permisos y qué le falta para subir de segmento.")
    c1, c2 = st.columns([1, 2])
    with c1:
        edad = st.number_input("Edad del cliente", 14, 90, 25)
        permisos = {}
        for p_ in ["B", "AM", "A1", "A2", "A"]:
            if st.checkbox(f"Tiene permiso {p_}", value=(p_ == "B")):
                permisos[p_] = st.number_input(f"Años con el {p_}", 0, 70, 4 if p_ == "B" else 1, key=f"anios_{p_}")
    with c2:
        ok = segmentos_permitidos(Conductor(edad, permisos))
        for s in D.SEGMENTOS:
            if s in ok:
                st.success(f"**{D.NOMBRE_SEGMENTO[s]}** - {ok[s]}")
            else:
                st.error(f"**{D.NOMBRE_SEGMENTO[s]}** - no puede conducirlas")
        st.info(siguiente_paso(Conductor(edad, permisos)))
        st.caption("Resumen orientativo de la normativa de permisos (Reglamento General de Conductores). "
                   "Consulta siempre la DGT para casos concretos.")

st.divider()
st.caption("MotoRadar · TFM de Juanjo · Máster en Data Science & IA, Evolve Academy · "
           "Fuente: DGT, microdatos de matriculaciones (agregados) · Código en GitHub")
