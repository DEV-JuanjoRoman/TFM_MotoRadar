"""Pruebas del proyecto. Se ejecutan con ``python tests/test_motoradar.py`` (o pytest).

Cubren: integridad de los datos, que el parser de la DGT clasifica igual que la
extracción original, que las variables no usan información del futuro (fuga de
datos), las métricas y las reglas de negocio de la app.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motoradar import datos as D  # noqa: E402
from motoradar import metricas as Me  # noqa: E402
from motoradar.features import construir_tabla, dias_laborables  # noqa: E402
from motoradar.negocio import Conductor, cuota_mercado, plan_pedido, segmentos_permitidos, siguiente_paso  # noqa: E402
from pipeline.fase0_descarga_dgt import campo, segmento  # noqa: E402


# ------------------------------------------------------------------ datos
def test_datos_completos():
    df = D.leer_provincia_segmento()
    assert len(D.meses()) == 140
    assert df.groupby(["provincia", "segmento"], observed=True).ngroups == 208
    assert (df["matriculaciones"] >= 0).all()


def test_totales_coherentes():
    prov = D.leer_provincia_segmento().groupby("mes")["matriculaciones"].sum()
    nac = D.leer_nacional_detalle()
    nac = nac[nac["estado"] == "Nueva"].groupby("mes")["matriculaciones"].sum()
    marcas = D.leer_marcas_mensual().groupby("mes")["matriculaciones"].sum()
    assert (prov == nac).all() and (marcas == nac).all()


def test_valores_conocidos():
    # Dos cifras comprobadas a mano contra la extracción de la DGT
    nac = D.leer_nacional_detalle()
    total = nac[nac["estado"] == "Nueva"]["matriculaciones"].sum()
    assert total == 2_305_568
    abril20 = nac[(nac["estado"] == "Nueva") & (nac["mes"] == pd.Period("2020-04", "M"))]["matriculaciones"].sum()
    assert abril20 < 1500  # confinamiento


# ------------------------------------------------------------------ parser DGT
def test_parser_dgt_reproduce_extraccion():
    muestra = json.loads((Path(__file__).parent / "fixtures" / "dgt_muestra.json").read_text(encoding="utf-8"))
    n = len(muestra) // 2
    for linea, esperado in zip(muestra[:n], muestra[n:]):
        s = segmento(linea)
        if s is None:
            obtenido = "-"
        else:
            obtenido = "|".join([s, "U" if linea[178] == "U" else "N", "E" if linea[93] == "2" else "C",
                                 "J" if linea[179] == "X" else "F", campo(linea, "provincia")])
        assert obtenido == esperado, (linea[:60], obtenido, esperado)


# ------------------------------------------------------------------ variables
def test_sin_fuga_de_datos():
    """Cambiar los datos posteriores al origen no puede cambiar las variables."""
    df = D.leer_provincia_segmento()
    origen = pd.Period("2025-06", "M")
    t1 = construir_tabla(df, origen_min=origen, origen_max=origen)
    df2 = df.copy()
    df2.loc[df2["mes"] > origen, "matriculaciones"] = 99999
    t2 = construir_tabla(df2, origen_min=origen, origen_max=origen)
    from motoradar.features import VARIABLES
    pd.testing.assert_frame_equal(t1[VARIABLES], t2[VARIABLES])
    assert not t1["y"].equals(t2["y"])  # el objetivo sí cambia


def test_dias_laborables():
    assert dias_laborables(pd.Period("2025-08", "M")) == 20  # 15 de agosto es festivo nacional
    assert dias_laborables(pd.Period("2026-02", "M")) == 20


# ------------------------------------------------------------------ métricas
def test_metricas():
    assert Me.wape([10, 10], [12, 8]) == 0.2
    assert Me.sesgo([10, 10], [8, 8]) == 0.2
    df = pd.DataFrame({"origen": [1, 1, 2, 2], "y": [10, 10, 10, 10], "a": [10, 10, 10, 10], "b": [5, 5, 5, 5]})
    p, lo, hi = Me.mejora_bootstrap(df, "a", "b", n=200)
    assert p == 1.0 and lo == 1.0 and hi == 1.0


# ------------------------------------------------------------------ negocio
def test_pedido():
    prev = pd.DataFrame({"segmento": ["A1", "A1", "A1", "A"], "h": [1, 2, 3, 1],
                         "prevision": [1000, 900, 800, 100.0], "inf_80": [800, 700, 600, 80.0],
                         "sup_80": [1200, 1100, 1000, 120.0]})
    cuotas = cuota_mercado({"A1": 120, "A": 0}, {"A1": 12000, "A": 1000})
    assert cuotas == {"A1": 0.01, "A": 0.0}
    plan = plan_pedido(prev, cuotas, {"A1": 5}, meses_cobertura=2).set_index("segmento")
    assert plan.loc["A1", "ventas_esperadas"] == 19.0
    assert plan.loc["A1", "pedido_sugerido"] == 14  # 19 - 5
    assert plan.loc["A", "pedido_sugerido"] == 0
    pru = plan_pedido(prev, cuotas, {"A1": 5}, 2, "Prudente").set_index("segmento")
    assert pru.loc["A1", "pedido_sugerido"] == 10  # 15 - 5
    try:
        plan_pedido(prev, cuotas, {}, 4)
        raise AssertionError("debía fallar")
    except ValueError:
        pass


def test_carnets():
    assert set(segmentos_permitidos(Conductor(30, {"B": 5}))) == {"AM", "A1"}
    assert set(segmentos_permitidos(Conductor(19, {"B": 1}))) == {"AM"}
    assert set(segmentos_permitidos(Conductor(25, {"A2": 1}))) == {"AM", "A1", "A2"}
    assert set(segmentos_permitidos(Conductor(40, {"A": 10}))) == {"AM", "A1", "A2", "A"}
    assert segmentos_permitidos(Conductor(14, {})) == {}
    assert "A2" in siguiente_paso(Conductor(19, {"B": 1}))
    assert "1 año" in siguiente_paso(Conductor(25, {"A2": 1}))


# ------------------------------------------------------------------ salidas
def test_previsiones_generadas():
    prev = pd.read_csv(D.GOLD / "previsiones.csv", keep_default_na=False)
    assert len(prev) == 208 * 3
    assert (prev["inf_80"] <= prev["prevision"]).all() and (prev["prevision"] <= prev["sup_80"]).all()
    assert np.isfinite(prev["prevision"]).all()


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        try:
            f()
            print(f"OK    {nombre}")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"FALLO {nombre}: {e!r}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas superadas")
    sys.exit(1 if fallos else 0)
