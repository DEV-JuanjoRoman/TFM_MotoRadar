"""Fase 1 - Preparación y control de calidad de los datos.

Lee los agregados de la DGT (``data/raw``), comprueba que son coherentes entre sí
y genera las tablas limpias de ``data/processed`` y la tabla de modelado de
``data/gold``.

    python pipeline/fase1_preparacion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from motoradar import datos as D  # noqa: E402
from motoradar.features import calendario, construir_tabla  # noqa: E402


def main():
    D.PROCESSED.mkdir(parents=True, exist_ok=True)
    D.GOLD.mkdir(parents=True, exist_ok=True)
    D.RESULTS.mkdir(parents=True, exist_ok=True)

    prov = D.leer_provincia_segmento()
    nac = D.leer_nacional_detalle()
    marcas = D.leer_marcas_mensual()
    provincias = D.leer_provincias()

    # ---------------- Controles de calidad ----------------
    controles = []
    m = D.meses()
    controles.append(("Meses consecutivos sin huecos",
                      bool(m.equals(pd.period_range(m[0], m[-1], freq="M")))))
    controles.append(("52 provincias x 4 segmentos", prov.groupby(["provincia", "segmento"],
                                                                 observed=True).ngroups == 208))
    controles.append(("Todas las provincias tienen nombre",
                      set(prov["provincia"]) == set(provincias["codigo_dgt"])))
    controles.append(("Sin valores negativos", bool((prov["matriculaciones"] >= 0).all())))

    total_prov = prov.groupby("mes")["matriculaciones"].sum()
    total_nac = nac[nac["estado"] == "Nueva"].groupby("mes")["matriculaciones"].sum()
    total_marca = marcas.groupby("mes")["matriculaciones"].sum()
    controles.append(("Suma provincial = total nacional (motos nuevas)",
                      bool((total_prov == total_nac).all())))
    controles.append(("Suma por marcas = total nacional (motos nuevas)",
                      bool((total_marca == total_nac).all())))
    covid = total_nac.loc[D.MESES_COVID]
    controles.append(("Abril 2020 es el mínimo de la serie (confinamiento)",
                      bool(total_nac.idxmin() == pd.Period("2020-04", "M")) and covid.min() < 2000))

    informe = pd.DataFrame(controles, columns=["control", "superado"])
    informe.to_csv(D.RESULTS / "controles_calidad.csv", index=False)
    print(informe.to_string(index=False))
    if not informe["superado"].all():
        raise SystemExit("Algún control de calidad ha fallado")

    # ---------------- Tablas limpias ----------------
    prov = prov.merge(provincias[["codigo_dgt", "provincia", "ccaa"]].rename(
        columns={"provincia": "nombre_provincia", "codigo_dgt": "provincia"}), on="provincia")
    prov.assign(mes=prov["mes"].astype(str)).to_csv(
        D.PROCESSED / "matriculaciones_provincia_segmento.csv", index=False)
    nac.assign(mes=nac["mes"].astype(str)).to_csv(
        D.PROCESSED / "matriculaciones_nacional_detalle.csv", index=False)
    marcas.assign(mes=marcas["mes"].astype(str)).to_csv(
        D.PROCESSED / "matriculaciones_marca.csv", index=False)
    cal = calendario(D.meses()).reset_index(names="mes")
    cal.assign(mes=cal["mes"].astype(str)).to_csv(D.PROCESSED / "calendario.csv", index=False)

    # ---------------- Capa gold: tabla de modelado ----------------
    tabla = construir_tabla(D.leer_provincia_segmento())
    tabla = tabla.assign(origen=tabla["origen"].astype(str),
                         mes_objetivo=tabla["mes_objetivo"].astype(str))
    tabla.to_parquet(D.GOLD / "gold_demanda_modelo.parquet", index=False)
    print(f"\nTabla de modelado: {len(tabla):,} filas, {tabla['serie'].nunique()} series, "
          f"orígenes {tabla['origen'].min()} a {tabla['origen'].max()}")
    print(f"Motos nuevas 2015-01 a 2026-08: {int(total_nac.sum()):,}")


if __name__ == "__main__":
    main()
