"""Construcción de la tabla de modelado (capa gold).

Planteamiento: modelo global y directo. En cada mes de origen ``t`` se conocen las
matriculaciones hasta ``t`` y se predice el mes ``t + h`` para ``h = 1, 2, 3``.
Todas las variables se calculan solo con información disponible en ``t`` (no hay
fuga de datos) y se normalizan por el nivel de la serie (media de los últimos 12
meses) para que un único modelo aprenda a la vez de Barcelona y de Soria.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from .datos import MESES_COVID, SEGMENTOS

HORIZONTES = (1, 2, 3)
LAGS = tuple(range(0, 12))  # y[t], y[t-1], ..., y[t-11]


@lru_cache(maxsize=None)
def _festivos_nacionales(anio: int) -> frozenset:
    return frozenset(holidays.Spain(years=anio).keys())


def dias_laborables(periodo: pd.Period) -> int:
    """Días de lunes a viernes del mes que no son festivo nacional."""
    dias = pd.date_range(periodo.start_time, periodo.end_time.normalize(), freq="D")
    fest = _festivos_nacionales(periodo.year)
    return int(sum(1 for d in dias if d.weekday() < 5 and d.date() not in fest))


def calendario(periodos: pd.PeriodIndex) -> pd.DataFrame:
    p = pd.PeriodIndex(periodos)
    return pd.DataFrame({
        "mes_num": p.month,
        "mes_sin": np.sin(2 * np.pi * p.month / 12),
        "mes_cos": np.cos(2 * np.pi * p.month / 12),
        "dias_laborables": [dias_laborables(x) for x in p],
        "es_covid": p.isin(MESES_COVID).astype(int),
    }, index=p)


def matriz_series(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa el formato largo a una matriz meses x (provincia, segmento)."""
    tabla = df.pivot_table(index="mes", columns=["provincia", "segmento"],
                           values="matriculaciones", aggfunc="sum", observed=True)
    return tabla.sort_index().fillna(0)


def construir_tabla(df: pd.DataFrame, origen_min: pd.Period | None = None,
                    origen_max: pd.Period | None = None, con_objetivo: bool = True) -> pd.DataFrame:
    """Tabla de modelado: una fila por serie, mes de origen y horizonte.

    ``con_objetivo=False`` genera también las filas cuyo mes objetivo aún no se
    conoce (las que se usan para la previsión final).
    """
    Y = matriz_series(df)
    meses = Y.index
    cal = calendario(pd.period_range(meses[0], meses[-1] + max(HORIZONTES), freq="M"))
    valores = Y.to_numpy(dtype=float)
    n_meses, n_series = valores.shape
    columnas = list(Y.columns)

    filas = []
    for i_origen in range(23, n_meses):  # hacen falta 24 meses de historia
        origen = meses[i_origen]
        if origen_min is not None and origen < origen_min:
            continue
        if origen_max is not None and origen > origen_max:
            continue
        hist = valores[: i_origen + 1]
        nivel = hist[-12:].mean(axis=0)
        nivel_prev = hist[-24:-12].mean(axis=0)
        base = np.log1p(nivel)
        for h in HORIZONTES:
            objetivo_mes = origen + h
            i_obj = i_origen + h
            if con_objetivo and i_obj >= n_meses:
                continue
            bloque = {
                "serie": np.arange(n_series),
                "origen": origen, "mes_objetivo": objetivo_mes, "h": h,
                "nivel_12m": nivel,
                "log_nivel": base,
                "tendencia_anual": np.log1p(nivel) - np.log1p(nivel_prev),
                "media_3m_rel": np.log1p(hist[-3:].mean(axis=0)) - base,
                "mismo_mes_anio_ant_rel": np.log1p(valores[i_obj - 12]) - base,
                "mismo_mes_hace2_rel": np.log1p(valores[i_obj - 24]) - base,
                "yoy_3m": np.log1p(hist[-3:].sum(axis=0)) - np.log1p(hist[-15:-12].sum(axis=0)),
            }
            for lag in LAGS:
                bloque[f"lag{lag}_rel"] = np.log1p(hist[-1 - lag]) - base
            c = cal.loc[objetivo_mes]
            for col in ["mes_num", "mes_sin", "mes_cos", "dias_laborables", "es_covid"]:
                bloque[col] = c[col]
            bloque["dias_lab_anio_ant"] = cal.loc[objetivo_mes - 12, "dias_laborables"]
            bloque["origen_covid"] = int(any(origen - k in MESES_COVID for k in range(0, 12)))
            if i_obj < n_meses:
                bloque["y"] = valores[i_obj]
            else:
                bloque["y"] = np.nan
            filas.append(pd.DataFrame(bloque))
    tabla = pd.concat(filas, ignore_index=True)
    tabla["provincia"] = [columnas[s][0] for s in tabla["serie"]]
    tabla["segmento"] = pd.Categorical([columnas[s][1] for s in tabla["serie"]],
                                       categories=SEGMENTOS, ordered=True)
    tabla["segmento_cod"] = tabla["segmento"].cat.codes
    # Código de provincia estable (orden alfabético del código DGT)
    codigos = {c: i for i, c in enumerate(sorted({c[0] for c in columnas}))}
    tabla["provincia_cod"] = tabla["provincia"].map(codigos).astype(int)
    tabla["objetivo_rel"] = np.log1p(tabla["y"]) - tabla["log_nivel"]
    return tabla


VARIABLES = (
    ["h", "segmento_cod", "provincia_cod", "log_nivel", "tendencia_anual", "media_3m_rel",
     "mismo_mes_anio_ant_rel", "mismo_mes_hace2_rel", "yoy_3m"]
    + [f"lag{lag}_rel" for lag in LAGS]
    + ["mes_num", "mes_sin", "mes_cos", "dias_laborables", "dias_lab_anio_ant", "origen_covid"]
)
