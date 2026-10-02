"""Modelos comparados en el TFM.

* ``naive_estacional``: el mismo mes del año anterior. Es lo que hace hoy un
  concesionario que planifica "como el año pasado".
* ``naive_tendencia``: el mismo mes del año anterior corregido por cómo va el
  año (últimos 3 meses frente a los mismos 3 meses del año anterior). Es el
  baseline exigente: el modelo tiene que ganarle a él.
* ``ets``: Holt-Winters (error aditivo sobre log1p) ajustado serie a serie.
* ``lightgbm``: modelo global con las variables de ``features.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import VARIABLES

PARAMS_LGBM = dict(
    objective="regression",  # L2 sobre log1p relativo (L1 y Huber dan lo mismo, ver memoria)
    learning_rate=0.03,
    n_estimators=900,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def naive_estacional(tabla: pd.DataFrame) -> np.ndarray:
    return np.expm1(tabla["mismo_mes_anio_ant_rel"] + tabla["log_nivel"]).to_numpy()


def naive_tendencia(tabla: pd.DataFrame) -> np.ndarray:
    base = naive_estacional(tabla)
    factor = np.exp(tabla["yoy_3m"].clip(-1.0, 1.0)).to_numpy()
    return base * factor


def entrenar_lgbm(tabla_train: pd.DataFrame, params: dict | None = None):
    import lightgbm as lgb

    datos = tabla_train[tabla_train["es_covid"] == 0].dropna(subset=["objetivo_rel"])
    modelo = lgb.LGBMRegressor(**(params or PARAMS_LGBM))
    modelo.fit(datos[VARIABLES], datos["objetivo_rel"],
               categorical_feature=["segmento_cod", "provincia_cod"])
    return modelo


def predecir_lgbm(modelo, tabla: pd.DataFrame) -> np.ndarray:
    rel = modelo.predict(tabla[VARIABLES])
    return np.clip(np.expm1(rel + tabla["log_nivel"].to_numpy()), 0, None)


def prever_ets(historia: np.ndarray, horizonte: int = 3) -> np.ndarray:
    """Holt-Winters aditivo amortiguado sobre log1p de una serie mensual."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.log1p(np.asarray(historia, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ajuste = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                          seasonal="add", seasonal_periods=12,
                                          initialization_method="estimated").fit()
            pred = ajuste.forecast(horizonte)
        except Exception:  # series casi vacías: se cae al naive estacional
            pred = y[-12:][:horizonte]
    return np.clip(np.expm1(pred), 0, None)
