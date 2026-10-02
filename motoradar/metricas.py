"""Métricas de evaluación.

La métrica principal es el WAPE (error absoluto ponderado): suma de errores
absolutos dividida entre la suma de lo real. Se interpreta directamente como
"de cada 100 motos que se matriculan, el modelo se equivoca en X", no explota con
series pequeñas (como el MAPE) y pondera cada provincia por su peso en el mercado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(real, pred) -> float:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    total = np.abs(real).sum()
    return float(np.abs(real - pred).sum() / total) if total else float("nan")


def mae(real, pred) -> float:
    return float(np.mean(np.abs(np.asarray(real, float) - np.asarray(pred, float))))


def sesgo(real, pred) -> float:
    """Positivo si el modelo se queda corto (infra-previsión)."""
    real = np.asarray(real, dtype=float)
    return float((real - np.asarray(pred, float)).sum() / real.sum())


def mejora_bootstrap(df: pd.DataFrame, col_modelo: str, col_ref: str, grupo: str = "origen",
                     n: int = 2000, semilla: int = 7) -> tuple[float, float, float]:
    """Mejora relativa del WAPE de ``col_modelo`` frente a ``col_ref``.

    Bootstrap por bloques: se remuestrean meses de origen completos, que es la
    unidad independiente de la validación (dentro de un mismo mes los errores de
    las provincias están correlacionados).
    """
    rng = np.random.default_rng(semilla)
    bloques = [g for _, g in df.groupby(grupo)]
    err_m = np.array([np.abs(g["y"] - g[col_modelo]).sum() for g in bloques])
    err_r = np.array([np.abs(g["y"] - g[col_ref]).sum() for g in bloques])
    punto = 1 - err_m.sum() / err_r.sum()
    idx = rng.integers(0, len(bloques), size=(n, len(bloques)))
    mejoras = 1 - err_m[idx].sum(axis=1) / err_r[idx].sum(axis=1)
    lo, hi = np.percentile(mejoras, [2.5, 97.5])
    return float(punto), float(lo), float(hi)
