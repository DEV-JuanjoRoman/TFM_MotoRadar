"""Reglas de negocio de la app: planificación de pedidos y carnets.

Todo es determinista y sin dependencias para poder probarlo (tests/test_negocio.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ESCENARIOS = {"Prudente": "inf_80", "Central": "prevision", "Optimista": "sup_80"}


def cuota_mercado(ventas_propias_12m: dict[str, float], mercado_12m: dict[str, float]) -> dict[str, float]:
    """Cuota del concesionario en su provincia por segmento (0-1)."""
    cuotas = {}
    for seg, ventas in ventas_propias_12m.items():
        mercado = mercado_12m.get(seg, 0)
        cuotas[seg] = min(ventas / mercado, 1.0) if mercado > 0 else 0.0
    return cuotas


def plan_pedido(prev: pd.DataFrame, cuotas: dict[str, float], stock_actual: dict[str, int],
                meses_cobertura: int = 2, escenario: str = "Central") -> pd.DataFrame:
    """Unidades a pedir por segmento.

    ``prev``: previsiones de UNA provincia (columnas segmento, h, prevision,
    inf_80, sup_80). Las ventas esperadas del concesionario son la previsión del
    mercado multiplicada por su cuota. El pedido cubre los ``meses_cobertura``
    próximos meses descontando el stock que ya tiene. Se redondea hacia arriba
    porque una moto que falta es una venta perdida.
    """
    if not 1 <= meses_cobertura <= 3:
        raise ValueError("La cobertura debe estar entre 1 y 3 meses (horizonte del modelo)")
    col = ESCENARIOS[escenario]
    filas = []
    for seg, g in prev[prev["h"] <= meses_cobertura].groupby("segmento", observed=True):
        mercado = float(g[col].sum())
        cuota = cuotas.get(seg, 0.0)
        demanda = mercado * cuota
        stock = stock_actual.get(seg, 0)
        pedido = max(0, -(-round(demanda * 100) // 100) - stock)  # ceil sin errores de coma flotante
        filas.append({"segmento": seg, "mercado_previsto": mercado, "cuota": cuota,
                      "ventas_esperadas": demanda, "stock_actual": stock, "pedido_sugerido": int(pedido)})
    return pd.DataFrame(filas)


@dataclass
class Conductor:
    edad: int
    permisos: dict[str, int]  # permiso -> años de antigüedad, p. ej. {"B": 5, "A2": 1}


def segmentos_permitidos(c: Conductor) -> dict[str, str]:
    """Qué segmentos de moto puede conducir alguien y por qué (normativa española).

    * AM: ciclomotores. Lo incluyen todos los demás permisos.
    * A1 (125 cc, 11 kW): permiso A1, o B con más de 3 años de antigüedad.
    * A2 (35 kW): permiso A2.
    * A (sin límite): permiso A (se obtiene tras 2 años con A2).
    """
    p = c.permisos
    tiene = lambda x: x in p  # noqa: E731
    res = {}
    if any(tiene(x) for x in ("AM", "A1", "A2", "A", "B")):
        res["AM"] = "Incluido en cualquier permiso de conducción"
    if tiene("A1") or tiene("A2") or tiene("A"):
        res["A1"] = "Permiso de moto A1 o superior"
    elif tiene("B") and p["B"] >= 3:
        res["A1"] = "Permiso B con más de 3 años (solo en España)"
    if tiene("A2") or tiene("A"):
        res["A2"] = "Permiso A2 o A"
    if tiene("A"):
        res["A"] = "Permiso A"
    return res


def siguiente_paso(c: Conductor) -> str:
    """Recomendación de carnet para subir de segmento (argumento de venta)."""
    p = c.permisos
    if "A" in p:
        return "Puede conducir cualquier moto."
    if "A2" in p:
        faltan = max(0, 2 - p["A2"])
        return ("Ya puede sacar el A (2 años de A2 cumplidos)." if faltan == 0
                else f"Podrá sacar el A dentro de {faltan} año(s), al cumplir 2 años con el A2.")
    if c.edad >= 18:
        return "Puede sacar el A2 (desde 18 años) y acceder a motos de hasta 35 kW."
    if c.edad >= 16:
        return "Puede sacar el A1 (desde 16 años) para motos de 125 cc."
    if c.edad >= 15:
        return "Puede sacar el AM (desde 15 años) para ciclomotores."
    return "Aún no tiene edad para ningún permiso de moto."
