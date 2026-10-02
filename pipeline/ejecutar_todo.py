"""Ejecuta el pipeline completo de principio a fin (fases 1 a 5) y las pruebas.

    python pipeline/ejecutar_todo.py            # ~8 min (incluye Holt-Winters)
    python pipeline/ejecutar_todo.py --rapido   # ~2 min (sin Holt-Winters)

La fase 0 (descarga de la DGT) no se lanza por defecto porque los agregados ya
están en data/raw; añade --descargar para regenerarlos desde la fuente.
"""

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def correr(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=RAIZ, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--descargar", action="store_true")
    a = ap.parse_args()
    if a.descargar:
        correr("pipeline/fase0_descarga_dgt.py")
    correr("pipeline/fase1_preparacion.py")
    correr("pipeline/fase2_eda.py")
    correr("pipeline/fase3_backtest.py", *(["--rapido"] if a.rapido else []))
    correr("pipeline/fase4_analisis_errores.py")
    correr("pipeline/fase5_modelo_final.py")
    correr("tests/test_motoradar.py")
