"""Fase 0 - Descarga de los microdatos de la DGT y construcción de ``data/raw``.

Fuente: DGT, "Microdatos de matriculaciones de vehículos. Fichero mensual"
https://www.dgt.es/menusecundario/dgt-en-cifras/matraba-listados/matriculaciones-automoviles-mensual.html

Cada mes es un ZIP (~12 MB) con un fichero de texto de ancho fijo y una línea por
vehículo matriculado (todos los tipos: turismos, camiones, motos...). Este script:

1. descarga los meses indicados (se guardan en ``data/dgt_zip/``, fuera de Git);
2. se queda con motocicletas (tipos 50, 51, 52) y ciclomotores de dos ruedas (90);
3. clasifica cada moto por el carnet que exige (AM, A1, A2, A) con cilindrada y kW;
4. agrega y escribe los ficheros compactos de ``data/raw`` que usa el resto del
   pipeline.

No se guarda ningún dato a nivel de vehículo: el bastidor viene enmascarado y las
localidades solo se usan agregadas por provincia.

    python pipeline/fase0_descarga_dgt.py --desde 2015-01 --hasta 2026-08

Los ficheros de ``data/raw`` que hay en el repositorio se generaron con esta lógica.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from motoradar import datos as D  # noqa: E402

URL = ("https://www.dgt.es/microdatos/salida/{anio}/{mes}/vehiculos/matriculaciones/"
       "export_mensual_mat_{anio}{mes:02d}.zip")
CACHE = D.DATA / "dgt_zip"
TIPOS_MOTO = {"50", "51", "52", "90"}

# Posiciones (base 0) del diseño de registro MATRABA de la DGT
CAMPOS = {
    "fec_matricula": (0, 8), "marca": (17, 47), "modelo": (47, 69), "cod_tipo": (91, 93),
    "cod_propulsion": (93, 94), "cilindrada": (94, 99), "provincia": (152, 154),
    "nuevo_usado": (178, 179), "persona": (179, 180), "kw": (227, 234),
}

NORMALIZAR_MARCA = {
    "S.Y.M": "SYM", "S.Y.M.": "SYM", "HARLEY-DAVIDSON": "HARLEY DAVIDSON", "GAS-GAS": "GASGAS",
    "GAS GAS": "GASGAS", "INDIAN MOTORCYCLE": "INDIAN", "KSR MOTO AUSTRIA": "KSR MOTO",
    "MOTRON MOTORRAD": "MOTRON", "FANTIC MOTOR": "FANTIC",
}


def campo(linea: str, nombre: str) -> str:
    a, b = CAMPOS[nombre]
    return linea[a:b].strip()


def segmento(linea: str) -> str | None:
    """Carnet necesario: AM ciclomotor; A1 <=125 cc y <=11 kW; A2 <=35 kW; A resto."""
    tipo = campo(linea, "cod_tipo")
    if tipo == "90":
        return "AM"
    if tipo not in {"50", "51", "52"}:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return 0.0

    cc, kw = num(campo(linea, "cilindrada")), num(campo(linea, "kw"))
    if cc <= 125 and kw <= 11:
        return "A1"
    if kw <= 35:
        return "A2"
    return "A"


def descargar(periodo: pd.Period) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    destino = CACHE / f"export_mensual_mat_{periodo.year}{periodo.month:02d}.zip"
    if not destino.exists():
        url = URL.format(anio=periodo.year, mes=periodo.month)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (TFM MotoRadar)"})
        with urlopen(req, timeout=120) as r:
            destino.write_bytes(r.read())
        time.sleep(1)
    return destino.read_bytes()


def procesar_lineas(lineas) -> dict:
    """Agrega las líneas de un mes. Devuelve contadores por clave."""
    out = {"prov": Counter(), "nac": Counter(), "marca": Counter(), "marca_seg": Counter(), "modelo": defaultdict(lambda: [0, 0.0, 0.0])}
    for linea in lineas:
        if len(linea) < 240:
            continue
        seg = segmento(linea)
        if seg is None:
            continue
        nu = "U" if linea[178] == "U" else "N"
        prop = "E" if linea[93] == "2" else "C"
        pj = "J" if linea[179] == "X" else "F"
        out["nac"][f"{seg}|{nu}|{prop}|{pj}"] += 1
        if nu == "U":
            continue
        out["prov"][f"{campo(linea, 'provincia') or '??'}|{seg}"] += 1
        marca = " ".join(campo(linea, "marca").upper().split())
        marca = NORMALIZAR_MARCA.get(marca, marca)
        out["marca"][marca] += 1
        out["marca_seg"][(marca, seg)] += 1
        m = out["modelo"][(marca, campo(linea, "modelo"), seg)]
        m[0] += 1
        try:
            m[1] += float(campo(linea, "kw") or 0)
            m[2] += float(campo(linea, "cilindrada") or 0)
        except ValueError:
            pass
    return out


def leer_mes(periodo: pd.Period) -> dict:
    with zipfile.ZipFile(io.BytesIO(descargar(periodo))) as z:
        nombre = z.namelist()[0]
        texto = z.read(nombre).decode("latin-1")
    return procesar_lineas(texto.splitlines()[1:])  # la 1.ª línea es una cabecera


def escribir_series(fichero: Path, series: dict[str, list[int]]):
    lineas = [f"{k}:" + ",".join(str(v) if v else "" for v in vals) for k, vals in sorted(series.items())]
    fichero.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main(desde: str, hasta: str, top_marcas: int = 40):
    periodos = pd.period_range(desde, hasta, freq="M")
    meses = []
    for p in periodos:
        print(f"Procesando {p}...", flush=True)
        meses.append(leer_mes(p))
    n = len(periodos)
    D.RAW.mkdir(parents=True, exist_ok=True)
    (D.RAW / "months.txt").write_text(",".join(f"{p.year}{p.month:02d}" for p in periodos))

    def serie(clave, campo_):
        return [m[campo_].get(clave, 0) for m in meses]

    claves_prov = sorted({k for m in meses for k in m["prov"]})
    escribir_series(D.RAW / "dgt_provincia_segmento.txt", {k: serie(k, "prov") for k in claves_prov})
    claves_nac = sorted({k for m in meses for k in m["nac"]})
    escribir_series(D.RAW / "dgt_nacional_segmento.txt", {k: serie(k, "nac") for k in claves_nac})

    total = Counter()
    ult12 = Counter()
    for i, m in enumerate(meses):
        total.update(m["marca"])
        if i >= n - 12:
            ult12.update(m["marca"])
    top = {b for b, _ in total.most_common(top_marcas)} | {b for b, _ in ult12.most_common(top_marcas)}
    marcas = defaultdict(lambda: [0] * n)
    for i, m in enumerate(meses):
        for b, v in m["marca"].items():
            marcas[b if b in top else "OTRAS"][i] += v
    escribir_series(D.RAW / "dgt_marca_mensual.txt", dict(marcas))

    seg12 = defaultdict(lambda: dict.fromkeys(D.SEGMENTOS, 0))
    modelos = defaultdict(lambda: [0, 0.0, 0.0])
    for m in meses[-12:]:
        for (b, s), v in m["marca_seg"].items():
            seg12[b if b in top else "OTRAS"][s] += v
        for k, (c, kw, cc) in m["modelo"].items():
            x = modelos[k]
            x[0] += c
            x[1] += kw
            x[2] += cc
    (D.RAW / "dgt_marca_segmento_12m.txt").write_text(
        "".join(f"{b}:{','.join(str(v[s]) for s in D.SEGMENTOS)}\n" for b, v in sorted(seg12.items())), encoding="utf-8")
    mejores = sorted(modelos.items(), key=lambda kv: -kv[1][0])[:100]
    (D.RAW / "dgt_top_modelos_12m.txt").write_text("\n".join(
        f"{b}|{mo}|{s}|{c}|{kw / c:.1f}|{round(cc / c)}" for (b, mo, s), (c, kw, cc) in mejores), encoding="utf-8")
    print("Ficheros escritos en", D.RAW)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2015-01")
    ap.add_argument("--hasta", default="2026-08")
    a = ap.parse_args()
    main(a.desde, a.hasta)
