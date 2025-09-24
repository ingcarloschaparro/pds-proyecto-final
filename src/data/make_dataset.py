# src/data/make_dataset.py
"""Crea el dataset limpio si unificado a partir de data/raw: - Procesa TXT/CSV/JSON/JSONL si omite PDFs (Entrega 1). - Infere metadatos desde la ruta: source_dataset, source_bucket, split, label. - Maneja TXT con: * pares"TEXTO ||| PLS"-> has_pair = True * por favor sueltos (carpetas"por favor/") -> resumen = texto, texto_original =""* non-por favor sueltos (carpetas"non_pls/") -> texto_original = texto, resumen =""- Amplía columnas candidatas para CSV/JSON (poner.ej., plain_language_summary). - Filtra calidad si existe longitud mínima en texto_original **o** resumen. - Deduplica por hash de (texto_original || resumen). - Escribe JSONL en streaming si luego deriva CSV."""

from __future__ import annotations
import csv
import json
import sys
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# --- Configuración ---
# Formatos textuales admitidos (Entrega 1):
ALLOWED_SUFFIXES = {".txt", ".csv", ".json", ".jsonl"}
SKIP_SUFFIXES = {".pdf"}  # explícito

# Heurísticas de columnas en CSV/JSON
TEXT_CANDIDATES = {
    "texto",
    "text",
    "source",
    "article",
    "document",
    "original",
    "input",
    "content",
    "body",
    "source_text",
    "original_text",
    "full_text",
    "document_text",
}
PLS_CANDIDATES = {
    "resumen",
    "summary",
    "por favor",
    "simple",
    "plain_language",
    "simplified",
    "plain_language_summary",
    "plainlanguage",
    "lay_summary",
    "plain_summary",
    "pls_text",
}

# Filtros simples de calidad - Ajustados para ser menos estrictos
MIN_LEN_TEXT = 10  # Reducido de 30 a 10 caracteres
MAX_LEN_TEXT = 50000  # Aumentado de 20000 a 50000 caracteres


# ----------------- Utilidades -----------------
def norm(s: str) -> str:
    """Normaliza unicode, colapsa espacios si homogeneiza saltos de línea."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Solo colapsar espacios múltiples, NO eliminar todos los espacios
    s = " ".join(s.split())
    return s.strip()


def pick_columns(header: List[str]) -> Tuple[str, str]:
    """Intenta adivinar columnas (texto, resumen) en CSV/JSON por nombres conocidos."""
    lower = [h.lower().strip() for h in header]
    text_col = ""
    pls_col = ""
    for c in lower:
        if not text_col and c in TEXT_CANDIDATES:
            text_col = c
        if not pls_col and c in PLS_CANDIDATES:
            pls_col = c
    # fallback por posición si no encontró nombres "amigables"
    if not text_col and lower:
        text_col = lower[0]
    if not pls_col and len(lower) > 1:
        pls_col = lower[1]
    return text_col, pls_col


def infer_meta_from_path(fp: Path):
    """Infere metadatos desde la ruta del archivo: - source_dataset: cochrane | pfizer | trialsummaries | clinicaltrials | otro - source_bucket: subcarpeta informativa (ej., train/por favor, test/non_pls, original_texts, ...) - split: train | test | unsplit - label: por favor | non_pls |""(si no aplica)"""
    parts = [p.lower() for p in fp.parts]

    # dataset top-level: buscar el primer directorio debajo de data/raw
    source_dataset = ""
    try:
        raw_idx = parts.index("raw")
        if raw_idx + 1 < len(parts):
            source_dataset = parts[raw_idx + 1]
    except ValueError:
        pass

    # bucket/subruta informativa (entre el dataset y el archivo)
    source_bucket = ""
    if source_dataset:
        start = parts.index(source_dataset) + 1
        if start < len(parts) - 1:  # hay algo entre dataset y archivo
            source_bucket = "/".join(parts[start:-1])

    # split y label por palabras clave en la ruta
    split = "unsplit"
    if "train" in parts:
        split = "train"
    if "test" in parts:
        split = "test"

    label = ""
    if "por favor" in parts:
        label = "por favor"
    if ("non_pls" in parts) or ("non-por favor" in parts):
        label = "non_pls"

    # normaliza algunos nombres conocidos
    mapping = {
        "cochrane": "cochrane",
        "pfizer": "pfizer",
        "trial summaries": "trialsummaries",
        "trialsummaries": "trialsummaries",
        "clinicaltrials.gov": "clinicaltrials",
        "clinicaltrials": "clinicaltrials",
    }
    source_dataset = mapping.get(source_dataset, source_dataset)

    return source_dataset, source_bucket, split, label


# ----------------- Parsers por tipo -----------------
def iter_txt(fp: Path) -> Iterator[Dict[str, str]]:
    """Lee .txt línea por línea. Reglas: - Si hay"TEXTO ||| por favor"-> construye par (has_pair=True). - Si NO hay"|||"si la ruta sugiere carpeta por favor -> resumen = texto (has_pair=False). - Si NO hay"|||"si sugiere NON_PLS -> texto_original = texto (has_pair=False). - Otros casos -> texto_original = texto (has_pair=False)."""
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    for i, line in enumerate(
        fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    ):
        if not line.strip():
            continue
        if "|||" in line:
            src, pls = line.split("|||", 1)
            yield {
                "texto_original": norm(src),
                "resumen": norm(pls),
                "source": fp.parent.name,
                "doc_id": f"{fp.name}#como{i+1}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
                "has_pair": True,
            }
        else:
            txt = norm(line)
            if label == "por favor":
                # PLS “suelto”
                yield {
                    "texto_original": "",
                    "resumen": txt,
                    "source": fp.parent.name,
                    "doc_id": f"{fp.name}#como{i+1}",
                    "split": split,
                    "label": label,
                    "source_dataset": source_dataset,
                    "source_bucket": source_bucket,
                    "has_pair": False,
                }
            else:
                # Non-PLS u otros casos
                yield {
                    "texto_original": txt,
                    "resumen": "",
                    "source": fp.parent.name,
                    "doc_id": f"{fp.name}#como{i+1}",
                    "split": split,
                    "label": label,
                    "source_dataset": source_dataset,
                    "source_bucket": source_bucket,
                    "has_pair": False,
                }


def iter_csv(fp: Path) -> Iterator[Dict[str, str]]:
    """Lee .csv intentando detectar columnas de texto/resumen por heurística."""
    import pandas as pd

    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(fp, encoding=enc)
            break
        except Exception:
            if enc == "latin1":
                raise
    if df.empty:
        return
    tcol, pcol = pick_columns(df.columns.tolist())
    for i, row in df.iterrows():
        texto = norm(row.get(tcol, ""))
        pls = norm(row.get(pcol, ""))
        if not texto and not pls:
            continue
        yield {
            "texto_original": texto,
            "resumen": pls,
            "source": fp.parent.name,
            "doc_id": f"{fp.name}#{i}",
            "split": split,
            "label": label,
            "source_dataset": source_dataset,
            "source_bucket": source_bucket,
            "has_pair": bool(texto and pls),
        }


def iter_jsonl(fp: Path) -> Iterator[Dict[str, str]]:
    """Lee .jsonl (un objeto por línea)."""
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            keys = list(obj.keys())
            tcol, pcol = pick_columns(keys)
            texto = norm(obj.get(tcol, ""))
            pls = norm(obj.get(pcol, ""))
            if not texto and not pls:
                continue
            yield {
                "texto_original": texto,
                "resumen": pls,
                "source": fp.parent.name,
                "doc_id": f"{fp.name}#{i}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
                "has_pair": bool(texto and pls),
            }


def iter_json(fp: Path) -> Iterator[Dict[str, str]]:
    """Lee .json como lista de objetos o un solo objeto."""
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    try:
        data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    if data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        tcol, pcol = pick_columns(keys)
        for i, obj in enumerate(data):
            texto = norm(obj.get(tcol, ""))
            pls = norm(obj.get(pcol, ""))
            if not texto and not pls:
                continue
            yield {
                "texto_original": texto,
                "resumen": pls,
                "source": fp.parent.name,
                "doc_id": f"{fp.name}#{i}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
                "has_pair": bool(texto and pls),
            }


# ----------------- Iterador maestro -----------------
def parse_files() -> Iterator[Dict[str, str]]:
    for fp in RAW.rglob("*"):
        if not fp.is_file():
            continue
        suf = fp.suffix.lower()
        if suf in SKIP_SUFFIXES or suf not in ALLOWED_SUFFIXES:
            continue  # ignora PDFs y otros binarios
        try:
            if suf == ".txt":
                yield from iter_txt(fp)
            elif suf == ".csv":
                yield from iter_csv(fp)
            elif suf == ".jsonl":
                yield from iter_jsonl(fp)
            elif suf == ".json":
                yield from iter_json(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}", file=sys.stderr)


# ----------------- Filtros y escritura -----------------
def quality_ok(r: Dict[str, str]) -> bool:
    """Acepta filas con texto o resumen (longitud mínima en cualquiera)."""
    txt = (r.get("texto_original", "") or "").strip()
    pls = (r.get("resumen", "") or "").strip()
    
    # Verificar longitud de texto original
    if txt:
        txt_ok = MIN_LEN_TEXT <= len(txt) <= MAX_LEN_TEXT
    else:
        txt_ok = False
    
    # Verificar longitud de resumen
    if pls:
        pls_ok = MIN_LEN_TEXT <= len(pls) <= MAX_LEN_TEXT
    else:
        pls_ok = False
    
    # Aceptar si tiene texto válido O resumen válido
    return txt_ok or pls_ok


def save_streaming_jsonl(stem: str = "dataset_clean_v1") -> Path:
    """Escribe JSONL en streaming con deduplicación (hash de texto||resumen). Devuelve la ruta al JSONL resultante."""
    out_jsonl = PROC / f"{stem}.jsonl"
    seen = set()
    kept = 0
    total = 0
    filtered_out = 0
    duplicates = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in parse_files():
            total += 1
            if not quality_ok(r):
                filtered_out += 1
                if total % 10000 == 0:  # Log cada 10k registros
                    print(f"[DEBUG] Procesados: {total}, Filtrados: {filtered_out}, Mantenidos: {kept}")
                continue
            key = (r.get("texto_original", "") + "||" + r.get("resumen", "")).encode(
                "utf-8"
            )
            h = hashlib.sha256(key).hexdigest()
            if h in seen:
                duplicates += 1
                continue
            seen.add(h)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"[RESULTADO] {out_jsonl}\n"
        f"  Total procesados: {total}\n"
        f"  Filtrados por calidad: {filtered_out}\n"
        f"  Duplicados: {duplicates}\n"
        f"  Mantenidos: {kept}\n"
        f"  Tasa de retención: {kept/total*100:.1f}%"
    )
    return out_jsonl


def jsonl_to_csv(jsonl_path: Path, stem: str = "dataset_clean_v1") -> Path:
    """Convierte el JSONL a CSV con columnas estandarizadas (usando pandas)."""
    import pandas as pd

    df = pd.read_json(jsonl_path, lines=True)

    cols = [
        c
        for c in [
            "texto_original",
            "resumen",
            "source",
            "doc_id",
            "split",
            "label",
            "source_dataset",
            "source_bucket",
            "has_pair",
        ]
        if c in df.columns
    ]

    out_csv = PROC / f"{stem}.csv"
    df.to_csv(out_csv, index=False, columns=cols)
    print(f"[bien] wrote {out_csv} (rows={len(df)})")
    return out_csv


# ----------------- Main -----------------
def main():
    jl = save_streaming_jsonl("dataset_clean_v1")
    jsonl_to_csv(jl, "dataset_clean_v1")


if __name__ == "__main__":
    main()
