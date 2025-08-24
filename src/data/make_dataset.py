# src/data/make_dataset.py
"""
Crea el dataset limpio y unificado a partir de data/raw:
- Ignora PDFs (solo procesa TXT/CSV/JSON/JSONL) para Entrega 1.
- Infere metadatos desde la ruta: source_dataset, source_bucket, split, label.
- Deduplica por hash de (texto_original || resumen).
- Escribe JSONL en streaming y luego deriva CSV (columnas estandarizadas).
"""

from __future__ import annotations
import csv, json, hashlib, unicodedata, sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# Procesamos SOLO formatos textuales directos para esta entrega
ALLOWED_SUFFIXES = {".txt", ".csv", ".json", ".jsonl"}
SKIP_SUFFIXES = {".pdf"}   # explícito

# Heurísticas de columnas en CSV/JSON
TEXT_CANDIDATES = {"texto","text","source","article","document","original","input","content","body"}
PLS_CANDIDATES  = {"resumen","summary","pls","simple","plain_language","simplified"}

# Filtros simples de calidad (ajusta si hace falta)
MIN_LEN_TEXT = 30
MAX_LEN_TEXT = 20000

def norm(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()

def valid_text(s: str) -> bool:
    n = len(s)
    return (n >= MIN_LEN_TEXT) and (n <= MAX_LEN_TEXT)

def pick_columns(header: List[str]) -> Tuple[str, str]:
    lower = [h.lower().strip() for h in header]
    text_col = ""
    pls_col  = ""
    for c in lower:
        if not text_col and c in TEXT_CANDIDATES:
            text_col = c
        if not pls_col and c in PLS_CANDIDATES:
            pls_col = c
    # fallback por posición
    if not text_col and lower:
        text_col = lower[0]
    if not pls_col and len(lower) > 1:
        pls_col = lower[1]
    return text_col, pls_col

def infer_meta_from_path(fp: Path):
    """
    Infere:
      - source_dataset: top-level (cochrane | pfizer | trialsummaries | clinicaltrials | otro)
      - source_bucket: subcarpeta significativa (p.ej. train/pls, test/non_pls, original_texts, etc.)
      - split: train | test | unsplit
      - label: pls | non_pls | "" (si no aplica)
    """
    parts = [p.lower() for p in fp.parts]
    # dataset top-level: busca el primer directorio debajo de data/raw
    source_dataset = ""
    try:
        raw_idx = parts.index("raw")
        if raw_idx + 1 < len(parts):
            source_dataset = parts[raw_idx + 1]
    except ValueError:
        pass

    # bucket/subruta informativa
    # ej: raw/cochrane/train/pls/...  -> "train/pls"
    #     raw/trialsummaries/original_texts/... -> "original_texts"
    source_bucket = ""
    if source_dataset:
        start = parts.index(source_dataset) + 1
        if start < len(parts) - 1:  # hay algo entre dataset y archivo
            source_bucket = "/".join(parts[start:-1])

    split = "unsplit"
    if "train" in parts: split = "train"
    if "test"  in parts: split = "test"

    label = ""
    if "pls" in parts:
        label = "pls"
    if ("non_pls" in parts) or ("non-pls" in parts):
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

def iter_txt(fp: Path) -> Iterator[Dict[str, str]]:
    source_dataset, source_bucket, split, label = infer_meta_from_path(fp)
    for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if not line.strip():
            continue
        if "|||" in line:
            src, pls = line.split("|||", 1)
            yield {
                "texto_original": norm(src),
                "resumen": norm(pls),
                "source": fp.parent.name,
                "doc_id": f"{fp.name}#L{i+1}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
            }
        else:
            txt = norm(line)
            yield {
                "texto_original": txt,
                "resumen": "",
                "source": fp.parent.name,
                "doc_id": f"{fp.name}#L{i+1}",
                "split": split,
                "label": label,
                "source_dataset": source_dataset,
                "source_bucket": source_bucket,
            }

def iter_csv(fp: Path) -> Iterator[Dict[str, str]]:
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
        pls   = norm(row.get(pcol, ""))
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
        }

def iter_jsonl(fp: Path) -> Iterator[Dict[str, str]]:
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
            pls   = norm(obj.get(pcol, ""))
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
            }

def iter_json(fp: Path) -> Iterator[Dict[str, str]]:
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
            pls   = norm(obj.get(pcol, ""))
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
            }

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

def quality_ok(r: Dict[str, str]) -> bool:
    txt = r.get("texto_original", "")
    if not valid_text(txt):
        return False
    return True

def save_streaming_jsonl(stem="dataset_clean_v1"):
    """
    Escritura en streaming a JSONL con deduplicación.
    Luego, una pasada para derivar CSV.
    """
    out_jsonl = PROC / f"{stem}.jsonl"
    seen = set()
    kept = 0
    total = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in parse_files():
            total += 1
            if not quality_ok(r):
                continue
            key = (r["texto_original"] + "||" + r.get("resumen", "")).encode("utf-8")
            h = hashlib.sha256(key).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] wrote {out_jsonl}  total_seen={total}  kept={kept}  deduped={total-kept}")
    return out_jsonl

def jsonl_to_csv(jsonl_path: Path, stem="dataset_clean_v1"):
    import pandas as pd
    df = pd.read_json(jsonl_path, lines=True)

    # Orden sugerido de columnas (incluye metadatos nuevos)
    cols = [c for c in [
        "texto_original","resumen","source","doc_id","split","label",
        "source_dataset","source_bucket"
    ] if c in df.columns]

    out_csv = PROC / f"{stem}.csv"
    df.to_csv(out_csv, index=False, columns=cols)
    print(f"[OK] wrote {out_csv} (rows={len(df)})")

def main():
    jl = save_streaming_jsonl("dataset_clean_v1")
    jsonl_to_csv(jl, "dataset_clean_v1")

if __name__ == "__main__":
    main()
