"""
Dataset creation and preprocessing functionality.
"""
import csv, json, hashlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Any
RAW = Path("data/raw")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)
# Columnas posibles para texto y resumen en CSV/JSON
TEXT_CANDIDATES = {"texto","text","source","article","document","original","input","content","body"}
PLS_CANDIDATES  = {"resumen","summary","pls","simple","plain_language","simplified"}
def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n","\n").replace("\r","\n")
    s = " ".join(s.split())
    return s.strip()
def pick_columns(header: List[str]) -> Tuple[str, str]:
    """Intenta adivinar columnas (texto, resumen) en CSV/JSON por nombres conocidos."""
    lower = [h.lower().strip() for h in header]
    text_col = ""
    pls_col  = ""
    for c in lower:
        if not text_col and c in TEXT_CANDIDATES:
            text_col = c
        if not pls_col and c in PLS_CANDIDATES:
            pls_col = c
    # si no encuentra explícitas, usa heurísticas por prefijos
    if not text_col and lower:
        text_col = lower[0]
    if not pls_col and len(lower) > 1:
        pls_col = lower[1]
    return text_col, pls_col
def iter_txt(fp: Path) -> Iterator[Dict[str, str]]:
    """Lee .txt línea por línea. Soporta pares 'TEXTO|||PLS'; si no hay PLS, deja vacío."""
    for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "|||" in line:
            src, pls = line.split("|||", 1)
            yield {"texto_original": norm(src), "resumen": norm(pls),
                   "source": fp.parent.name, "doc_id": f"{fp.name}#L{i+1}", "split":"unsplit"}
        else:
            txt = norm(line)
            if txt:
                yield {"texto_original": txt, "resumen": "",
                       "source": fp.parent.name, "doc_id": f"{fp.name}#L{i+1}", "split":"unsplit"}
def iter_csv(fp: Path) -> Iterator[Dict[str, str]]:
    import pandas as pd
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
        yield {"texto_original": texto, "resumen": pls,
               "source": fp.parent.name, "doc_id": f"{fp.name}#{i}", "split":"unsplit"}
def iter_jsonl(fp: Path) -> Iterator[Dict[str, str]]:
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
            yield {"texto_original": texto, "resumen": pls,
                   "source": fp.parent.name, "doc_id": f"{fp.name}#{i}", "split":"unsplit"}
def iter_json(fp: Path) -> Iterator[Dict[str, str]]:
    try:
        data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    # si es lista de objetos
    if data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        tcol, pcol = pick_columns(keys)
        for i, obj in enumerate(data):
            texto = norm(obj.get(tcol, ""))
            pls   = norm(obj.get(pcol, ""))
            if not texto and not pls:
                continue
            yield {"texto_original": texto, "resumen": pls,
                   "source": fp.parent.name, "doc_id": f"{fp.name}#{i}", "split":"unsplit"}
def parse_files() -> Iterator[Dict[str, str]]:
    for fp in RAW.rglob("*"):
        if not fp.is_file():
            continue
        suf = fp.suffix.lower()
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
            print(f"[WARN] Skipping {fp}: {e}")
def dedup(rows: Iterable[Dict[str, str]]) -> Iterator[Dict[str, str]]:
    seen = set()
    for r in rows:
        key = r["texto_original"] + "||" + r.get("resumen","")
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            yield r
def save_csv_jsonl(rows: List[Dict[str,str]], stem="dataset_clean_v1"):
    rows = list(rows)
    csv_path = PROC / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["texto_original","resumen","source","doc_id","split"])
        w.writeheader()
        w.writerows(rows)
    jsonl_path = PROC / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {csv_path} and {jsonl_path} (rows={len(rows)})")
if __name__ == "__main__":
    rows = parse_files()
    rows = dedup(rows)
    save_csv_jsonl(rows)