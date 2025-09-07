import pandas as pd

def prepare_xy(df, text_col="texto", label_col="label"):
    # Asegurar solo las dos clases de interés
    df = df[df[label_col].isin(["pls", "non_pls"])].copy()
    # Mapear a 0/1 de forma explícita
    mapping = {"non_pls": 0, "pls": 1}
    df["y"] = df[label_col].map(mapping).astype("int64")
    # Chequeo de sanidad
    if df["y"].nunique() < 2:
        raise ValueError("Se necesitan ambas clases ('pls' y 'non_pls') para entrenar.")
    X = df[text_col].astype(str).tolist()
    y = df["y"].tolist()
    return X, y
