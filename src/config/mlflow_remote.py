# src/config/mlflow_remote.py
from __future__ import annotations
import os, socket, subprocess, mlflow

DEFAULT_URI = "http://52.0.127.25:5001"
DEFAULT_EXPERIMENT = "Entrega2"

def apply_tracking_uri(experiment: str | None = None) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI)
    mlflow.set_tracking_uri(uri)

    exp_name = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    mlflow.set_experiment(exp_name)

    try:
        mlflow.set_tag("runner.user", os.getenv("USER") or os.getenv("USERNAME") or "unknown")
    except Exception:
        pass
    try:
        mlflow.set_tag("runner.host", socket.gethostname())
    except Exception:
        pass
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        mlflow.set_tag("git_commit", sha)
    except Exception:
        pass
