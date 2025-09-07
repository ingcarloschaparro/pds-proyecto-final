from __future__ import annotations
import os, socket, time, random
import mlflow
from src.config.mlflow_remote import apply_tracking_uri as _mlf_apply

def log_quick_run(exp, run_name, params, metrics):
    _mlf_apply(experiment=exp)
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.set_tag("runner.host", socket.gethostname())
        mlflow.set_tag("status", "smoke")
        # pequeño artefacto de texto
        p = f"artifacts_{run_name.replace(' ','_')}.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"Run: {run_name}\nExperiment: {exp}\nHost: {socket.gethostname()}\n")
        mlflow.log_artifact(p)

exps = [
    ("E2-Classifier-Baseline", "smoke-baseline", {"model":"tfidf+logreg"}, {"f1_macro":0.0, "accuracy":0.0}),
    ("E2-DistilBERT", "smoke-distilbert", {"model":"distilbert-base-uncased"}, {"f1_macro":0.0}),
    ("E2-Compare-PLS", "smoke-compare", {"candidates":"bart_base,bart_large,t5_base"}, {"count":3}),
    ("E2-Pipeline", "smoke-pipeline", {"stages":"classify->summarize->evaluate"}, {"ok":1}),
]
for exp, rn, params, metrics in exps:
    log_quick_run(exp, rn, params, metrics)
print("Smoke runs listos.")
