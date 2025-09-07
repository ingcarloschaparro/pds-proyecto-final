"""Clasificador DistilBERT para distinguir entre textos por favor si non-por favor"""

from src.config.mlflow_remote import apply_tracking_uri as _mlf_apply
_mlf_apply(experiment="E2-DistilBERT")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import os
from pathlib import Path
import json
from typing import Dict, Any, Tuple
import warnings
import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")


class TextClassificationDataset(Dataset):
    """Dataset personalizado para clasificación de textos"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenizar
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def preparar_datos_distilbert(
    ruta_datos: str, sample_size: int = None
) -> Tuple[pd.Series, pd.Series]:
    """Prepara datos para DistilBERT Args: ruta_datos: Ruta al archivo CSV con datos procesados sample_size: Tamaño de muestra para entrenamiento rápido (opcional) Returns: Tuple con features (ex) si labels (si)"""
    print("Cargando datos para DistilBERT...")

    # Cargar datos
    df = pd.read_csv(ruta_datos, low_memory=False)

    # Filtrar solo registros con labels válidos
    df_valid = df[df["label"].notna()].copy()
    print(f"Registros con labels válidos: {len(df_valid)} de {len(df)}")

    # Preparar features
    textos = []
    labels = []

    print(f"Procesando {len(df_valid)} registros válidos...")
    print("Distribución de labels:", df_valid["label"].value_counts().to_dict())

    # Procesar datos PLS y non-PLS
    for _, row in df_valid.iterrows():
        # Para PLS: usar el resumen como feature
        if row["label"] == "pls":
            resumen = row["resumen"]
            if pd.isna(resumen):
                texto = ""
            else:
                texto = str(resumen).strip()

            if len(texto) > 10:
                textos.append(texto)
                labels.append(1)  # PLS

        # Para non-PLS: usar el texto original
        elif row["label"] == "non_pls":
            texto_original = row["texto_original"]
            if pd.isna(texto_original):
                texto = ""
            else:
                texto = str(texto_original).strip()

            if len(texto) > 10:
                textos.append(texto)
                labels.append(0)  # non-PLS

    # Aplicar sample si se especifica (para pruebas rápidas)
    if sample_size and len(textos) > sample_size:
        indices = np.random.choice(len(textos), sample_size, replace=False)
        textos = [textos[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"Aplicando muestra de {sample_size} registros...")

    print(f"Datos preparados: {len(textos)} textos, {len(labels)} labels")
    print(f"Distribución de clases: {pd.Series(labels).value_counts()}")

    return pd.Series(textos), pd.Series(labels)


def entrenar_distilbert(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> Dict[str, Any]:
    """Entrena modelo DistilBERT para clasificación con logging MLflow"""

    with mlflow.start_run():
        # Loggear parámetros
        mlflow.log_params({
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        })

        print("=== ENTRENAMIENTO DISTILBERT ===")
        print(f"Modelo: {model_name}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {learning_rate}")

        # Verificar si hay GPU disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")
        
        # Loggear información del dispositivo
        mlflow.log_param("device", str(device))

        # Cargar tokenizer y modelo
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2, output_attentions=False, output_hidden_states=False
        )
        model.to(device)

        # Crear datasets
        train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_length)
        test_dataset = TextClassificationDataset(X_test, y_test, tokenizer, max_length)

        # Crear dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Configurar optimizador y scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Entrenamiento
        model.train()
        best_f1 = 0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0
            model.train()

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Pérdida promedio: {avg_loss:.4f}")
            
            # Loggear pérdida por época
            mlflow.log_metric(f"train_loss_epoch_{epoch+1}", avg_loss)

            # Evaluación en validación
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Calcular métricas
            val_f1 = f1_score(val_labels, val_preds, average="macro")
            print(f"F1-Score validación: {val_f1:.4f}")
            
            # Loggear métricas por época
            mlflow.log_metric(f"val_f1_epoch_{epoch+1}", val_f1)

            # Guardar mejor modelo
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()

        # Cargar mejor modelo
        model.load_state_dict(best_model_state)

        # Evaluación final
        print("\n=== EVALUACIÓN FINAL ===")
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                probs = torch.softmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Métricas finales
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        f1_pls = f1_score(all_labels, all_preds, pos_label=1)
        f1_non_pls = f1_score(all_labels, all_preds, pos_label=0)

        print(f"F1-Score macro: {f1_macro:.4f}")
        print(f"F1-Score weighted: {f1_weighted:.4f}")
        print(f"F1-Score PLS: {f1_pls:.4f}")
        print(f"F1-Score non-PLS: {f1_non_pls:.4f}")

        # Loggear métricas finales
        mlflow.log_metrics({
            "final_f1_macro": f1_macro,
            "final_f1_weighted": f1_weighted,
            "final_f1_pls": f1_pls,
            "final_f1_non_pls": f1_non_pls,
            "best_val_f1": best_f1
        })

        # Reporte de clasificación
        print("\nReporte de clasificación:")
        report = classification_report(
            all_labels, all_preds, target_names=["non-por favor", "por favor"]
        )
        print(report)

        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        print("\nMatriz de confusión:")
        print(cm)

        # Loggear matriz de confusión
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Guardar modelo en MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_text(str(tokenizer), "tokenizer_info.txt")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_pls": f1_pls,
            "f1_non_pls": f1_non_pls,
            "classification_report": classification_report(
                all_labels,
                all_preds,
                target_names=["non-por favor", "por favor"],
                output_dict=True,
            ),
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds,
            "probabilities": all_probs,
            "best_f1": best_f1,
        }


def guardar_modelo_distilbert(resultados: Dict[str, Any], ruta_modelo: str) -> None:
    """Guarda el modelo DistilBERT entrenado"""
    os.makedirs(ruta_modelo, exist_ok=True)

    # Guardar modelo y tokenizer
    resultados["model"].save_pretrained(ruta_modelo)
    resultados["tokenizer"].save_pretrained(ruta_modelo)

    # Guardar métricas
    metricas = {
        "f1_macro": resultados["f1_macro"],
        "f1_weighted": resultados["f1_weighted"],
        "f1_pls": resultados["f1_pls"],
        "f1_non_pls": resultados["f1_non_pls"],
        "best_f1": resultados["best_f1"],
        "classification_report": resultados["classification_report"],
        "confusion_matrix": resultados["confusion_matrix"],
    }

    with open(f"{ruta_modelo}/metricas_distilbert.json", "w") as f:
        json.dump(metricas, f, indent=2)

    print(f"Modelo DistilBERT guardado en: {ruta_modelo}")


def main():
    """Función principal para entrenar DistilBERT"""

    # Configuración
    RUTA_DATOS = "data/processed/dataset_clean_v1.csv"
    RUTA_MODELO = "models/clasificador_distilbert"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Configuración de entrenamiento (reducida para pruebas rápidas)
    SAMPLE_SIZE = 1000  # Reducir para pruebas más rápidas
    BATCH_SIZE = 8  # Reducir batch size para memoria
    EPOCHS = 2  # Pocas épocas para pruebas
    MAX_LENGTH = 256  # Longitud máxima de secuencia

    print("=== ENTRENAMIENTO CLASIFICADOR DISTILBERT ===")
    print(f"Usando muestra de {SAMPLE_SIZE} registros para entrenamiento rápido...")

    try:
        # Preparar datos
        X, y = preparar_datos_distilbert(RUTA_DATOS, SAMPLE_SIZE)

        # Verificar que tenemos ambas clases
        unique_labels = y.unique()
        print(f"Clases encontradas: {unique_labels}")

        if len(unique_labels) < 2:
            print("Error: Se necesita al menos dos clases para clasificación")
            return None

        # Convertir a listas para evitar problemas de indexación
        X_list = X.tolist()
        y_list = y.tolist()
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_list, y_list, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_list
        )
        
        # Convertir de vuelta a Series
        X_train = pd.Series(X_train, name='textos')
        X_test = pd.Series(X_test, name='textos')
        y_train = pd.Series(y_train, name='labels')
        y_test = pd.Series(y_test, name='labels')

        print(f"Conjunto de entrenamiento: {len(X_train)} registros")
        print(f"Conjunto de prueba: {len(X_test)} registros")

        # Entrenar modelo
        resultados = entrenar_distilbert(
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            max_length=MAX_LENGTH,
        )

        # Guardar modelo
        guardar_modelo_distilbert(resultados, RUTA_MODELO)

        print("Entrenamiento DistilBERT completado exitosamente!")

        return resultados

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

