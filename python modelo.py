#!/usr/bin/env python3
"""
Entrenamiento incremental por eras con SGDRegressor y subida de predicciones.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from numerapi import NumerAPI
from dotenv import load_dotenv

# Parámetros y paths
N_EPOCHS       = 3
TRAIN_PATH     = "train.parquet"
LIVE_PATH      = "live.parquet"
FEATURES_PATH  = "features.json"
SUBMISSION_CSV = "submission.csv"

def load_env():
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()

def get_client():
    public_id  = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    model_id   = os.getenv("NUMERAI_MODEL_ID")
    if not (public_id and secret_key and model_id):
        raise EnvironmentError("Define NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY y NUMERAI_MODEL_ID en .env")
    return NumerAPI(public_id=public_id, secret_key=secret_key, verbosity="info"), model_id

def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"No existe {FEATURES_PATH}")
    with open(FEATURES_PATH, "r") as f:
        sets = json.load(f).get("feature_sets", {})
    feats = sets.get("small")
    if not feats:
        raise KeyError("No se encontró el set 'small' en features.json")
    return feats

def train_incremental(features):
    print("⚙️ Cargando TRAIN...")
    df = pd.read_parquet(TRAIN_PATH, columns=features + ["era", "target"])
    eras = sorted(df["era"].unique())
    print(f"▶️ {len(eras)} eras detectadas.")

    scaler = StandardScaler()
    model  = SGDRegressor(
        loss="huber", penalty="l2", alpha=1e-4,
        learning_rate="invscaling", eta0=0.01, power_t=0.25,
        max_iter=1, warm_start=True, tol=None, random_state=42
    )
    start = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        print(f"=== Epoch {epoch}/{N_EPOCHS} ===")
        for era in eras:
            block = df[df["era"] == era]
            X = block[features].to_numpy()
            y = block["target"].to_numpy()
            scaler.partial_fit(X)
            Xs = scaler.transform(X)
            y_z = (y - y.mean()) / (y.std() + 1e-8)
            model.partial_fit(Xs, y_z)
        print(f"▶️ Fin epoch {epoch} (t={time.time()-start:.1f}s)")

    return scaler, model

def predict_and_save(scaler, model, features):
    print("⚙️ Cargando LIVE...")
    df_live = pd.read_parquet(LIVE_PATH)
    for feat in features:
        if feat not in df_live.columns:
            df_live[feat] = 0.0
    X_live = scaler.transform(df_live[features].to_numpy())
    preds  = model.predict(X_live)

    if "id" in df_live.columns:
        id_col = "id"
    elif "row_id" in df_live.columns:
        id_col = "row_id"
    else:
        print("⚠️ No hay 'id' ni 'row_id'; uso índice")
        df_live["id"] = df_live.index
        id_col = "id"

    sub = pd.DataFrame({
        "id": df_live[id_col].to_numpy(),
        "prediction": preds
    })
    sub.to_csv(SUBMISSION_CSV, index=False)
    print(f"💾 {SUBMISSION_CSV} creado ({len(sub)} filas).")
    return SUBMISSION_CSV

def upload(sub_csv, client, model_id):
    try:
        client.upload_predictions(sub_csv, model_id=model_id)
        print("📡 Predicciones subidas con éxito.")
    except Exception as e:
        print("⚠️ Error al subir:", e)

def main():
    load_env()
    client, model_id = get_client()
    feats = load_features()
    scaler, model = train_incremental(feats)
    csv_path = predict_and_save(scaler, model, feats)
    upload(csv_path, client, model_id)

if __name__ == "__main__":
    main()















