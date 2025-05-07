# hello_numerai.py
# -----------------
# 1) Instala las dependencias:
#    pip install numerapi pandas numpy lightgbm python-dotenv pyarrow

import os
import json
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI
from dotenv import load_dotenv

# 2) Carga tu .env
load_dotenv()

# 3) Inicializa NumerAPI
napi = NumerAPI(
    public_id=os.getenv("NUMERAI_PUBLIC_ID"),
    secret_key=os.getenv("NUMERAI_SECRET_KEY"),
    verbosity="info"
)

# 4) Descarga train/validation/live si no existen
for split in ["train", "validation", "live"]:
    remote = f"v4/{split}.parquet"
    local  = f"{split}.parquet"
    if not os.path.exists(local):
        print(f"⬇️  Descargando {remote} → {local}")
        napi.download_dataset(remote, local)

# 5) Carga features.json y elige ’small’
with open("features.json") as f:
    feature_sets = json.load(f)["feature_sets"]
features = feature_sets.get("small", [])
if not features:
    raise KeyError("No se encontró el set 'small' en features.json")
print(f"▶️ Usando {len(features)} features ('small').")

# 6) Carga TRAIN / VALIDATION
print("⚙️ Cargando TRAIN y VALIDATION...")
train = pd.read_parquet("train.parquet", columns=features + ["target"])
val   = pd.read_parquet("validation.parquet", columns=features + ["target"])

X_train, y_train = train[features], train["target"]
X_val,   y_val   = val  [features], val  ["target"]

# 7) Entrena LightGBM con iteraciones fijas
model = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
print("⚙️ Entrenando LightGBM (100 iteraciones)...")
model.fit(X_train, y_train)
print("▶️ Entrenamiento completado.")

# 8) Predice sobre LIVE y genera submission.csv
print("⚙️ Cargando LIVE y calculando predicciones...")
df_live = pd.read_parquet("live.parquet")

# Detecta columna de ID
if "id" in df_live.columns:
    id_col = "id"
elif "row_id" in df_live.columns:
    id_col = "row_id"
else:
    # crea un id si no existe
    df_live["id"] = df_live.index
    id_col = "id"

# Asegura que estén todas las features
for feat in features:
    if feat not in df_live.columns:
        df_live[feat] = 0.0

df_live["prediction"] = model.predict(df_live[features])

# Exporta
df_live[[id_col, "prediction"]].to_csv("submission.csv", index=False)
print("💾 submission.csv creado")

# 9) Sube las predicciones
print("📡 Subiendo a Numerai…")
napi.upload_predictions("submission.csv", model_id=os.getenv("NUMERAI_MODEL_ID"))
print("✅ ¡Hecho!")





