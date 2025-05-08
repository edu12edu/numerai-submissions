
import os
from numerapi import NumerAPI
# Ejecutando para comprobar secrets

# Recuperar variables del entorno
public_id = os.getenv("NUMERAI_PUBLIC_ID")
secret_key = os.getenv("NUMERAI_SECRET_KEY")
model_id = os.getenv("NUMERAI_MODEL_ID")

# Verificar si las variables se están leyendo correctamente
print("Public ID encontrado:", bool(public_id))
print("Secret Key encontrada:", bool(secret_key))
print("Model ID encontrado:", bool(model_id))

# Lanzar error si falta alguna
if not all([public_id, secret_key, model_id]):
    raise ValueError("Alguna variable de entorno no está definida. Revisa los GitHub Secrets.")

# Continuar con la API de Numerai
napi = NumerAPI(public_id=public_id, secret_key=secret_key)
submission_file = "submission.csv"
napi.upload_predictions(submission_file, model_id=model_id)


