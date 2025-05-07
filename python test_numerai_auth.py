# test_numerai_auth.py

import os
from dotenv import load_dotenv
from numerapi import NumerAPI

load_dotenv()

public_id  = os.getenv("NUMERAI_PUBLIC_ID")
secret_key = os.getenv("NUMERAI_SECRET_KEY")
model_id   = os.getenv("NUMERAI_MODEL_ID")

print("PUBLIC_ID :", public_id)
print("SECRET_KEY:", "[oculto]" if secret_key else None)
print("MODEL_ID  :", model_id)

napi = NumerAPI(public_id=public_id, secret_key=secret_key, verbosity="info")

try:
    models = napi.get_models()
    # Simplemente imprime el resultado completo, sin slice:
    print("✅ Conexión OK. Modelos disponibles:", models)
except Exception as e:
    print("❌ Error al llamar a get_models():", e)


