import os
from numerapi import NumerAPI
#verificando secrets
public_id = os.getenv("NUMERAI_PUBLIC_ID")
secret_key = os.getenv("NUMERAI_SECRET_KEY")
model_id = os.getenv("NUMERAI_MODEL_ID")

napi = NumerAPI(public_id=public_id, secret_key=secret_key)
submission_file = "submission.csv"
napi.upload_predictions(submission_file, model_id=model_id)

