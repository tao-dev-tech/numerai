import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI

def make_submission(model, model_id) -> str:
    # Authenticate using keys from .env file
    load_dotenv()
    NUMERAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID")
    NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY")
    if not NUMERAI_PUBLIC_ID or not NUMERAI_SECRET_KEY:
        raise ValueError(
            "Missing NUMERAI_PUBLIC_ID or NUMERAI_SECRET_KEY in environment."
        )
    napi = NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # Get current round
    current_round = napi.get_current_round()
    print(f"Current round: {current_round}")

    # Download latest live features
    VERSION = "v5.2"
    napi.download_dataset(f"{VERSION}/live_{current_round}.parquet")
    live_data = pd.read_parquet(f"{VERSION}/live_{current_round}.parquet")
    live_features = live_data[[f for f in live_data.columns if "feature" in f]]

    # Generate live predictions
    if hasattr(model, "predict"):
        live_predictions = model.predict(live_features)
    else:
        print("No model with predict() provided; using random predictions.")
        live_predictions = np.random.default_rng().random(len(live_features))

    # Format submission
    submission = pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
    submission.to_csv(f"prediction_{current_round}.csv")
    print(f"Submission file created: prediction_{current_round}.csv")

    # Upload submission 
    submission_id = napi.upload_predictions(f"prediction_{current_round}.csv", model_id)
    return submission_id