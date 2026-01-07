from dotenv import load_dotenv
from numerapi import NumerAPI
import os
import pandas as pd

def make_submission(model, model_id) -> str:
    # Authenticate using keys from .env file
    load_dotenv()
    NERMAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID")
    NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY")
    napi = NumerAPI(NERMAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # Get current round
    current_round = napi.get_current_round()
    print(f"Current round: {current_round}")

    # Download latest live features
    VERSION = "v5.2"
    napi.download_dataset(f"{VERSION}/live_{current_round}.parquet")
    live_data = pd.read_parquet(f"{VERSION}/live_{current_round}.parquet")
    live_features = live_data[[f for f in live_data.columns if "feature" in f]]

    # Generate live predictions
    # live_predictions = model.predict(live_features)
    live_predictions = pd.random.rand(len(live_features))

    # Format submission
    submission = pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
    submission.to_csv(f"prediction_{current_round}.csv")
    print(f"Submission file created: prediction_{current_round}.csv")

    # Upload submission 
    submission_id = napi.upload_predictions(f"prediction_{current_round}.csv", model_id)
    return submission_id