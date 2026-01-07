import lightgbm as lgb
import matplotlib.pyplot as plt
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution
import json
import pandas as pd

# Initialize NumerAPI
napi = NumerAPI()

# list the datasets and available versions
all_datasets = napi.list_datasets()
dataset_versions = list(set(d.split("/")[0] for d in all_datasets))
print("Available versions:\n", dataset_versions)

# Set data version to one of the latest datasets
DATA_VERSION = "v5.2"

# Print all files available for download for our version
current_version_files = [f for f in all_datasets if f.startswith(DATA_VERSION)]
print("Available", DATA_VERSION, "files:\n", current_version_files)

# download the feature metadata file
napi.download_dataset(f"{DATA_VERSION}/features.json")

# read the metadata and display
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_sets = feature_metadata["feature_sets"]

feature_set = feature_sets["small"]
napi.download_dataset(f"{DATA_VERSION}/train.parquet")
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet", columns=["era", "target"] + feature_set
)

model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5 - 1,
    colsample_bytree=0.1,
    device="gpu",  # Enable GPU acceleration
    gpu_platform_id=0,  # OpenCL platform ID (usually 0)
    gpu_device_id=0,  # GPU device ID (usually 0 for single GPU)
)

model.fit(train[feature_set], train["target"])

# Download validation data - this will take a few minutes
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

# Load the validation data and filter for data_type == "validation"
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type", "target"] + feature_set,
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# Generate predictions against the out-of-sample validation features
# This will take a few minutes 🍵
validation["prediction"] = model.predict(validation[feature_set])
validation[["era", "prediction", "target"]]

# Download and join in the meta_model for the validation eras
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(f"v4.3/meta_model.parquet")[
    "numerai_meta_model"
]

# Compute the per-era corr between our predictions and the target values
per_era_corr = validation.groupby("era").apply(
    lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
)

# Compute the per-era mmc between our predictions, the meta model, and the target values
per_era_mmc = (
    validation.dropna()
    .groupby("era")
    .apply(
        lambda x: correlation_contribution(
            x[["prediction"]], x["meta_model"], x["target"]
        )
    )
)

# Compute performance metrics
corr_mean = per_era_corr.mean()
corr_std = per_era_corr.std(ddof=0)
corr_sharpe = corr_mean / corr_std
corr_max_drawdown = (
    per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()
).max()

mmc_mean = per_era_mmc.mean()
mmc_std = per_era_mmc.std(ddof=0)
mmc_sharpe = mmc_mean / mmc_std
mmc_max_drawdown = (
    per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()
).max()

pd.DataFrame(
    {
        "mean": [corr_mean, mmc_mean],
        "std": [corr_std, mmc_std],
        "sharpe": [corr_sharpe, mmc_sharpe],
        "max_drawdown": [corr_max_drawdown, mmc_max_drawdown],
    },
    index=["CORR", "MMC"],
).T

# Download latest live features
napi.download_dataset(f"{DATA_VERSION}/live.parquet")

# Load live features
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=feature_set)

# Generate live predictions
live_predictions = model.predict(live_features[feature_set])

# Format submission
pd.Series(live_predictions, index=live_features.index).to_frame("prediction")

live_predictions = model.predict(live_features[feature_set])
submission = pd.Series(live_predictions, index=live_features.index)
submission = submission.to_frame("prediction")

print(submission)
