"""Train and save the Streamlit app's CatBoost model bundle."""

from pathlib import Path

import joblib
import pandas as pd

from insurance_model import train_models


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "health_insurance.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def main() -> None:
    if not DATA_PATH.is_file() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Training dataset is missing or empty: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip().str.lower()
    bundle = train_models(data)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved {bundle['model_name']} bundle to {MODEL_PATH}")
    print("A separate scaler is not required for CatBoost.")


if __name__ == "__main__":
    main()
