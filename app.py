from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "health_insurance.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


@st.cache_data
def load_data():
    if not DATA_PATH.is_file() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError("health_insurance.csv is missing or empty.")
    data = pd.read_csv(DATA_PATH)
    if data.empty or "charges" not in data.columns:
        raise ValueError("CSV must contain data and a 'charges' column.")
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("model.pkl is missing. Run create_model.py first.")
    return joblib.load(MODEL_PATH)


def main():
    st.title("Health Insurance Cost Prediction")

    try:
        data = load_data()
        model = load_model()
    except Exception as error:
        st.error(str(error))
        st.stop()

    features = data.drop(columns="charges")
    st.sidebar.header("Enter your details")
    user_input = {}

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            user_input[column] = st.sidebar.number_input(
                column.replace("_", " ").title(),
                value=float(features[column].median()),
            )
        else:
            options = sorted(features[column].dropna().astype(str).unique())
            if not options:
                st.error(f"No available values for {column}.")
                st.stop()
            user_input[column] = st.sidebar.selectbox(
                column.replace("_", " ").title(), options
            )

    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=features.columns)
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted charges: ${max(0, prediction):,.2f}")


if __name__ == "__main__":
    main()