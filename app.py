import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "health_insurance.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")
    if DATA_PATH.stat().st_size == 0:
        raise ValueError(f"{DATA_PATH.name} is empty. Add the dataset to the GitHub repository.")

    data = pd.read_csv(DATA_PATH)
    if data.empty or "charges" not in data.columns:
        raise ValueError("Dataset must contain data and a 'charges' column.")
    return data


# Preprocess the data
def preprocess_data(data):
    # Example preprocessing steps
    data = data.dropna()  # Drop missing values
    data = pd.get_dummies(data, drop_first=True)  # One-hot encoding for categorical variables
    return data

# Train the model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Main function to run the app
def main():
    st.title("Health Insurance Cost Prediction")
    
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Split the data into features and target
    X = processed_data.drop('charges', axis=1)
    y = processed_data['charges']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # User input for prediction
    st.sidebar.header("User Input")
    user_input = {}
    for column in X.columns:
        user_input[column] = st.sidebar.number_input(f"Enter {column}", value=0.0)
    
    input_df = pd.DataFrame([user_input])
    
    # Make prediction
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_df)
        st.write(f"Predicted Charges: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()