# Insurance Regression App

This project is a Streamlit application designed to analyze health insurance data and make predictions based on various features. The application utilizes a regression model to estimate insurance charges based on user inputs.

## Project Structure

- `app.py`: The Streamlit interface for entering insurance details and viewing a prediction.
- `insurance_model.py`: Training and inference helpers for the smoker-segmented CatBoost model.
- `make_model.py`: Rebuilds the serialized model bundle from the bundled CSV.
- `model.pkl`: Serialized CatBoost models used by the app. It is generated locally before deployment.
- `life_cover_illustration.svg`: Locally hosted life-insurance themed artwork used in the responsive page background.
- `health_insurance.csv`: The dataset used by the application, containing health insurance data for analysis and predictions.
- `requirements.txt`: Lists the Python packages required to run the application, including Streamlit, pandas, numpy, and other dependencies.
- `.gitignore`: Specifies files and directories to be ignored by Git, such as `__pycache__/`, `.DS_Store`, and environment files.
- `.streamlit/config.toml`: Configuration file for Streamlit-specific settings, including theme and layout of the application.
- `notebooks/Biswarup_Debbarma_B15_Mini_Project_Regression.ipynb`: A Jupyter notebook containing exploratory data analysis and model training code, serving as a reference for the development of the Streamlit app.

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd insurance-regression-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Train and save the model bundle:
   ```
   python make_model.py
   ```
   This creates `model.pkl` beside the app. CatBoost handles the numeric values and
   categorical features directly, so a separate scaler is not needed or used.

## Usage

After generating `model.pkl`, run the Streamlit application:
```
streamlit run app.py
```

The responsive interface uses a local family-and-shield illustration, so the themed
background does not depend on a remote image host. The profile form accepts the
six fields used by the model and displays its estimated annual cost after submission.
The development server is available at `http://localhost:8501`.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.