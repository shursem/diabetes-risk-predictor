"""Streamlit app for diabetes risk prediction."""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    """Load the trained pipeline once per app session."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}. Train the model first with `python train.py`."
        )
    with path.open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:")
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

try:
    pipeline = load_pipeline(MODEL_PATH)
except Exception as exc:  # pragma: no cover - streamlit runtime path
    st.error("Unable to load the trained model.")
    st.info("Install dependencies and run `python train.py`, then restart the app.")
    st.exception(exc)
    st.stop()

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=0.0, value=20.0)
hba1c = st.number_input("HbA1c Level", min_value=0.0, value=5.0)
glucose = st.number_input("Blood Glucose Level", min_value=0.0, value=100.0)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
smoking = st.selectbox(
    "Smoking History",
    ["never", "No Info", "current", "former", "ever", "not current"],
)

hypertension = st.selectbox("Hypertension", [0, 1])
heart = st.selectbox("Heart Disease", [0, 1])

if st.button("Predict"):
    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart],
            "smoking_history": [smoking],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose],
        }
    )

    prediction = pipeline.predict(input_data)
    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
