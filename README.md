# Diabetes Prediction ML Project

## Overview

This project is an end-to-end machine learning pipeline that predicts diabetes risk from patient data. It includes:

- Data preprocessing for numerical and categorical features
- Model training and evaluation
- Model serialization
- A Streamlit web app for predictions

## Project Structure

```text
Diabetis_app/
|-- data/
|   `-- diabetes_prediction_dataset.csv
|-- model/
|   `-- model.pkl
|-- notebooks/
|   `-- diabetes_analysis.ipynb
|-- app.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## Setup

Use one Python environment for both training and app execution.

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Training

```powershell
python train.py
```

This creates `model/model.pkl`.

## Run the App

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` ensures Streamlit runs in the same environment as your installed packages.

## Input Features

- gender
- age
- hypertension
- heart_disease
- smoking_history
- bmi
- HbA1c_level
- blood_glucose_level

## Target

- diabetes (`0` = no diabetes, `1` = diabetes)

## Troubleshooting

- If training fails with `ModuleNotFoundError`, install dependencies in the active environment.
- If the app says model could not be loaded, run `python train.py` again to regenerate `model/model.pkl`.
 To run the app use the following Command 
 streamlit run d:/Diabetis_app/app.py