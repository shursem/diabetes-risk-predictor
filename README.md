# 🩺 Diabetes Risk Predictor

An end-to-end machine learning project that predicts the likelihood of diabetes based on patient health data using a trained classification model and an interactive Streamlit web app.

---

## 🚀 Features

* 📊 Exploratory Data Analysis (EDA) and feature engineering
* 🧠 Machine learning model training with preprocessing pipeline
* 💾 Model serialization using Pickle
* ⚡ Real-time predictions via Streamlit UI
* 📁 Organized project structure for reproducibility

---

## 📂 Project Structure

```
diabetes-risk-predictor/
│
├── app.py                  # Streamlit app
├── train.py                # Model training script
├── requirements.txt
├── README.md
│
├── model/
│   └── model.pkl           # Trained model
│
├── data/
│   └── diabetes_prediction_dataset.csv
│
└── notebooks/
    └── diabetes_analysis.ipynb  # EDA & feature engineering
```

---

## ⚙️ Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python train.py
```

This will generate the trained model at:

```
model/model.pkl
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📊 Input Features

* Gender
* Age
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

---

## 🎯 Output

* `0` → Low Risk
* `1` → High Risk

---

## 📊 Exploratory Data Analysis

The `notebooks/` folder contains:

* Data cleaning
* Feature encoding
* Distribution analysis
* Insights used for model building

---

## 💡 Key Learning

This project demonstrates:

* End-to-end ML workflow
* Data preprocessing & encoding
* Model deployment using Streamlit
* Building interactive ML applications

---

## 🚀 Future Improvements

* Add prediction probability (confidence score)
* Improve UI with charts and visualizations
* Deploy app to cloud (Streamlit Cloud / Render)
* Add user authentication & history tracking

---

## 👨‍💻 Author

Shursem Vashum
