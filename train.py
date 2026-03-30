"""Train and persist the diabetes prediction model."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes_prediction_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

CAT_COLUMNS = ["gender", "smoking_history"]
NUM_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
TARGET_COLUMN = "diabetes"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    expected_columns = set(CAT_COLUMNS + NUM_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(expected_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    x = df[CAT_COLUMNS + NUM_COLUMNS]
    y = df[TARGET_COLUMN]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLUMNS),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(cm)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    print(f"Model saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
