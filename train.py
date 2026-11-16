"""Train the reach prediction model.

Usage:
    python train.py

Reads data/data.csv, trains a RandomForest pipeline, prints MAE / R2,
and saves the model to models/model.pkl.
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

CATEGORICAL = ["post_type", "post_day", "platform"]
NUMERIC = ["post_hour"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "engagement"
RAW_COLUMNS = ["post_type", "post_time", "platform", "post_day", "likes", "comments", "shares"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data.csv: {missing}")

    # Build the model's columns from the raw data
    df["post_hour"] = pd.to_datetime(df["post_time"], format="%m/%d/%Y %H:%M").dt.hour
    df["post_day"] = df["post_day"].str[:3]  # "Thursday" -> "Thu"
    df[TARGET] = df["likes"] + df["comments"] + df["shares"]

    return df.dropna(subset=FEATURES + [TARGET])


def build_model():
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("reg", RandomForestRegressor(n_estimators=200, random_state=42)),
    ])


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Rows used : {len(df)}")
    print(f"MAE       : {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2        : {r2_score(y_test, preds):.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()