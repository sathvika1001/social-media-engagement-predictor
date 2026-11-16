import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from train import FEATURES, MODEL_PATH, NUMERIC, TARGET, load_data

app = Flask(__name__)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit
MAX_UPLOAD_ROWS = 500

POST_TYPES = ["image", "video", "carousel", "poll", "text"]
PLATFORMS = ["Facebook", "Instagram", "Twitter"]

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOURS = list(range(24))
_model = None


def get_model():
    """Load the trained model once and reuse it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("models/model.pkl not found. Run: python train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


def json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def clean_frame(raw):
    """Check columns and values, then return a clean DataFrame with only FEATURES."""
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = raw[FEATURES].copy()
    if df.empty:
        raise ValueError("No rows found")

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[NUMERIC].isna().any().any():
        raise ValueError("Numeric columns have empty or invalid values")
    df[NUMERIC] = df[NUMERIC].astype(int)

    if not df["post_type"].isin(POST_TYPES).all():
        raise ValueError(f"post_type must be one of {POST_TYPES}")
    if not df["platform"].isin(PLATFORMS).all():
        raise ValueError(f"platform must be one of {PLATFORMS}")
    if not df["post_hour"].between(0, 23).all():
        raise ValueError("post_hour must be between 0 and 23")

    return df.reset_index(drop=True)


def hourly_predictions(df):
    """Predicted reach for every row at each hour 0-23. Shape: (rows, 24)."""
    n = len(df)
    expanded = df.loc[df.index.repeat(24)].reset_index(drop=True)
    expanded["post_hour"] = HOURS * n
    return get_model().predict(expanded[FEATURES]).reshape(n, 24)


def to_bars(series):
    """Turn a pandas Series into bar dicts for the dashboard."""
    max_value = float(series.max()) if len(series) else 1.0
    max_value = max_value or 1.0
    return [
        {"label": str(label), "value": int(value), "pct": round(float(value) / max_value * 100)}
        for label, value in series.items()
    ]


@app.errorhandler(FileNotFoundError)
def handle_missing_file(e):
    return jsonify(error=str(e)), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify(error="File is too large (max 2 MB)"), 413


@app.post("/api/predict")
def predict():
    """Predict reach for one post."""
    try:
        df = clean_frame(pd.DataFrame([json_body()]))
    except ValueError as e:
        return jsonify(error=str(e)), 400

    reach = get_model().predict(df[FEATURES])[0]
    return jsonify(predicted_reach=max(0, round(float(reach))))


@app.post("/api/best-hours")
def best_hours():
    """Try all 24 hours for one post and return the top 3."""
    try:
        df = clean_frame(pd.DataFrame([json_body()]))
    except ValueError as e:
        return jsonify(error=str(e)), 400

    preds = hourly_predictions(df)[0]
    top = np.argsort(preds)[::-1][:3]
    return jsonify(
        best=[
            {"hour": int(h), "predicted_reach": max(0, round(float(preds[h])))}
            for h in top
        ]
    )


@app.post("/api/upload")
def upload():
    """Predict reach for every post in an uploaded CSV."""
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="Choose a CSV file first"), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify(error="Only .csv files are supported"), 400

    try:
        df = clean_frame(pd.read_csv(file.stream))
    except (ValueError, pd.errors.ParserError) as e:
        return jsonify(error=str(e)), 400

    if len(df) > MAX_UPLOAD_ROWS:
        return jsonify(error=f"Maximum {MAX_UPLOAD_ROWS} rows allowed"), 400

    reach = get_model().predict(df[FEATURES])
    hourly = hourly_predictions(df)
    best_idx = hourly.argmax(axis=1)

    rows = []
    for i, row in df.iterrows():
        rows.append({
            "row": i + 1,
            "post_type": str(row["post_type"]),
            "post_day": str(row["post_day"]),
            "post_hour": int(row["post_hour"]),
            "predicted_reach": max(0, round(float(reach[i]))),
            "best_hour": int(best_idx[i]),
            "best_hour_reach": max(0, round(float(hourly[i, best_idx[i]]))),
        })

    per_type = df.assign(reach=reach).groupby("post_type")["reach"].mean()
    summary = {
        "total_posts": len(rows),
        "avg_predicted_reach": max(0, round(float(reach.mean()))),
        "top_post_type": str(per_type.idxmax()),
        "overall_best_hour": int(np.bincount(best_idx, minlength=24).argmax()),
    }
    return jsonify(rows=rows, summary=summary)


@app.get("/api/dashboard")
def dashboard():
    """Average reach breakdown from data/data.csv."""
    df = load_data()

    by_type = df.groupby("post_type")[TARGET].mean().round().sort_values(ascending=False)
    by_day = df.groupby("post_day")[TARGET].mean().round().reindex(DAYS).dropna()
    by_hour = df.groupby("post_hour")[TARGET].mean().round().reindex(HOURS).dropna()

    return jsonify(
        total_posts=len(df),
        avg_reach=round(float(df[TARGET].mean())),
        best_post_type=str(by_type.index[0]),
        best_hour=int(by_hour.idxmax()),
        by_type=to_bars(by_type),
        by_day=to_bars(by_day),
        by_hour=to_bars(by_hour),
    )

@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)