"""Tests for the reach prediction model.

Run:
    pytest ml_test.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest

from train import CATEGORICAL, DATA_PATH, FEATURES, NUMERIC, TARGET, build_model, load_data


def make_fake_df(n=60, seed=0):
    """Small synthetic dataset so tests don't depend on data.csv."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "post_type": rng.choice(["image", "video", "carousel", "reel"], n),
        "post_day": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
        "caption_length": rng.integers(10, 300, n),
        "num_hashtags": rng.integers(0, 30, n),
        "post_hour": rng.integers(0, 24, n),
        "followers": rng.integers(500, 5000, n),
    })
    df[TARGET] = (df["followers"] * rng.uniform(0.2, 1.5, n)).round()
    return df


def test_feature_lists_are_consistent():
    assert FEATURES == CATEGORICAL + NUMERIC
    assert TARGET not in FEATURES


def test_model_fits_and_predicts_finite_values():
    df = make_fake_df()
    model = build_model().fit(df[FEATURES], df[TARGET])

    preds = model.predict(df[FEATURES])

    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_unknown_category_does_not_crash():
    df = make_fake_df()
    model = build_model().fit(df[FEATURES], df[TARGET])

    new_row = df[FEATURES].head(1).copy()
    new_row["post_type"] = "story"  # never seen during training

    preds = model.predict(new_row)
    assert len(preds) == 1


def test_real_data_has_required_columns():
    if not os.path.exists(DATA_PATH):
        pytest.skip("data/data.csv not found yet")

    df = load_data()

    assert len(df) > 0
    assert set(FEATURES + [TARGET]).issubset(df.columns)