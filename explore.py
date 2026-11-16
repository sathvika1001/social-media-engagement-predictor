"""Quick exploratory analysis of data/data.csv.

Usage:
    python explore.py

Prints a summary and saves two charts to static/plots/.
"""
import os

import matplotlib

matplotlib.use("Agg")  # no GUI window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from train import DATA_PATH, NUMERIC, TARGET  # noqa: E402

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "plots")


def save_bar(series, title, xlabel, filename):
    plt.figure(figsize=(7, 4))
    series.plot(kind="bar", color="#4f46e5")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(f"Average {TARGET}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    print("=== First 5 rows ===")
    print(df.head())
    print("\n=== Info ===")
    df.info()
    print("\n=== Summary ===")
    print(df.describe(include="all"))
    print("\n=== Missing values ===")
    print(df.isna().sum())

    print(f"\n=== Correlation with {TARGET} ===")
    print(df[NUMERIC + [TARGET]].corr()[TARGET].sort_values(ascending=False))

    os.makedirs(PLOT_DIR, exist_ok=True)

    by_type = df.groupby("post_type")[TARGET].mean().sort_values(ascending=False)
    print("\n=== Average reach by post type ===")
    print(by_type.round(0))
    save_bar(by_type, "Average reach by post type", "Post type", "by_post_type.png")

    by_hour = df.groupby("post_hour")[TARGET].mean()
    save_bar(by_hour, "Average reach by posting hour", "Hour (0-23)", "by_hour.png")

    print(f"\nCharts saved in {PLOT_DIR}")


if __name__ == "__main__":
    main()