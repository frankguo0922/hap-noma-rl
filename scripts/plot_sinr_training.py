"""Plot training SINR statistics from train_metrics.csv."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ma-window", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    required = {"step", "sinr_db_p50", "sinr_db_p90", "avg_sinr_db_mean_tx"}
    if not required.issubset(df.columns):
        print(f"Unexpected CSV format: {args.csv}")
        return 1

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["step"])
    df = df.sort_values("step")

    steps = df["step"].to_numpy(dtype=np.float64)
    p50 = df["sinr_db_p50"].to_numpy(dtype=np.float64)
    p90 = df["sinr_db_p90"].to_numpy(dtype=np.float64)
    sinr_db_mean = df["avg_sinr_db_mean_tx"].to_numpy(dtype=np.float64)

    out_dir = os.path.join(os.path.dirname(args.csv), "figs")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, p50, linewidth=2, label="SINR p50 (dB)")
    plt.plot(steps, p90, linewidth=2, label="SINR p90 (dB)")
    plt.xlabel("Training Steps")
    plt.ylabel("SINR (dB)")
    plt.title("Training SINR Percentiles")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sinr_percentiles.png"), dpi=200)
    plt.show()

    if sinr_db_mean.size > 0:
        sinr_db_ma = moving_average(sinr_db_mean, args.ma_window)
        if sinr_db_ma.size == sinr_db_mean.size:
            steps_ma = steps
        else:
            steps_ma = steps[args.ma_window - 1 :]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, sinr_db_mean, linewidth=1.5, alpha=0.6, label="Mean SINR (dB)")
        plt.plot(steps_ma, sinr_db_ma, linewidth=2.0, label=f"MA{args.ma_window}")
        plt.xlabel("Training Steps")
        plt.ylabel("avg_sinr_db_mean_tx (dB)")
        plt.title("Training SINR Mean")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sinr_db_mean_ma20.png"), dpi=200)
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
