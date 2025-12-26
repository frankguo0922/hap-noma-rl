"""Plot training SINR behavior over steps."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise FileNotFoundError(f"Empty metrics file: {csv_path}")
        header_map = {name: idx for idx, name in enumerate(header)}
        if "step" not in header_map or "avg_sinr_mean_tx" not in header_map:
            raise ValueError(f"Unexpected CSV format: {csv_path}")
        step_idx = header_map["step"]
        sinr_idx = header_map["avg_sinr_mean_tx"]
        sinr_db_idx = header_map.get("avg_sinr_db_mean_tx")
        steps = []
        sinr = []
        sinr_db = []
        for row in reader:
            if not row:
                continue
            try:
                steps.append(float(row[step_idx]))
                sinr.append(float(row[sinr_idx]))
                if sinr_db_idx is not None:
                    sinr_db.append(float(row[sinr_db_idx]))
            except (ValueError, IndexError):
                continue
    sinr_db_arr = np.array(sinr_db, dtype=np.float64) if sinr_db else np.array([])
    return np.array(steps, dtype=np.float64), np.array(sinr, dtype=np.float64), sinr_db_arr


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def cumulative_average(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.cumsum(values) / np.arange(1, values.size + 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    run_dir = os.path.join("logs", "runs", args.run_id)
    csv_path = os.path.join(run_dir, "train_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    try:
        steps, sinr, sinr_db = load_metrics(csv_path)
    except ValueError as exc:
        print(str(exc))
        return 1

    mask = np.isfinite(steps) & np.isfinite(sinr)
    steps = steps[mask]
    sinr = sinr[mask]

    order = np.argsort(steps)
    steps = steps[order]
    sinr = sinr[order]

    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, sinr, linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("avg_sinr_mean_tx (linear)")
    plt.title("Training SINR (Raw)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out1 = os.path.join(figs_dir, "train_sinr_raw.png")
    plt.savefig(out1, dpi=200)
    plt.show()

    window = 20
    sinr_ma = moving_average(sinr, window)
    steps_ma = steps[window - 1 :] if sinr_ma.size != sinr.size else steps
    sinr_db_ma = 10.0 * np.log10(np.maximum(sinr_ma, 1e-12))
    plt.figure(figsize=(8, 5))
    plt.plot(steps_ma, sinr_db_ma, linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("avg_sinr_mean_tx (dB)")
    plt.title("Training SINR (dB, MA20)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out2 = os.path.join(figs_dir, "train_sinr_ma20.png")
    plt.savefig(out2, dpi=200)
    plt.show()

    sinr_cum = cumulative_average(sinr)
    sinr_db_cum = 10.0 * np.log10(np.maximum(sinr_cum, 1e-12))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, sinr_db_cum, linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("avg_sinr_mean_tx (dB)")
    plt.title("Training SINR (dB, CUMAVG)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out3 = os.path.join(figs_dir, "train_sinr_cumavg.png")
    plt.savefig(out3, dpi=200)
    plt.show()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
