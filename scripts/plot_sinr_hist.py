"""Plot SINR histogram from training metrics."""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def style_axes(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": dpi,
        }
    )


def load_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    columns[name].append(np.nan)
    return {k: np.array(v, dtype=np.float64) for k, v in columns.items()}


def find_column(columns: List[str], include: List[str], exclude: List[str] | None = None) -> str | None:
    exclude = exclude or []
    for col in columns:
        key = col.lower()
        if all(token in key for token in include) and not any(token in key for token in exclude):
            return col
    return None


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if values.size == 0 or sigma <= 0:
        return values
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return 1

    try:
        data = load_csv(args.csv)
    except ValueError as exc:
        print(f"Failed to read CSV: {exc}")
        return 1

    columns = list(data.keys())
    sinr_col = find_column(columns, ["sinr"], ["target"])
    if sinr_col is None:
        print("SINR column not found.")
        print("Available columns:", ", ".join(columns))
        return 1

    sinr = data[sinr_col]
    sinr = sinr[np.isfinite(sinr)]
    if sinr.size == 0:
        print("No valid SINR values found.")
        return 1

    style_axes(args.dpi)
    plt.figure(figsize=(8, 5))
    hist, bin_edges = np.histogram(sinr, bins=args.bins, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.bar(centers, hist, width=np.diff(bin_edges), alpha=0.6, label="Histogram", color="#4c78a8")
    if args.smooth_sigma > 0:
        smooth = gaussian_smooth(hist, sigma=args.smooth_sigma)
        plt.plot(centers, smooth, color="#f58518", linewidth=2, label="Smoothed")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Density")
    plt.title("Training SINR Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = args.out_path or os.path.join("figures", "sinr_hist.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
