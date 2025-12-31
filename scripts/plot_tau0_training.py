"""Plot tau0 behavior from training metrics."""
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ma_window", type=int, default=20)
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
    tau0_col = find_column(columns, ["tau0"])
    step_col = find_column(columns, ["step"]) or find_column(columns, ["steps"])
    if tau0_col is None or step_col is None:
        print("Required columns not found for tau0 plots.")
        print("Available columns:", ", ".join(columns))
        return 1

    steps = data[step_col]
    tau0 = data[tau0_col]
    mask = np.isfinite(steps) & np.isfinite(tau0)
    steps = steps[mask]
    tau0 = tau0[mask]
    if steps.size == 0:
        print("No valid tau0 data points found.")
        return 1

    order = np.argsort(steps)
    steps = steps[order]
    tau0 = tau0[order]

    out_dir = args.out_dir or os.path.join("figures", "generated")
    os.makedirs(out_dir, exist_ok=True)

    style_axes(args.dpi)
    tau0_ma = moving_average(tau0, args.ma_window)
    if tau0_ma.size == tau0.size:
        steps_ma = steps
    else:
        steps_ma = steps[args.ma_window - 1 :]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, tau0, alpha=0.4, linewidth=1.2, label="tau0")
    plt.plot(steps_ma, tau0_ma, linewidth=2.0, label=f"Rolling mean (w={args.ma_window})")
    plt.xlabel("Training steps")
    plt.ylabel("tau0")
    plt.title("tau0 vs Training Steps")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path_line = os.path.join(out_dir, "fig_tau0_vs_steps.png")
    plt.savefig(out_path_line, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(tau0, bins=40, color="#4c78a8", alpha=0.75)
    plt.xlabel("tau0")
    plt.ylabel("Count")
    plt.title("tau0 Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path_hist = os.path.join(out_dir, "fig_tau0_hist.png")
    plt.savefig(out_path_hist, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    print(f"Saved: {out_path_line}")
    print(f"Saved: {out_path_hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
