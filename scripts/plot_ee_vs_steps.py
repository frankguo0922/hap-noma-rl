"""Plot EE vs training steps from logged CSV."""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def find_latest_run_csv() -> str | None:
    runs_dir = os.path.join("logs", "runs")
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, "ee_vs_steps.csv")
        if os.path.exists(path):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to ee_vs_steps.csv")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = find_latest_run_csv()
    if csv_path is None:
        legacy_path = os.path.join("logs", "ee_vs_steps.csv")
        if os.path.exists(legacy_path):
            csv_path = legacy_path
        else:
            print("CSV not found: logs/runs/<run_id>/ee_vs_steps.csv")
            return 1
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=0)
    if data.size == 0:
        print(f"CSV is empty: {csv_path}")
        return 1

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        print(f"CSV has invalid format: {csv_path}")
        return 1
    steps = data[:, 0]
    ee = data[:, 1]
    ee_std = data[:, 2] if data.shape[1] >= 3 else None

    mask = np.isfinite(steps) & np.isfinite(ee)
    steps = steps[mask]
    ee = ee[mask]
    if ee_std is not None:
        ee_std = ee_std[mask]

    order = np.argsort(steps)
    steps = steps[order]
    ee = ee[order]
    if ee_std is not None:
        ee_std = ee_std[order]

    if steps.size == 0:
        print(f"No valid data points in: {csv_path}")
        return 1

    unique_steps, unique_idx = np.unique(steps, return_index=True)
    steps = unique_steps
    ee = ee[unique_idx]
    if ee_std is not None:
        ee_std = ee_std[unique_idx]

    if len(steps) <= 1:
        print(f"Not enough points in: {csv_path}")
        return 1

    style_axes(args.dpi)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, ee, marker="o", linewidth=2, markersize=6, label="EE (eval)")
    plt.title("EE Convergence (Eval)")
    plt.xlabel("Training steps")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.xlim(float(np.min(steps)), float(np.max(steps)))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    out_path = args.out_path or os.path.join("figures", "sac_eval_convergence_clean.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
