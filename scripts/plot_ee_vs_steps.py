"""Plot EE vs training steps from logged CSV."""
from __future__ import annotations

import os
import sys

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


def main() -> int:
    csv_path = find_latest_run_csv()
    if csv_path is None:
        legacy_path = os.path.join("logs", "ee_vs_steps.csv")
        if os.path.exists(legacy_path):
            csv_path = legacy_path
        else:
            print("CSV not found: logs/runs/<run_id>/ee_vs_steps.csv")
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

    print("N points:", len(steps))
    print("steps:", steps)
    print("steps min/max:", float(np.min(steps)), float(np.max(steps)))
    print("ee head/tail:", ee[:3], ee[-3:])
    assert len(steps) == len(ee) and len(steps) > 3

    plt.figure(figsize=(8, 5))
    plt.plot(steps, ee, marker="o", linewidth=2, markersize=6, label="EE (eval)")
    plt.title("SAC Convergence (Eval EE)")
    plt.xlabel("Training Steps")
    plt.ylabel("System Energy Efficiency (bps/Hz/J)")
    plt.xlim(float(np.min(steps)), float(np.max(steps)))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(x / 1000)}k")
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0.0, 9.5)
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "sac_eval_convergence_clean.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
