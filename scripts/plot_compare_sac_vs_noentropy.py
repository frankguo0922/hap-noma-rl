"""Compare EE convergence between SAC and no-entropy baseline."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import json


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def load_eval_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=0)
    if data.size == 0:
        raise ValueError(f"CSV is empty: {csv_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"CSV has invalid format: {csv_path}")
    steps = data[:, 0]
    ee = data[:, 1]
    mask = np.isfinite(steps) & np.isfinite(ee)
    steps = steps[mask]
    ee = ee[mask]
    if steps.size == 0:
        raise ValueError(f"No valid data points in: {csv_path}")
    order = np.argsort(steps)
    steps = steps[order]
    ee = ee[order]
    unique_steps, unique_idx = np.unique(steps, return_index=True)
    return unique_steps, ee[unique_idx]


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
    parser.add_argument("--sac-run-dir", type=str, required=True)
    parser.add_argument("--noentropy-run-dir", type=str, required=True)
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    sac_summary_path = os.path.join(args.sac_run_dir, "summary.json")
    noent_summary_path = os.path.join(args.noentropy_run_dir, "summary.json")
    if not os.path.exists(sac_summary_path) or not os.path.exists(noent_summary_path):
        print("summary.json not found in one or both run directories.")
        print("Run the exporter to generate summary.json before comparing.")
        return 1
    with open(sac_summary_path, "r", encoding="utf-8") as f:
        sac_summary = json.load(f)
    with open(noent_summary_path, "r", encoding="utf-8") as f:
        noent_summary = json.load(f)
    sac_type = sac_summary.get("run_type")
    noent_type = noent_summary.get("run_type")
    if {sac_type, noent_type} != {"sac_auto", "no_entropy"}:
        print(
            "Run type mismatch. Expected one sac_auto and one no_entropy. "
            f"Got sac_run={sac_type}, noentropy_run={noent_type}."
        )
        return 1

    sac_csv = os.path.join(args.sac_run_dir, "ee_vs_steps.csv")
    noent_csv = os.path.join(args.noentropy_run_dir, "ee_vs_steps.csv")
    if not os.path.exists(sac_csv):
        print(f"CSV not found: {sac_csv}")
        return 1
    if not os.path.exists(noent_csv):
        print(f"CSV not found: {noent_csv}")
        return 1

    try:
        sac_steps, sac_ee = load_eval_csv(sac_csv)
        base_steps, base_ee = load_eval_csv(noent_csv)
    except ValueError as exc:
        print(str(exc))
        return 1

    sac_ma = moving_average(sac_ee, args.ma_window)
    base_ma = moving_average(base_ee, args.ma_window)
    sac_steps_ma = sac_steps[-len(sac_ma) :] if sac_ma.size > 0 else sac_steps
    base_steps_ma = base_steps[-len(base_ma) :] if base_ma.size > 0 else base_steps
    if sac_steps_ma.size > 0 and base_steps_ma.size > 0:
        max_step = min(float(sac_steps_ma[-1]), float(base_steps_ma[-1]))
        sac_mask = sac_steps_ma <= max_step
        base_mask = base_steps_ma <= max_step
        sac_steps_ma = sac_steps_ma[sac_mask]
        sac_ma = sac_ma[sac_mask]
        base_steps_ma = base_steps_ma[base_mask]
        base_ma = base_ma[base_mask]

    mean_sac = float(np.nanmean(sac_ma)) if sac_ma.size > 0 else 0.0
    mean_base = float(np.nanmean(base_ma)) if base_ma.size > 0 else 0.0

    style_axes(args.dpi)
    plt.figure(figsize=(9, 5))
    plt.plot(
        sac_steps_ma,
        sac_ma,
        linewidth=2,
        label=f"SAC (entropy-enabled), mean={mean_sac:.2f}",
    )
    plt.plot(
        base_steps_ma,
        base_ma,
        linewidth=2,
        label=f"RL without entropy (ablation), mean={mean_base:.2f}",
    )
    plt.title("EE Convergence Comparison: SAC vs No-Entropy RL")
    plt.xlabel("Training steps")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()

    if args.out_path is None:
        run_id = datetime.now().strftime("compare_%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", "runs", run_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ee_compare_sac_vs_noentropy.png")
    else:
        out_path = args.out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
