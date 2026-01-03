"""Export eval results to summary JSON and plots."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def find_latest_run_dir() -> Optional[str]:
    runs_dir = os.path.join("logs", "runs")
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, name)
        csv_path = os.path.join(run_dir, "ee_vs_steps.csv")
        if os.path.exists(csv_path):
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


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


def load_avg_reward(metrics_path: str) -> Optional[float]:
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if len(rows) < 2:
        return None
    header = rows[0]
    last = rows[-1]
    if "avg_reward_raw" in header:
        idx = header.index("avg_reward_raw")
    elif "avg_reward_norm" in header:
        idx = header.index("avg_reward_norm")
    else:
        return None
    if idx >= len(last):
        return None
    try:
        return float(last[idx])
    except ValueError:
        return None


def load_sac_diagnostics(diag_path: str) -> dict:
    if not os.path.exists(diag_path):
        return {}
    with open(diag_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row]
    if not rows:
        return {}
    ent_vals = []
    ent_coef_final = None
    has_twin_critics = None
    for row in rows:
        if "policy_entropy" in row and row["policy_entropy"] != "":
            try:
                ent_vals.append(float(row["policy_entropy"]))
            except ValueError:
                pass
        if "ent_coef" in row and row["ent_coef"] != "":
            ent_coef_final = row["ent_coef"]
        if "has_twin_critics" in row and row["has_twin_critics"] != "":
            has_twin_critics = row["has_twin_critics"]
    avg_entropy = float(np.mean(ent_vals)) if ent_vals else None
    if ent_coef_final is not None:
        try:
            ent_coef_final = float(ent_coef_final)
        except ValueError:
            pass
    if has_twin_critics is not None:
        has_twin_critics = str(has_twin_critics).lower() in ("true", "1")
    return {
        "ent_coef_final": ent_coef_final,
        "avg_policy_entropy": avg_entropy,
        "has_twin_critics": has_twin_critics,
    }


def plot_eval(steps: np.ndarray, ee: np.ndarray, out_path: str, title: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    plt.figure(figsize=(8, 5))
    plt.plot(steps, ee, marker="o", linewidth=2, markersize=5)
    plt.title(title)
    plt.xlabel("Training steps")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--total-runtime-seconds", type=float, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    if run_dir is None:
        print("Run directory not found: logs/runs/<run_id>")
        return 1

    csv_path = os.path.join(run_dir, "ee_vs_steps.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    try:
        steps, ee = load_eval_csv(csv_path)
    except ValueError as exc:
        print(str(exc))
        return 1

    final_step = int(steps[-1])
    final_eval_ee_mean = float(ee[-1])
    best_idx = int(np.argmax(ee))
    best_eval_ee_mean = float(ee[best_idx])
    best_step = int(steps[best_idx])

    raw_path = os.path.join(run_dir, "ee_eval_raw.png")
    plot_eval(steps, ee, raw_path, "EE Convergence (Eval)")

    window = 20
    ee_ma = moving_average(ee, window)
    steps_ma = steps[-len(ee_ma) :] if ee_ma.size > 0 else steps
    ma_path = os.path.join(run_dir, "ee_eval_20ma.png")
    plot_eval(steps_ma, ee_ma, ma_path, "EE Convergence (Eval)")

    avg_reward = load_avg_reward(os.path.join(run_dir, "train_metrics.csv"))

    run_meta_path = os.path.join(run_dir, "run_meta.json")
    run_meta = {}
    if os.path.exists(run_meta_path):
        with open(run_meta_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)

    summary = {
        "run_dir": os.path.abspath(run_dir),
        "final_step": final_step,
        "final_eval_ee_mean": final_eval_ee_mean,
        "best_eval_ee_mean": best_eval_ee_mean,
        "best_step": best_step,
        "avg_reward": avg_reward,
        "total_runtime_seconds": args.total_runtime_seconds,
        "timestamp_exported": datetime.now().isoformat(timespec="seconds"),
        "eval_csv": os.path.abspath(csv_path),
        "plot_raw": os.path.abspath(raw_path),
        "plot_20ma": os.path.abspath(ma_path),
    }
    sac_diag_path = os.path.join(run_dir, "sac_diagnostics.csv")
    rl_trace_path = os.path.join(run_dir, "rl_trace.csv")
    if os.path.exists(sac_diag_path):
        summary["sac_diagnostics_csv"] = os.path.abspath(sac_diag_path)
    if os.path.exists(rl_trace_path):
        summary["rl_trace_csv"] = os.path.abspath(rl_trace_path)
    if run_meta:
        summary.update(run_meta)
    diag_summary = load_sac_diagnostics(os.path.join(run_dir, "sac_diagnostics.csv"))
    if diag_summary:
        summary.update(diag_summary)

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("Export finished")
    print(f"Run: {run_dir}")
    print(f"Final step={final_step}, Final eval EE={final_eval_ee_mean:.6f}")
    print(f"Best eval EE={best_eval_ee_mean:.6f} at step={best_step}")
    print("Saved: summary.json, ee_eval_raw.png, ee_eval_20ma.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
