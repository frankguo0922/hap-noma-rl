"""Generate paper-ready figures into figures/generated."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional


def find_latest_run_id(runs_dir: str) -> Optional[str]:
    if not os.path.isdir(runs_dir):
        return None
    candidates: List[tuple[float, str]] = []
    for name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metric_path = os.path.join(run_dir, "train_metrics.csv")
        eval_path = os.path.join(run_dir, "ee_vs_steps.csv")
        timestamps = []
        if os.path.exists(metric_path):
            timestamps.append(os.path.getmtime(metric_path))
        if os.path.exists(eval_path):
            timestamps.append(os.path.getmtime(eval_path))
        if not timestamps:
            timestamps.append(os.path.getmtime(run_dir))
        candidates.append((max(timestamps), name))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def run_cmd(args: List[str]) -> int:
    result = subprocess.run(args, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=os.path.join("figures", "generated"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    runs_dir = os.path.join("logs", "runs")
    run_id = args.run_id or find_latest_run_id(runs_dir)
    if run_id is None:
        print("No runs found in logs/runs.")
        return 1

    run_dir = os.path.join(runs_dir, run_id)
    train_csv = os.path.join(run_dir, "train_metrics.csv")
    ee_vs_steps_csv = os.path.join(run_dir, "ee_vs_steps.csv")

    if not os.path.exists(train_csv):
        print(f"Training metrics not found: {train_csv}")
        return 1

    if not os.path.exists(ee_vs_steps_csv):
        print(f"Eval EE curve not found: {ee_vs_steps_csv}")
        return 1

    python = sys.executable

    tasks = [
        (
            [python, "-m", "scripts.plot_ee_vs_steps", "--csv", ee_vs_steps_csv, "--out_path",
             os.path.join(out_dir, "fig_ee_vs_steps.png"), "--dpi", str(args.dpi)],
            os.path.join(out_dir, "fig_ee_vs_steps.png"),
        ),
        (
            [
                python,
                "-m",
                "train.eval_ee_vs_wd",
                "--out_path",
                os.path.join(out_dir, "fig_ee_vs_wd.png"),
                "--dpi",
                str(args.dpi),
            ],
            os.path.join(out_dir, "fig_ee_vs_wd.png"),
        ),
        (
            [python, "-m", "scripts.plot_dynamic_ee", "--out_path", os.path.join(out_dir, "fig_dynamic_ee.png"),
             "--dpi", str(args.dpi), "--no_extra"],
            os.path.join(out_dir, "fig_dynamic_ee.png"),
        ),
        (
            [python, "-m", "scripts.plot_sinr_hist", "--csv", train_csv,
             "--out_path", os.path.join(out_dir, "fig_sinr_hist.png"), "--dpi", str(args.dpi)],
            os.path.join(out_dir, "fig_sinr_hist.png"),
        ),
        (
            [python, "-m", "scripts.plot_tau0_training", "--csv", train_csv, "--out_dir", out_dir,
             "--dpi", str(args.dpi)],
            os.path.join(out_dir, "fig_tau0_vs_steps.png"),
        ),
        (
            [python, "-m", "scripts.plot_power_breakdown", "--csv", train_csv, "--out_dir", out_dir,
             "--dpi", str(args.dpi)],
            os.path.join(out_dir, "fig_power_components_vs_steps.png"),
        ),
        (
            [python, "-m", "scripts.plot_reward_vs_ee_se", "--csv", train_csv,
             "--out_path", os.path.join(out_dir, "fig_reward_ee_se.png"), "--dpi", str(args.dpi)],
            os.path.join(out_dir, "fig_reward_ee_se.png"),
        ),
    ]

    for cmd, expected in tasks:
        ret = run_cmd(cmd)
        if ret != 0:
            print(f"Command failed ({ret}): {' '.join(cmd)}")
            return ret
        if expected:
            print(f"Generated: {expected}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
