"""Plot reward, EE, and sum-SE vs steps."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_path", type=str, default=None)
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
    step_col = find_column(columns, ["step"]) or find_column(columns, ["steps"])
    reward_col = find_column(columns, ["reward"])
    ee_col = find_column(columns, ["ee"], ["std"])
    se_col = find_column(columns, ["sum", "se"]) or find_column(columns, ["sum_se"])

    if step_col is None or reward_col is None or ee_col is None or se_col is None:
        print("Required columns not found for reward/EE/SE plot.")
        print("Available columns:", ", ".join(columns))
        return 1

    steps = data[step_col]
    reward = data[reward_col]
    ee = data[ee_col]
    se = data[se_col]

    mask = np.isfinite(steps) & np.isfinite(reward) & np.isfinite(ee) & np.isfinite(se)
    steps = steps[mask]
    reward = reward[mask]
    ee = ee[mask]
    se = se[mask]
    if steps.size == 0:
        print("No valid data points found for reward/EE/SE plot.")
        return 1

    order = np.argsort(steps)
    steps = steps[order]
    reward = reward[order]
    ee = ee[order]
    se = se[order]

    style_axes(args.dpi)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, reward, linewidth=2, label="Reward", color="#4c78a8")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Reward", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")

    ax2 = ax1.twinx()
    ax2.plot(steps, ee, linewidth=2, label="EE", color="#f58518")
    ax2.plot(steps, se, linewidth=2, label="Sum-SE", color="#54a24b")
    ax2.set_ylabel("EE / Sum-SE", color="#000000")
    ax2.tick_params(axis="y")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.title("Reward vs EE vs Sum-SE")
    plt.tight_layout()

    out_path = args.out_path or os.path.join("figures", "reward_ee_se.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
