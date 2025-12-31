"""Plot EE vs number of WDs from eval_ee_vs_wd CSV."""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

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


def load_csv(path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")
        rows = list(reader)
    columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames}
    for row in rows:
        for name in reader.fieldnames:
            value = row.get(name, "")
            try:
                columns[name].append(float(value))
            except (TypeError, ValueError):
                columns[name].append(np.nan)
    return reader.fieldnames, {k: np.array(v, dtype=np.float64) for k, v in columns.items()}


def find_column(columns: List[str], include: List[str], exclude: List[str] | None = None) -> str | None:
    exclude = exclude or []
    for col in columns:
        key = col.lower()
        if all(token in key for token in include) and not any(token in key for token in exclude):
            return col
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=os.path.join("logs", "ee_vs_wd.csv"))
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return 1

    try:
        headers, data = load_csv(args.csv)
    except ValueError as exc:
        print(f"Failed to read CSV: {exc}")
        return 1

    n_wd_col = find_column(headers, ["n", "wd"]) or find_column(headers, ["wd"])
    ee_col = find_column(headers, ["ee", "mean"])
    ee_std_col = find_column(headers, ["ee", "std"])
    if n_wd_col is None or ee_col is None:
        print("Required columns not found in CSV.")
        print("Available columns:", ", ".join(headers))
        return 1

    n_wd = data[n_wd_col]
    ee = data[ee_col]
    ee_std = data[ee_std_col] if ee_std_col in data else None

    mask = np.isfinite(n_wd) & np.isfinite(ee)
    n_wd = n_wd[mask]
    ee = ee[mask]
    if ee_std is not None:
        ee_std = ee_std[mask]

    if n_wd.size == 0:
        print("No valid data points found in CSV.")
        return 1

    order = np.argsort(n_wd)
    n_wd = n_wd[order]
    ee = ee[order]
    if ee_std is not None:
        ee_std = ee_std[order]

    style_axes(args.dpi)
    plt.figure(figsize=(8, 5))
    if ee_std is not None and np.any(np.isfinite(ee_std)):
        plt.fill_between(n_wd, ee - ee_std, ee + ee_std, alpha=0.2, label="Std")
    plt.plot(n_wd, ee, marker="o", linewidth=2, label="EE (mean)")
    ymin = float(np.min(ee)) - 0.5
    ymax = float(np.max(ee)) + 0.5
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_autoscaley_on(False)
    plt.xlabel("Number of WDs")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.title("Scalability: EE vs Number of WDs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = args.out_path or os.path.join("figures", "ee_vs_wd.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
