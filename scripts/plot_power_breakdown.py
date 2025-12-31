"""Plot power components and breakdown ratio from training metrics."""
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


def select_power_columns(columns: List[str]) -> Tuple[str | None, str | None, str | None, str | None]:
    total = find_column(columns, ["total", "power"]) or find_column(columns, ["power", "total"])
    wet = find_column(columns, ["power", "wet"], ["total"]) or find_column(columns, ["wet"], ["total"])
    ul = find_column(columns, ["power", "ul"], ["total"]) or find_column(columns, ["ul"], ["total"])
    circuit = (
        find_column(columns, ["power", "circuit"])
        or find_column(columns, ["power", "c"])
        or find_column(columns, ["circuit"])
    )
    return total, wet, ul, circuit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
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
    total_col, wet_col, ul_col, circuit_col = select_power_columns(columns)

    if step_col is None or wet_col is None or ul_col is None or circuit_col is None:
        print("Required power columns not found.")
        print("Available columns:", ", ".join(columns))
        return 1

    steps = data[step_col]
    wet = data[wet_col]
    ul = data[ul_col]
    circuit = data[circuit_col]
    total = data[total_col] if total_col is not None else wet + ul + circuit

    mask = np.isfinite(steps) & np.isfinite(wet) & np.isfinite(ul) & np.isfinite(circuit)
    steps = steps[mask]
    wet = wet[mask]
    ul = ul[mask]
    circuit = circuit[mask]
    total = total[mask]
    if steps.size == 0:
        print("No valid power data points found.")
        return 1

    order = np.argsort(steps)
    steps = steps[order]
    wet = wet[order]
    ul = ul[order]
    circuit = circuit[order]
    total = total[order]

    out_dir = args.out_dir or os.path.join("figures", "generated")
    os.makedirs(out_dir, exist_ok=True)

    style_axes(args.dpi)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, wet, linewidth=2, label="WET power")
    plt.plot(steps, ul, linewidth=2, label="UL power")
    plt.plot(steps, circuit, linewidth=2, label="Circuit power")
    if total_col is not None:
        plt.plot(steps, total, linewidth=2, linestyle="--", label="Total power")
    plt.xlabel("Training steps")
    plt.ylabel("Power")
    plt.title("Power Components vs Training Steps")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path_lines = os.path.join(out_dir, "fig_power_components_vs_steps.png")
    plt.savefig(out_path_lines, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    denom = wet + ul + circuit
    denom = np.where(denom == 0.0, np.nan, denom)
    wet_ratio = wet / denom
    ul_ratio = ul / denom
    circuit_ratio = circuit / denom

    plt.figure(figsize=(8, 5))
    plt.stackplot(
        steps,
        wet_ratio,
        ul_ratio,
        circuit_ratio,
        labels=["WET", "UL", "Circuit"],
        alpha=0.8,
    )
    plt.xlabel("Training steps")
    plt.ylabel("Power ratio")
    plt.title("Power Breakdown Ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_path_ratio = os.path.join(out_dir, "fig_power_breakdown_ratio.png")
    plt.savefig(out_path_ratio, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    print(f"Saved: {out_path_lines}")
    print(f"Saved: {out_path_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
