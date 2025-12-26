"""Plot EE bar comparison between ICC 2025 baseline and SAC results.

Note: The EE value of ICC 2025 is an approximate representative value read from the reported figure.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt


def main() -> int:
    ee_baseline = 7.90
    wd_points = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    ee_sum_mean = [
        8.4171,
        8.1209,
        8.1023,
        8.1500,
        7.9733,
        7.8925,
        7.8931,
        7.7889,
        7.8456,
        7.7784,
    ]

    labels = ["ICC 2025 (Hybrid)"] + [f"Ours WD={wd}" for wd in wd_points]
    values = [ee_baseline] + ee_sum_mean
    colors = ["#0b3d91"] + ["#f28e2b"] * len(ee_sum_mean)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=colors, label="EE")
    plt.title("Energy Efficiency Comparison with ICC 2025")
    plt.ylabel("Energy Efficiency (bps/Hz/J)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "ee_bar_comparison_icc2025.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
