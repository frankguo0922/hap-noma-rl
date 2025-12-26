"""Plot training SINR behavior and its relation to EE."""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


NOTE = (
    "SINR does not monotonically increase during training, as the objective is energy efficiency "
    "maximization rather than throughput maximization. The learned policy converges to an operating "
    "SINR region that balances spectral efficiency and power consumption."
)


def find_latest_metrics_csv() -> str | None:
    runs_dir = os.path.join("logs", "runs")
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, "train_metrics.csv")
        if os.path.exists(path):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> int:
    csv_path = find_latest_metrics_csv()
    if csv_path is None:
        print("CSV not found: logs/runs/<run_id>/train_metrics.csv")
        return 1

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.size == 0:
        print(f"CSV is empty: {csv_path}")
        return 1

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        print(f"CSV has invalid format: {csv_path}")
        return 1

    steps = data[:, 0]
    ee_sum = data[:, 1]
    sinr = data[:, 2]
    log2_mean = data[:, 3]

    mask = np.isfinite(steps) & np.isfinite(ee_sum) & np.isfinite(sinr) & np.isfinite(log2_mean)
    steps = steps[mask]
    ee_sum = ee_sum[mask]
    sinr = sinr[mask]
    log2_mean = log2_mean[mask]

    order = np.argsort(steps)
    steps = steps[order]
    ee_sum = ee_sum[order]
    sinr = sinr[order]
    log2_mean = log2_mean[order]

    if steps.size == 0:
        print(f"No valid data points in: {csv_path}")
        return 1

    step_delta = float(np.median(np.diff(steps))) if steps.size > 1 else 1.0
    window = max(1, int(5000 / max(step_delta, 1.0)))
    sinr_ma = moving_average(sinr, window)
    steps_ma = steps[window - 1 :] if sinr_ma.size != sinr.size else steps

    os.makedirs("figures", exist_ok=True)

    # 1) SINR vs steps with MA
    plt.figure(figsize=(8, 5))
    plt.plot(steps, sinr, linewidth=1.5, label="avg_sinr_mean_tx")
    if sinr_ma.size > 1:
        plt.plot(steps_ma, sinr_ma, linewidth=2, label=f"MA{window}")
    plt.xlabel("Training Steps")
    plt.ylabel("avg_sinr_mean_tx")
    plt.title("Training SINR Behavior")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.figtext(0.5, -0.02, NOTE, ha="center", wrap=True, fontsize=9)
    plt.tight_layout()
    out1 = os.path.join("figures", "training_sinr_over_steps.png")
    plt.savefig(out1, dpi=200)
    plt.show()

    # 2) SINR vs EE scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(sinr, ee_sum, s=18, alpha=0.7)
    plt.xlabel("avg_sinr_mean_tx")
    plt.ylabel("avg_ee_sum")
    plt.title("SINR vs EE (Training)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.figtext(0.5, -0.02, NOTE, ha="center", wrap=True, fontsize=9)
    plt.tight_layout()
    out2 = os.path.join("figures", "sinr_vs_ee_scatter.png")
    plt.savefig(out2, dpi=200)
    plt.show()

    # 3) log2 mean vs steps
    plt.figure(figsize=(8, 5))
    plt.plot(steps, log2_mean, linewidth=2, label="avg_log2_mean_tx")
    plt.xlabel("Training Steps")
    plt.ylabel("avg_log2_mean_tx")
    plt.title("Spectral Efficiency Indicator over Training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.figtext(0.5, -0.02, NOTE, ha="center", wrap=True, fontsize=9)
    plt.tight_layout()
    out3 = os.path.join("figures", "log2_mean_over_steps.png")
    plt.savefig(out3, dpi=200)
    plt.show()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
