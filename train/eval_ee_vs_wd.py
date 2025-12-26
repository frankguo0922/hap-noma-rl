"""Evaluate EE vs number of WDs using a trained SAC policy."""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_eval(env: HapWpcnNomaEnv, model: SAC, episodes: int, seed: int) -> Dict[str, float]:
    ee_ep = []
    se_ep = []
    power_ep = []
    tau0_ep = []
    n_tx_ep = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ee_steps = []
        se_steps = []
        power_steps = []
        tau0_steps = []
        n_tx_steps = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ee_steps.append(float(info.get("ee_sum", 0.0)))
            se_steps.append(float(info.get("sum_se", 0.0)))
            power_steps.append(float(info.get("total_power", 0.0)))
            tau0_steps.append(float(info.get("tau0", 0.0)))
            n_tx_steps.append(float(info.get("n_tx_users", 0.0)))

        ee_ep.append(float(np.mean(ee_steps)) if ee_steps else 0.0)
        se_ep.append(float(np.mean(se_steps)) if se_steps else 0.0)
        power_ep.append(float(np.mean(power_steps)) if power_steps else 0.0)
        tau0_ep.append(float(np.mean(tau0_steps)) if tau0_steps else 0.0)
        n_tx_ep.append(float(np.mean(n_tx_steps)) if n_tx_steps else 0.0)

    return {
        "ee_sum_mean": float(np.mean(ee_ep)),
        "ee_sum_std": float(np.std(ee_ep)),
        "sum_se_mean": float(np.mean(se_ep)),
        "total_power_mean": float(np.mean(power_ep)),
        "tau0_mean": float(np.mean(tau0_ep)),
        "n_tx_users_mean": float(np.mean(n_tx_ep)),
    }


def main() -> int:
    cfg = load_config("configs/default_paper_scale.yaml")
    env_cfg = build_env_config(cfg.get("env", {}))

    max_wd = 200
    env_cfg.max_wd = max_wd

    wd_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    n_eval_episodes = 30
    seed = int(cfg.get("seed", 42))

    model_paths = [
        os.path.join("models", "best", "best_model.zip"),
        os.path.join("models", "sac_hap_noma.zip"),
        os.path.join("models", "sac_hap_wpcn_noma.zip"),
    ]
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            "No SAC model found: models/best/best_model.zip, models/sac_hap_noma.zip, "
            "or models/sac_hap_wpcn_noma.zip"
        )
    print(f"Loaded SAC model: {model_path}")

    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "ee_vs_wd.csv")

    rows: List[Dict[str, float]] = []
    ee_mean_list = []
    ee_std_list = []

    for n_wd in wd_list:
        env_cfg.n_wd = n_wd
        env = HapWpcnNomaEnv(config=env_cfg, seed=seed)
        model = SAC.load(model_path, env=env)
        stats = run_eval(env, model, n_eval_episodes, seed)
        env.close()

        rows.append(
            {
                "n_wd": n_wd,
                "ee_sum_mean": stats["ee_sum_mean"],
                "ee_sum_std": stats["ee_sum_std"],
                "sum_se_mean": stats["sum_se_mean"],
                "total_power_mean": stats["total_power_mean"],
                "tau0_mean": stats["tau0_mean"],
            }
        )
        ee_mean_list.append(stats["ee_sum_mean"])
        ee_std_list.append(stats["ee_sum_std"])
        print(
            f"n_wd={n_wd} ee_sum_mean={stats['ee_sum_mean']:.4f} "
            f"ee_sum_std={stats['ee_sum_std']:.4f} sum_se_mean={stats['sum_se_mean']:.4f} "
            f"total_power_mean={stats['total_power_mean']:.4f} tau0_mean={stats['tau0_mean']:.4f} "
            f"n_tx_users_mean={stats['n_tx_users_mean']:.2f}"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_wd", "ee_sum_mean", "ee_sum_std", "sum_se_mean", "total_power_mean", "tau0_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(wd_list, ee_mean_list, marker="o", linewidth=2)
    plt.title("System Energy Efficiency vs Number of WDs (1 HAP, 2-Antenna, SAC)")
    plt.xlabel("Number of WDs")
    plt.ylabel("System Energy Efficiency (bps/Hz/J)")
    plt.xticks(wd_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join("figures", "ee_vs_wd.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved: {csv_path}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
