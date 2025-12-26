"""Evaluate scalability of trained SAC policy vs number of WDs."""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config
from envs.wpcn import harvest_energy


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_action_from_components(tau0: float, beam: np.ndarray, power: np.ndarray, env: HapWpcnNomaEnv) -> np.ndarray:
    a_tau = 2.0 * (tau0 - env.cfg.tau_min) / (env.cfg.tau_max - env.cfg.tau_min) - 1.0
    a_tau = float(np.clip(a_tau, -1.0, 1.0))

    a_beam = np.full(env.max_wd, -0.5, dtype=np.float32)
    a_beam[beam == 1] = 0.0
    a_beam[beam == 2] = 0.5

    a_p = 2.0 * (power / env.cfg.p_max) - 1.0
    a_p = np.clip(a_p, -1.0, 1.0).astype(np.float32)

    action = np.concatenate(([a_tau], a_beam, a_p)).astype(np.float32)
    return action


def eval_policy(env: HapWpcnNomaEnv, policy: str, model: SAC | None, episodes: int, seed: int):
    metrics = {"ee": [], "sum_se": [], "total_power": [], "tau0": [], "n_tx_users": []}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if policy == "sac":
                action, _ = model.predict(obs, deterministic=True)
            elif policy == "random":
                action = env.action_space.sample()
            else:
                tau0 = 0.2
                tau1 = 1.0 - tau0
                beam = np.zeros(env.max_wd, dtype=np.int64)
                active = env.active.copy()
                beam[active & (env.channel.aoa < 0.0)] = 1
                beam[active & (env.channel.aoa >= 0.0)] = 2

                harvested = harvest_energy(env.channel.gain, active, tau0, env.cfg.p_wet, env.cfg.eta)
                available = env.energy + harvested
                power = np.zeros(env.max_wd, dtype=np.float32)
                if tau1 > 0.0:
                    max_power = np.minimum(available / tau1, env.cfg.p_max)
                    power = max_power.astype(np.float32)
                power[~active] = 0.0
                power[beam == 0] = 0.0
                action = make_action_from_components(tau0, beam, power, env)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            metrics["ee"].append(float(info.get("ee_sum", 0.0)))
            metrics["sum_se"].append(float(info.get("sum_se", 0.0)))
            metrics["total_power"].append(float(info.get("total_power", 0.0)))
            metrics["tau0"].append(float(info.get("tau0", 0.0)))
            metrics["n_tx_users"].append(float(info.get("n_tx_users", 0.0)))

    def agg(values: List[float]) -> Tuple[float, float]:
        return float(np.mean(values)), float(np.std(values))

    return {
        "ee": agg(metrics["ee"]),
        "sum_se": agg(metrics["sum_se"]),
        "total_power": agg(metrics["total_power"]),
        "tau0": agg(metrics["tau0"]),
        "n_tx_users": agg(metrics["n_tx_users"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_paper_scale.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg.get("env", {}))

    model_paths = [os.path.join("models", "best", "best_model.zip")]
    runs_dir = os.path.join("logs", "runs")
    if os.path.isdir(runs_dir):
        run_candidates = []
        for name in os.listdir(runs_dir):
            candidate = os.path.join(runs_dir, name, "best_model.zip")
            if os.path.exists(candidate):
                run_candidates.append(candidate)
        if run_candidates:
            model_paths.append(max(run_candidates, key=os.path.getmtime))
    model_paths.append(os.path.join("models", "sac_hap_wpcn_noma.zip"))

    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            "No SAC model found: models/best/best_model.zip, logs/runs/<latest>/best_model.zip, "
            "or models/sac_hap_wpcn_noma.zip"
        )
    print(f"Loaded SAC model: {model_path}")

    model = SAC.load(model_path)
    action_dim = int(model.action_space.shape[0])
    max_wd = (action_dim - 1) // 2
    if max_wd <= 0:
        print("Invalid action space shape in model.")
        return 1
    env_cfg.max_wd = max_wd

    n_eval_episodes = int(cfg.get("eval", {}).get("n_eval_episodes", 30))
    seed = int(cfg.get("seed", 42))

    n_wd_list = [n for n in range(20, 201, 20) if n <= max_wd]
    if not n_wd_list:
        print(f"Model max_wd={max_wd} is smaller than 20; no valid points to evaluate.")
        return 1

    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "ee_vs_wd.csv")

    rows = []
    wd_points: List[int] = []
    ee_mean_sac: List[float] = []
    ee_std_sac: List[float] = []
    for n_wd in n_wd_list:
        env_cfg.n_wd = n_wd
        env = HapWpcnNomaEnv(config=env_cfg, seed=seed)
        model = SAC.load(model_path, env=env)

        sac_stats = eval_policy(env, "sac", model, n_eval_episodes, seed)

        ee_mean_val = float(sac_stats["ee"][0])
        ee_std_val = float(sac_stats["ee"][1])
        if not np.isfinite(ee_mean_val) or not np.isfinite(ee_std_val):
            print(f"Invalid EE at n_wd={n_wd}")
            print("ee_list[:5]:", sac_stats["ee"][0:1])
            print(
                "total_power mean:", sac_stats["total_power"][0],
                "sum_se mean:", sac_stats["sum_se"][0],
                "n_tx_users mean:", sac_stats["n_tx_users"][0],
            )
            env.close()
            continue

        wd_points.append(n_wd)
        ee_mean_sac.append(ee_mean_val)
        ee_std_sac.append(ee_std_val)

        rows.append(
            {
                "n_wd": n_wd,
                "ee_mean_sac": ee_mean_val,
                "ee_std_sac": ee_std_val,
            }
        )
        env.close()
        print(
            f"n_wd={n_wd} ee_sum_mean={ee_mean_val:.4f} "
            f"total_power_mean={sac_stats['total_power'][0]:.4f} sum_se_mean={sac_stats['sum_se'][0]:.4f}"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n_wd", "ee_mean_sac", "ee_std_sac"])
        writer.writeheader()
        writer.writerows(rows)

    print("wd_points:", wd_points)
    print("ee_mean_sac:", ee_mean_sac[:10], "len=", len(ee_mean_sac))
    assert len(wd_points) == len(ee_mean_sac) and len(wd_points) > 0

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(wd_points, ee_mean_sac, marker="o", linewidth=2, label="SAC")
    plt.title("Energy efficiency vs number of WDs")
    plt.xlabel("Number of WDs")
    plt.ylabel("Energy Efficiency (bps/Hz/J)")
    plt.xticks(n_wd_list)
    plt.xlim(20, 200)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("figures", "ee_vs_wd.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {csv_path}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
