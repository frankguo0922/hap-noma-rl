"""Calibrate EE scale by sweeping key parameters."""
from __future__ import annotations

import argparse
import csv
import os
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import yaml
from stable_baselines3 import SAC

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config
from envs.wpcn import harvest_energy


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_policy(env: HapWpcnNomaEnv, model: SAC | None, episodes: int, seed: int) -> Dict[str, float]:
    ee_list = []
    sum_se_list = []
    total_power_list = []
    sinr_list = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # heuristic: tau0=0.2, beam by AoA, max feasible power
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

                a_tau = 2.0 * (tau0 - env.cfg.tau_min) / (env.cfg.tau_max - env.cfg.tau_min) - 1.0
                a_tau = float(np.clip(a_tau, -1.0, 1.0))
                a_beam = np.full(env.max_wd, -0.5, dtype=np.float32)
                a_beam[beam == 1] = 0.0
                a_beam[beam == 2] = 0.5
                a_p = 2.0 * (power / env.cfg.p_max) - 1.0
                a_p = np.clip(a_p, -1.0, 1.0).astype(np.float32)
                action = np.concatenate(([a_tau], a_beam, a_p)).astype(np.float32)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ee_list.append(float(info.get("ee_sum", 0.0)))
            sum_se_list.append(float(info.get("sum_se", 0.0)))
            total_power_list.append(float(info.get("total_power", 0.0)))
            sinr_list.append(float(info.get("sinr_mean_tx", 0.0)))

    return {
        "ee_sum_mean": float(np.mean(ee_list)),
        "sum_se_mean": float(np.mean(sum_se_list)),
        "total_power_mean": float(np.mean(total_power_list)),
        "sinr_mean_tx": float(np.mean(sinr_list)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip")
    parser.add_argument("--wd", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg.get("env", {}))
    env_cfg.n_wd = args.wd
    env_cfg.max_wd = max(env_cfg.max_wd, env_cfg.n_wd)

    model_path = args.model if os.path.exists(args.model) else None
    if model_path:
        print(f"Loaded SAC model: {model_path}")
    else:
        print("Model not found, using heuristic policy.")

    pmax_list = [0.2, 0.3, 0.5, 0.8, 1.0]
    pc_list = [0.05, 0.1, 0.2, 0.5]
    pwet_list = [0.5, 1.0, 2.0]
    gain_scale_list = [0.5, 1.0, 2.0, 5.0]

    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "calibration_results.csv")
    rows: List[Dict[str, float]] = []

    for pmax, pc, pwet, gain_scale in product(pmax_list, pc_list, pwet_list, gain_scale_list):
        env_cfg.p_max = pmax
        env_cfg.p_c = pc
        env_cfg.p_wet = pwet
        env_cfg.channel.gain_scale = gain_scale

        env = HapWpcnNomaEnv(config=env_cfg, seed=int(cfg.get("seed", 42)))
        model = SAC.load(model_path, env=env) if model_path else None
        stats = eval_policy(env, model, args.episodes, int(cfg.get("seed", 42)))
        env.close()

        rows.append(
            {
                "Pmax": pmax,
                "Pc": pc,
                "Pwet": pwet,
                "gain_scale": gain_scale,
                "ee_sum_mean": stats["ee_sum_mean"],
                "sum_se_mean": stats["sum_se_mean"],
                "total_power_mean": stats["total_power_mean"],
                "sinr_mean_tx": stats["sinr_mean_tx"],
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Pmax",
                "Pc",
                "Pwet",
                "gain_scale",
                "ee_sum_mean",
                "sum_se_mean",
                "total_power_mean",
                "sinr_mean_tx",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rows_sorted = sorted(rows, key=lambda r: abs(r["ee_sum_mean"] - 8.0))
    print("Top 5 configs (EE closest to 8):")
    for row in rows_sorted[:5]:
        print(row)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
