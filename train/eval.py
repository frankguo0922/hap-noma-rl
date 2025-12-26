"""Evaluate a trained model on HapWpcnNomaEnv."""
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import yaml
from stable_baselines3 import PPO, SAC

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(env: HapWpcnNomaEnv, model, seed: int) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    done = False
    sum_se = 0.0
    ee = 0.0
    tau0_list = []
    infeasible = 0
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        sum_se += float(info.get("sum_se", 0.0))
        ee += float(info.get("ee", 0.0))
        tau0_list.append(float(info.get("tau0", 0.0)))
        infeasible += int(info.get("infeasible_count", 0))
        total_reward += float(reward)
    return {
        "sum_se": sum_se,
        "ee": ee,
        "tau0_mean": float(np.mean(tau0_list)) if tau0_list else 0.0,
        "infeasible": float(infeasible),
        "reward": total_reward,
    }


def load_model(path: str, env: HapWpcnNomaEnv):
    if path.endswith("sac_hap_wpcn_noma.zip"):
        return SAC.load(path, env=env)
    if path.endswith("ppo_hap_wpcn_noma.zip"):
        return PPO.load(path, env=env)
    try:
        return SAC.load(path, env=env)
    except Exception:
        return PPO.load(path, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="models/sac_hap_wpcn_noma.zip")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg.get("env", {}))
    seed = int(cfg.get("seed", 42))

    env = HapWpcnNomaEnv(config=env_cfg, seed=seed)
    model = load_model(args.model, env)

    metrics = {"sum_se": [], "ee": [], "tau0_mean": [], "infeasible": [], "reward": []}
    for i in range(args.episodes):
        res = run_episode(env, model, seed + i)
        for key in metrics:
            metrics[key].append(res[key])

    avg_se = float(np.mean(metrics["sum_se"]))
    avg_ee = float(np.mean(metrics["ee"]))
    avg_tau0 = float(np.mean(metrics["tau0_mean"]))
    avg_infeasible = float(np.mean(metrics["infeasible"]))
    avg_reward = float(np.mean(metrics["reward"]))

    print(
        f"episodes={args.episodes} avg_ee={avg_ee:.4f} avg_sum_se={avg_se:.4f} "
        f"avg_tau0={avg_tau0:.4f} avg_infeasible={avg_infeasible:.2f} avg_reward={avg_reward:.4f}"
    )


if __name__ == "__main__":
    main()
