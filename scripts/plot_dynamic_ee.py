"""Plot dynamic EE under WD join/leave dynamics."""
from __future__ import annotations

import copy
import os
from typing import Dict

import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config
from baselines.sus_baseline import sus_baseline_step


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(env: HapWpcnNomaEnv) -> SAC:
    model_paths = [
        os.path.join("models", "best", "best_model.zip"),
        os.path.join("models", "sac_hap_noma.zip"),
        os.path.join("models", "sac_hap_wpcn_noma.zip"),
    ]
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError("No SAC model found in models/.")
    print(f"Loaded SAC model: {model_path}")
    return SAC.load(model_path, env=env)


def clamp_users(env: HapWpcnNomaEnv, current_n: int) -> None:
    env.n_wd = current_n
    if current_n < env.max_wd:
        inactive_idx = np.arange(current_n, env.max_wd)
        env.active[inactive_idx] = False
        env.channel.distance[inactive_idx] = 0.0
        env.channel.aoa[inactive_idx] = 0.0
        env.channel.gain[inactive_idx] = 0.0
        env.energy[inactive_idx] = 0.0


def add_one_user(env: HapWpcnNomaEnv, current_n: int) -> int:
    if current_n >= env.max_wd:
        return current_n
    new_idx = current_n
    current_n += 1
    env.n_wd = current_n
    env.active[new_idx] = True
    env.channel.sample_positions(np.array([new_idx]))
    env.channel.sample_gain(np.array([new_idx]))
    env.energy[new_idx] = env.rng.uniform(0.2, env.cfg.e_max)
    return current_n


def remove_one_user(env: HapWpcnNomaEnv, current_n: int) -> int:
    if current_n <= 20:
        return current_n
    remove_idx = current_n - 1
    env.active[remove_idx] = False
    env.channel.distance[remove_idx] = 0.0
    env.channel.aoa[remove_idx] = 0.0
    env.channel.gain[remove_idx] = 0.0
    env.energy[remove_idx] = 0.0
    current_n -= 1
    env.n_wd = current_n
    return current_n


def run_episode(
    env_sac: HapWpcnNomaEnv,
    env_baseline: HapWpcnNomaEnv,
    model: SAC,
    steps: int = 500,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    obs_sac, _ = env_sac.reset(seed=seed)
    obs_base, _ = env_baseline.reset(seed=seed)
    current_n = 100
    clamp_users(env_sac, current_n)
    clamp_users(env_baseline, current_n)

    ee_sac = []
    ee_sus = []
    wd_counts = []
    rng = np.random.default_rng(seed)

    for t in range(steps):
        if t % 10 == 0:
            if rng.random() < 0.2:
                current_n = add_one_user(env_sac, current_n)
                current_n = add_one_user(env_baseline, current_n)
            if rng.random() < 0.2:
                current_n = remove_one_user(env_sac, current_n)
                current_n = remove_one_user(env_baseline, current_n)
            clamp_users(env_sac, current_n)
            clamp_users(env_baseline, current_n)

        action, _ = model.predict(obs_sac, deterministic=True)
        obs_sac, _, terminated, truncated, info = env_sac.step(action)
        ee_sac.append(float(info.get("ee_sum", 0.0)))
        wd_counts.append(current_n)

        sus_action, ee_est, _ = sus_baseline_step(env_baseline)
        obs_base, _, terminated_r, truncated_r, info_r = env_baseline.step(sus_action)
        ee_sus.append(float(info_r.get("ee_sum", ee_est)))

        if t < 5:
            print("SAC cfg use_solver:", env_sac.cfg.use_solver)
            print("SUS cfg use_solver:", env_baseline.cfg.use_solver)
            print("SAC info solver_used:", info.get("solver_used"), "EE:", info.get("ee_sum"))
            print("SUS info solver_used:", info_r.get("solver_used"), "EE:", info_r.get("ee_sum"))
            assert env_sac.cfg.use_solver is True
            if info.get("solver_used") is not True:
                raise RuntimeError("SAC solver_used is False during evaluation")
            if info_r.get("solver_used") is True:
                raise RuntimeError("solver_used=True in baseline rollout")

        done = terminated or truncated
        done_r = terminated_r or truncated_r
        if done or done_r:
            obs_sac, _ = env_sac.reset(seed=seed + t + 1)
            obs_base, _ = env_baseline.reset(seed=seed + t + 1)
            clamp_users(env_sac, current_n)
            clamp_users(env_baseline, current_n)

    return {
        "ee_sac": np.array(ee_sac, dtype=np.float32),
        "ee_sus": np.array(ee_sus, dtype=np.float32),
        "wd_counts": np.array(wd_counts, dtype=np.int32),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def cumulative_average(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.cumsum(values) / np.arange(1, values.size + 1)


def main() -> int:
    cfg = load_config("configs/default_paper_scale.yaml")
    base_env_cfg = build_env_config(cfg.get("env", {}))
    base_env_cfg.max_wd = 200
    base_env_cfg.n_wd = 100

    sac_cfg = copy.deepcopy(base_env_cfg)
    sus_cfg = copy.deepcopy(base_env_cfg)
    sac_cfg.use_solver = True
    sus_cfg.use_solver = False

    env_sac = HapWpcnNomaEnv(config=sac_cfg, seed=42)
    env_baseline = HapWpcnNomaEnv(config=sus_cfg, seed=42)
    model = load_model(env_sac)

    data = run_episode(env_sac, env_baseline, model, steps=500, seed=42)

    os.makedirs("figures", exist_ok=True)

    mean_sac = float(np.mean(data["ee_sac"])) if data["ee_sac"].size > 0 else 0.0
    mean_sus = float(np.mean(data["ee_sus"])) if data["ee_sus"].size > 0 else 0.0

    plt.figure(figsize=(10, 4.5))
    plt.plot(data["ee_sac"], linewidth=2, label=f"SAC (dynamic), mean={mean_sac:.3f}")
    plt.plot(data["ee_sus"], linewidth=2, label=f"SUS baseline (Amer-25-style), mean={mean_sus:.3f}")
    plt.xlabel("Time step")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join("figures", "dynamic_ee.png")
    plt.savefig(out1, dpi=200)
    plt.show()

    window = 20
    ee_sac_ma = moving_average(data["ee_sac"], window)
    ee_sus_ma = moving_average(data["ee_sus"], window)
    steps_ma = np.arange(ee_sac_ma.size)
    plt.figure(figsize=(10, 4.5))
    plt.plot(steps_ma, ee_sac_ma, linewidth=2, label=f"SAC (dynamic) MA20, mean={mean_sac:.3f}")
    plt.plot(steps_ma, ee_sus_ma, linewidth=2, label=f"SUS baseline (Amer-25-style) MA20, mean={mean_sus:.3f}")
    plt.xlabel("Time step")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out1b = os.path.join("figures", "dynamic_ee_ma20.png")
    plt.savefig(out1b, dpi=200)
    plt.show()

    ee_sac_cum = cumulative_average(data["ee_sac"])
    ee_sus_cum = cumulative_average(data["ee_sus"])
    plt.figure(figsize=(10, 4.5))
    plt.plot(ee_sac_cum, linewidth=2, label=f"SAC (dynamic) CUM, mean={mean_sac:.3f}")
    plt.plot(ee_sus_cum, linewidth=2, label=f"SUS baseline (Amer-25-style) CUM, mean={mean_sus:.3f}")
    plt.xlabel("Time step")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out1c = os.path.join("figures", "dynamic_ee_cumavg.png")
    plt.savefig(out1c, dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4.5))
    plt.plot(data["wd_counts"], linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("Number of WDs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out2 = os.path.join("figures", "dynamic_wd_over_time.png")
    plt.savefig(out2, dpi=200)
    plt.show()

    print(f"Saved: {out1}")
    print(f"Saved: {out1b}")
    print(f"Saved: {out1c}")
    print(f"Saved: {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
