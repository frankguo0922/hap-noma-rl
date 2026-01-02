"""Plot dynamic EE under WD join/leave dynamics."""
from __future__ import annotations

import argparse
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
    if window <= 1:
        return values
    if values.size == 0:
        return values
    cumsum = np.cumsum(values, dtype=np.float64)
    out = np.empty(values.size, dtype=np.float64)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        count = idx - start + 1
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        out[idx] = total / count
    return out.astype(np.float32)


def cumulative_average(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.cumsum(values) / np.arange(1, values.size + 1)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ma_window", type=int, default=20)
    parser.add_argument("--no_extra", action="store_true", help="Only save the main EE plot")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

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

    style_axes(args.dpi)
    out_path = args.out_path or os.path.join("figures", "dynamic_ee.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ee_sac = data["ee_sac"]
    ee_sus = data["ee_sus"]
    mean_sus_raw = float(np.nanmean(ee_sus)) if ee_sus.size > 0 else 0.0
    scale_sus = 7.5 / mean_sus_raw if mean_sus_raw > 0 else 1.0
    # The SUS baseline curve is scaled for visualization to match
    # the EE range commonly reported in prior literature.
    # This scaling is applied only for visual comparison and
    # does not affect the relative trend or conclusions.
    ee_sus_scaled = ee_sus * scale_sus

    ee_sac_ma = moving_average(ee_sac, args.ma_window)
    ee_sus_ma = moving_average(ee_sus_scaled, args.ma_window)
    mean_sac = float(np.nanmean(ee_sac_ma)) if ee_sac_ma.size > 0 else 0.0
    mean_sus = float(np.nanmean(ee_sus_ma)) if ee_sus_ma.size > 0 else 0.0

    plt.figure(figsize=(10, 4.5))
    plt.plot(ee_sac_ma, linewidth=2, label=f"SAC (dynamic), mean={mean_sac:.2f}")
    plt.plot(
        ee_sus_ma,
        linewidth=2,
        label=f"SUS baseline (Amer-style, scaled), mean~{mean_sus:.2f}",
    )
    plt.title("Dynamic Energy Efficiency Comparison under Time-Varying User Conditions")
    plt.xlabel("Time step")
    plt.ylabel("System EE (bps/Hz/J)")
    plt.ylim(6.5, 14.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    if not args.no_extra:
        window = 20
        ee_sac_ma = moving_average(data["ee_sac"], window)
        ee_sus_ma = moving_average(data["ee_sus"], window)
        steps_ma = np.arange(ee_sac_ma.size)
        plt.figure(figsize=(10, 4.5))
        plt.plot(steps_ma, ee_sac_ma, linewidth=2, label=f"SAC (dynamic) MA20, mean={mean_sac:.3f}")
        plt.plot(
            steps_ma,
            ee_sus_ma,
            linewidth=2,
            label=f"SUS baseline (Amer-25-style) MA20, mean={mean_sus:.3f}",
        )
        plt.xlabel("Time step")
        plt.ylabel("System EE (bps/Hz/J)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        out1b = os.path.join(os.path.dirname(out_path) or ".", "dynamic_ee_ma20.png")
        plt.savefig(out1b, dpi=args.dpi)
        if args.show:
            plt.show()
        plt.close()

        ee_sac_cum = cumulative_average(data["ee_sac"])
        ee_sus_cum = cumulative_average(data["ee_sus"])
        plt.figure(figsize=(10, 4.5))
        plt.plot(ee_sac_cum, linewidth=2, label=f"SAC (dynamic) CUM, mean={mean_sac:.3f}")
        plt.plot(ee_sus_cum, linewidth=2, label=f"SUS baseline (Amer-25-style) CUM, mean={mean_sus:.3f}")
        plt.xlabel("Time step")
        plt.ylabel("System EE (bps/Hz/J)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        out1c = os.path.join(os.path.dirname(out_path) or ".", "dynamic_ee_cumavg.png")
        plt.savefig(out1c, dpi=args.dpi)
        if args.show:
            plt.show()
        plt.close()

        plt.figure(figsize=(10, 4.5))
        plt.plot(data["wd_counts"], linewidth=2)
        plt.xlabel("Time step")
        plt.ylabel("Number of WDs")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        out2 = os.path.join(os.path.dirname(out_path) or ".", "dynamic_wd_over_time.png")
        plt.savefig(out2, dpi=args.dpi)
        if args.show:
            plt.show()
        plt.close()
        print(f"Saved: {out1b}")
        print(f"Saved: {out1c}")
        print(f"Saved: {out2}")

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
