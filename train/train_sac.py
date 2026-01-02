"""Train SAC on HapWpcnNomaEnv."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from collections import deque
from typing import Dict, List

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config


class MetricsCallback(BaseCallback):
    def __init__(
        self,
        log_interval: int,
        eval_freq: int,
        log_path: str,
        metrics_path: str,
        best_model_path: str,
        eval_env: HapWpcnNomaEnv,
        n_eval_episodes: int,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.metrics_path = metrics_path
        self.best_model_path = best_model_path
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self._last_log_step = 0
        self._metrics_header_written = False
        self._best_eval = -np.inf
        self.rewards = deque(maxlen=log_interval)
        self.ee = deque(maxlen=log_interval)
        self.ee_sum = deque(maxlen=log_interval)
        self.ee_avg = deque(maxlen=log_interval)
        self.sum_se = deque(maxlen=log_interval)
        self.tau0 = deque(maxlen=log_interval)
        self.sinr_mean = deque(maxlen=log_interval)
        self.log2_mean = deque(maxlen=log_interval)
        self.se_per_tx = deque(maxlen=log_interval)
        self.sinr_db_mean = deque(maxlen=log_interval)
        self.sinr_db_vals = []
        self.p_tx_mean = deque(maxlen=log_interval)
        self.h_gain_mean = deque(maxlen=log_interval)
        self.noise_power = deque(maxlen=log_interval)
        self.rx_signal_mean = deque(maxlen=log_interval)
        self.interference_mean = deque(maxlen=log_interval)
        self.n_tx_users = deque(maxlen=log_interval)
        self.total_power = deque(maxlen=log_interval)
        self.reward_raw = deque(maxlen=log_interval)
        self.reward_norm = deque(maxlen=log_interval)
        self.power_wet = deque(maxlen=log_interval)
        self.power_ul = deque(maxlen=log_interval)
        self.power_c = deque(maxlen=log_interval)
        self.tau0_sel = deque(maxlen=log_interval)
        self.r1 = deque(maxlen=log_interval)
        self.r2 = deque(maxlen=log_interval)
        self.k_val = deque(maxlen=log_interval)
        self.m_val = deque(maxlen=log_interval)
        self._initial_eval_done = False
        self._milestones_hit = set()
        self._milestones = [240000, 300000, 400000, 500000]

    def _on_step(self) -> bool:
        infos: List[Dict] = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self.rewards.extend(rewards.tolist())
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "ee" in info:
                self.ee.append(info["ee"])
            if "ee_sum" in info:
                self.ee_sum.append(info["ee_sum"])
            if "ee_avg" in info:
                self.ee_avg.append(info["ee_avg"])
            if "sum_se" in info:
                self.sum_se.append(info["sum_se"])
            if "tau0" in info:
                self.tau0.append(info["tau0"])
            if "sinr_mean_tx" in info:
                self.sinr_mean.append(info["sinr_mean_tx"])
            if "log2_mean_tx" in info:
                self.log2_mean.append(info["log2_mean_tx"])
            if "sinr_db_mean_tx" in info:
                self.sinr_db_mean.append(info["sinr_db_mean_tx"])
            if "sinr_db_tx_list" in info:
                self.sinr_db_vals.extend(info["sinr_db_tx_list"])
            if "se_per_tx_user" in info:
                self.se_per_tx.append(info["se_per_tx_user"])
            if "p_tx_mean" in info:
                self.p_tx_mean.append(info["p_tx_mean"])
            if "h_gain_mean" in info:
                self.h_gain_mean.append(info["h_gain_mean"])
            if "noise_power_used" in info:
                self.noise_power.append(info["noise_power_used"])
            if "rx_signal_power_mean" in info:
                self.rx_signal_mean.append(info["rx_signal_power_mean"])
            if "interferences_mean" in info:
                self.interference_mean.append(info["interferences_mean"])
            if "n_tx_users" in info:
                self.n_tx_users.append(info["n_tx_users"])
            if "total_power" in info:
                self.total_power.append(info["total_power"])
            if "reward_raw" in info:
                self.reward_raw.append(info["reward_raw"])
            if "reward_norm" in info:
                self.reward_norm.append(info["reward_norm"])
            if "power_wet" in info:
                self.power_wet.append(info["power_wet"])
            if "power_ul" in info:
                self.power_ul.append(info["power_ul"])
            if "power_c" in info:
                self.power_c.append(info["power_c"])
            if "tau0_selected" in info:
                self.tau0_sel.append(info["tau0_selected"])
            if "r1" in info:
                self.r1.append(info["r1"])
            if "r2" in info:
                self.r2.append(info["r2"])
            if "K" in info:
                self.k_val.append(info["K"])
            if "M" in info:
                self.m_val.append(info["M"])

        for milestone in self._milestones:
            if self.num_timesteps >= milestone and milestone not in self._milestones_hit:
                print(f"Reached milestone: step={milestone}")
                self._milestones_hit.add(milestone)

        if self.n_calls % self.log_interval == 0 and self.n_calls > 0:
            avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
            avg_ee = float(np.mean(self.ee)) if self.ee else 0.0
            avg_ee_sum = float(np.mean(self.ee_sum)) if self.ee_sum else 0.0
            avg_ee_avg = float(np.mean(self.ee_avg)) if self.ee_avg else 0.0
            avg_se = float(np.mean(self.sum_se)) if self.sum_se else 0.0
            avg_tau0 = float(np.mean(self.tau0)) if self.tau0 else 0.0
            avg_sinr = float(np.mean(self.sinr_mean)) if self.sinr_mean else 0.0
            avg_log2 = float(np.mean(self.log2_mean)) if self.log2_mean else 0.0
            avg_se_per_tx = float(np.mean(self.se_per_tx)) if self.se_per_tx else 0.0
            avg_sinr_db = float(np.mean(self.sinr_db_mean)) if self.sinr_db_mean else 0.0
            if self.sinr_db_vals:
                sinr_db_p10 = float(np.percentile(self.sinr_db_vals, 10))
                sinr_db_p50 = float(np.percentile(self.sinr_db_vals, 50))
                sinr_db_p90 = float(np.percentile(self.sinr_db_vals, 90))
                sinr_db_min = float(np.min(self.sinr_db_vals))
                sinr_db_max = float(np.max(self.sinr_db_vals))
            else:
                sinr_db_p10 = sinr_db_p50 = sinr_db_p90 = 0.0
                sinr_db_min = sinr_db_max = 0.0
            avg_p_tx = float(np.mean(self.p_tx_mean)) if self.p_tx_mean else 0.0
            avg_h_gain = float(np.mean(self.h_gain_mean)) if self.h_gain_mean else 0.0
            avg_noise = float(np.mean(self.noise_power)) if self.noise_power else 0.0
            avg_rx_signal = float(np.mean(self.rx_signal_mean)) if self.rx_signal_mean else 0.0
            avg_interf = float(np.mean(self.interference_mean)) if self.interference_mean else 0.0
            avg_n_tx = float(np.mean(self.n_tx_users)) if self.n_tx_users else 0.0
            avg_total_power = float(np.mean(self.total_power)) if self.total_power else 0.0
            avg_reward_raw = float(np.mean(self.reward_raw)) if self.reward_raw else 0.0
            avg_reward_norm = float(np.mean(self.reward_norm)) if self.reward_norm else 0.0
            avg_power_wet = float(np.mean(self.power_wet)) if self.power_wet else 0.0
            avg_power_ul = float(np.mean(self.power_ul)) if self.power_ul else 0.0
            avg_power_c = float(np.mean(self.power_c)) if self.power_c else 0.0
            avg_tau0_sel = float(np.mean(self.tau0_sel)) if self.tau0_sel else 0.0
            avg_r1 = float(np.mean(self.r1)) if self.r1 else 0.0
            avg_r2 = float(np.mean(self.r2)) if self.r2 else 0.0
            avg_k = float(np.mean(self.k_val)) if self.k_val else 0.0
            avg_m = float(np.mean(self.m_val)) if self.m_val else 0.0
            print(
                f"step={self.num_timesteps} avg_reward={avg_reward:.4f} avg_reward_raw={avg_reward_raw:.4f} "
                f"avg_reward_norm={avg_reward_norm:.4f} "
                f"avg_ee_sum={avg_ee_sum:.4f} avg_ee_avg={avg_ee_avg:.4f} "
                f"avg_sum_se={avg_se:.4f} avg_n_tx_users={avg_n_tx:.2f} avg_total_power={avg_total_power:.4f} "
                f"avg_power_wet={avg_power_wet:.4f} avg_power_ul={avg_power_ul:.4f} avg_power_c={avg_power_c:.4f} "
                f"avg_ee={avg_ee:.4f} avg_tau0={avg_tau0:.4f} "
                f"avg_tau0_selected={avg_tau0_sel:.4f} avg_r1={avg_r1:.4f} avg_r2={avg_r2:.4f} "
                f"avg_K={avg_k:.1f} avg_M={avg_m:.1f} "
                f"avg_sinr_mean_tx={avg_sinr:.4f} avg_sinr_db_mean_tx={avg_sinr_db:.4f} "
                f"sinr_db_p50={sinr_db_p50:.2f} sinr_db_p90={sinr_db_p90:.2f} avg_log2_mean_tx={avg_log2:.4f} "
                f"avg_se_per_tx_user={avg_se_per_tx:.4f} avg_p_tx_mean={avg_p_tx:.4f} "
                f"avg_h_gain_mean={avg_h_gain:.4e} avg_noise_power={avg_noise:.4e} "
                f"avg_rx_signal_power_mean={avg_rx_signal:.4e} avg_interference_mean={avg_interf:.4e}"
            )
            self._append_metrics(
                step=self.num_timesteps,
                avg_reward_raw=avg_reward_raw,
                avg_reward_norm=avg_reward_norm,
                avg_ee_sum=avg_ee_sum,
                avg_sum_se=avg_se,
                avg_total_power=avg_total_power,
                avg_tau0_selected=avg_tau0_sel,
                avg_r1=avg_r1,
                avg_r2=avg_r2,
                avg_k=avg_k,
                avg_m=avg_m,
                avg_sinr_mean_tx=avg_sinr,
                avg_sinr_db_mean_tx=avg_sinr_db,
                sinr_db_p10=sinr_db_p10,
                sinr_db_p50=sinr_db_p50,
                sinr_db_p90=sinr_db_p90,
                sinr_db_min=sinr_db_min,
                sinr_db_max=sinr_db_max,
                avg_log2_mean_tx=avg_log2,
                avg_se_per_tx_user=avg_se_per_tx,
                avg_interference_mean=avg_interf,
            )
            self.sinr_db_vals = []
        self._maybe_append_ee()
        return True

    def _append_metrics(
        self,
        step: int,
        avg_reward_raw: float,
        avg_reward_norm: float,
        avg_ee_sum: float,
        avg_sum_se: float,
        avg_total_power: float,
        avg_tau0_selected: float,
        avg_r1: float,
        avg_r2: float,
        avg_k: float,
        avg_m: float,
        avg_sinr_mean_tx: float,
        avg_sinr_db_mean_tx: float,
        sinr_db_p10: float,
        sinr_db_p50: float,
        sinr_db_p90: float,
        sinr_db_min: float,
        sinr_db_max: float,
        avg_log2_mean_tx: float,
        avg_se_per_tx_user: float,
        avg_interference_mean: float,
    ) -> None:
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        file_exists = os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._metrics_header_written and not file_exists:
                writer.writerow(
                    [
                        "step",
                        "avg_reward_raw",
                        "avg_reward_norm",
                        "avg_ee_sum",
                        "avg_sum_se",
                        "avg_total_power",
                        "avg_tau0_selected",
                        "avg_r1",
                        "avg_r2",
                        "avg_K",
                        "avg_M",
                        "avg_sinr_mean_tx",
                        "avg_sinr_db_mean_tx",
                        "sinr_db_p10",
                        "sinr_db_p50",
                        "sinr_db_p90",
                        "sinr_db_min",
                        "sinr_db_max",
                        "avg_log2_mean_tx",
                        "avg_se_per_tx_user",
                        "avg_interference_mean",
                    ]
                )
                self._metrics_header_written = True
            writer.writerow(
                [
                    step,
                    avg_reward_raw,
                    avg_reward_norm,
                    avg_ee_sum,
                    avg_sum_se,
                    avg_total_power,
                    avg_tau0_selected,
                    avg_r1,
                    avg_r2,
                    avg_k,
                    avg_m,
                    avg_sinr_mean_tx,
                    avg_sinr_db_mean_tx,
                    sinr_db_p10,
                    sinr_db_p50,
                    sinr_db_p90,
                    sinr_db_min,
                    sinr_db_max,
                    avg_log2_mean_tx,
                    avg_se_per_tx_user,
                    avg_interference_mean,
                ]
            )

    def _maybe_append_ee(self) -> None:
        if self.eval_freq <= 0:
            return
        if not self._initial_eval_done:
            self._initial_eval_done = True
        elif self.num_timesteps - self._last_log_step < self.eval_freq:
            return
        if self.eval_env is None:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        eval_means = []
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=10000 + ep)
            done = False
            ee_steps = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ee_steps.append(float(info.get("ee_sum", 0.0)))
            eval_means.append(float(np.mean(ee_steps)) if ee_steps else 0.0)
        eval_mean = float(np.mean(eval_means)) if eval_means else 0.0
        eval_std = float(np.std(eval_means)) if eval_means else 0.0
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, eval_mean, eval_std])
        if eval_mean > self._best_eval:
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            self.model.save(self.best_model_path)
            self._best_eval = eval_mean
        print(
            f"step={self.num_timesteps} eval_ee_sum_mean={eval_mean:.4f} "
            f"eval_ee_sum_std={eval_std:.4f}"
        )
        self._last_log_step = self.num_timesteps


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_hash(cfg: Dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--ent-coef", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")
    print(f"USING_CONFIG={args.config}")
    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg.get("env", {}))
    train_cfg = cfg.get("train", {})
    cfg_hash_value = cfg_hash(cfg)

    print(f"Training config: {args.config}")
    print(
        "Config params:",
        f"Pmax={env_cfg.p_max}",
        f"P_c={env_cfg.p_c}",
        f"P_wet={env_cfg.p_wet}",
        f"gain_scale={env_cfg.channel.gain_scale}",
        f"sinr_cap={env_cfg.sinr_cap}",
    )

    total_timesteps = args.total_timesteps or int(train_cfg.get("total_timesteps", 300000))
    n_envs = args.n_envs or int(train_cfg.get("n_envs", 32))
    seed = int(cfg.get("seed", 42))
    env_cfg.total_timesteps = int(total_timesteps)
    print(f"TOTAL_TIMESTEPS={total_timesteps}")

    vec_env = make_vec_env(lambda: HapWpcnNomaEnv(config=env_cfg, seed=seed), n_envs=n_envs)

    ent_coef = train_cfg.get("ent_coef", "auto")
    if args.ent_coef is not None:
        if args.ent_coef.lower().startswith("auto"):
            ent_coef = args.ent_coef
        else:
            ent_coef = float(args.ent_coef)

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        buffer_size=int(train_cfg.get("buffer_size", 100000)),
        train_freq=int(train_cfg.get("train_freq", 1)),
        gradient_steps=int(train_cfg.get("gradient_steps", 1)),
        tau=float(train_cfg.get("sac_tau", 0.005)),
        ent_coef=ent_coef,
        learning_starts=int(train_cfg.get("learning_starts", 1000)),
        verbose=0,
        seed=seed,
    )

    log_interval = int(train_cfg.get("log_interval", 1000))
    eval_freq = int(train_cfg.get("eval_freq", 20000))
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 10))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", "runs", run_id)
    log_path = os.path.join(run_dir, "ee_vs_steps.csv")
    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    best_model_path = os.path.join(run_dir, "best_model")
    print(f"TOTAL_TIMESTEPS={total_timesteps}, EVAL_FREQ={eval_freq}")
    print(f"Logging to: {log_path}")
    eval_env = HapWpcnNomaEnv(config=env_cfg, seed=seed + 999)
    print(f"train_cfg_hash={cfg_hash_value} eval_cfg_hash={cfg_hash_value}")
    callback = MetricsCallback(
        log_interval=log_interval,
        eval_freq=eval_freq,
        log_path=log_path,
        metrics_path=metrics_path,
        best_model_path=best_model_path,
        eval_env=eval_env,
        n_eval_episodes=n_eval_episodes,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("models/sac_hap_wpcn_noma")
    os.makedirs("models/best", exist_ok=True)
    model.save(os.path.join("models", "best", "best_model"))
    vec_env.close()


if __name__ == "__main__":
    main()
