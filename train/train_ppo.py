"""Train PPO on HapWpcnNomaEnv."""
from __future__ import annotations

import argparse
from collections import deque
from typing import Dict, List

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config


class MetricsCallback(BaseCallback):
    def __init__(self, log_interval: int):
        super().__init__()
        self.log_interval = log_interval
        self.rewards = deque(maxlen=log_interval)
        self.ee = deque(maxlen=log_interval)
        self.ee_sum = deque(maxlen=log_interval)
        self.ee_avg = deque(maxlen=log_interval)
        self.sum_se = deque(maxlen=log_interval)
        self.tau0 = deque(maxlen=log_interval)
        self.sinr_mean = deque(maxlen=log_interval)
        self.log2_mean = deque(maxlen=log_interval)
        self.se_per_tx = deque(maxlen=log_interval)
        self.p_tx_mean = deque(maxlen=log_interval)
        self.h_gain_mean = deque(maxlen=log_interval)
        self.noise_power = deque(maxlen=log_interval)
        self.rx_signal_mean = deque(maxlen=log_interval)
        self.interference_mean = deque(maxlen=log_interval)
        self.n_tx_users = deque(maxlen=log_interval)
        self.total_power = deque(maxlen=log_interval)
        self.reward_raw = deque(maxlen=log_interval)
        self.power_wet = deque(maxlen=log_interval)
        self.power_ul = deque(maxlen=log_interval)
        self.power_c = deque(maxlen=log_interval)
        self.tau0_sel = deque(maxlen=log_interval)
        self.r1 = deque(maxlen=log_interval)
        self.r2 = deque(maxlen=log_interval)
        self.k_val = deque(maxlen=log_interval)
        self.m_val = deque(maxlen=log_interval)

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
            avg_p_tx = float(np.mean(self.p_tx_mean)) if self.p_tx_mean else 0.0
            avg_h_gain = float(np.mean(self.h_gain_mean)) if self.h_gain_mean else 0.0
            avg_noise = float(np.mean(self.noise_power)) if self.noise_power else 0.0
            avg_rx_signal = float(np.mean(self.rx_signal_mean)) if self.rx_signal_mean else 0.0
            avg_interf = float(np.mean(self.interference_mean)) if self.interference_mean else 0.0
            avg_n_tx = float(np.mean(self.n_tx_users)) if self.n_tx_users else 0.0
            avg_total_power = float(np.mean(self.total_power)) if self.total_power else 0.0
            avg_reward_raw = float(np.mean(self.reward_raw)) if self.reward_raw else 0.0
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
                f"avg_ee_sum={avg_ee_sum:.4f} avg_ee_avg={avg_ee_avg:.4f} "
                f"avg_sum_se={avg_se:.4f} avg_n_tx_users={avg_n_tx:.2f} avg_total_power={avg_total_power:.4f} "
                f"avg_power_wet={avg_power_wet:.4f} avg_power_ul={avg_power_ul:.4f} avg_power_c={avg_power_c:.4f} "
                f"avg_ee={avg_ee:.4f} avg_tau0={avg_tau0:.4f} "
                f"avg_tau0_selected={avg_tau0_sel:.4f} avg_r1={avg_r1:.4f} avg_r2={avg_r2:.4f} "
                f"avg_K={avg_k:.1f} avg_M={avg_m:.1f} "
                f"avg_sinr_mean_tx={avg_sinr:.4f} avg_log2_mean_tx={avg_log2:.4f} "
                f"avg_se_per_tx_user={avg_se_per_tx:.4f} avg_p_tx_mean={avg_p_tx:.4f} "
                f"avg_h_gain_mean={avg_h_gain:.4e} avg_noise_power={avg_noise:.4e} "
                f"avg_rx_signal_power_mean={avg_rx_signal:.4e} avg_interference_mean={avg_interf:.4e}"
            )
        return True


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg.get("env", {}))
    train_cfg = cfg.get("train", {})

    total_timesteps = args.total_timesteps or int(train_cfg.get("total_timesteps", 300000))
    n_envs = args.n_envs or int(train_cfg.get("n_envs", 32))
    seed = int(cfg.get("seed", 42))

    vec_env = make_vec_env(lambda: HapWpcnNomaEnv(config=env_cfg, seed=seed), n_envs=n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        n_steps=int(train_cfg.get("ppo_n_steps", 256)),
        batch_size=int(train_cfg.get("ppo_batch_size", 256)),
        verbose=0,
        seed=seed,
    )

    log_interval = int(train_cfg.get("log_interval", 1000))
    callback = MetricsCallback(log_interval=log_interval)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("models/ppo_hap_wpcn_noma")
    vec_env.close()


if __name__ == "__main__":
    main()
