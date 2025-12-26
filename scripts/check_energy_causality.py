"""Runtime sanity check for WPCN energy causality."""
from __future__ import annotations

import yaml
import numpy as np

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv, build_env_config

TOL = 1e-6
N_EPISODES = 5
MAX_STEPS = 200


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_from_obs(obs: np.ndarray) -> np.ndarray:
    # obs shape: (n_wd, feat_dim), active at column 0
    return obs[:, 0] > 0.5


def main():
    cfg = load_config("configs/default.yaml")
    env_cfg = build_env_config(cfg.get("env", {}))
    env = HapWpcnNomaEnv(config=env_cfg, seed=int(cfg.get("seed", 42)))

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=int(cfg.get("seed", 42)) + ep)
        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            active = info.get("active", None)
            if active is None:
                active = get_active_from_obs(obs)

            e_old = info.get("battery_before")
            e_harvest = info.get("harvested_energy")
            e_tx = info.get("tx_energy")
            e_new = info.get("battery_after")
            tau1 = float(info.get("tau1", 0.0))
            e_max = float(env_cfg.e_max)

            if any(x is None for x in (e_old, e_harvest, e_tx, e_new)):
                raise AssertionError("Missing energy fields in info dict.")

            active_idx = np.where(active)[0]
            inactive_idx = np.where(~active)[0]

            if active_idx.size > 0:
                lhs = e_tx[active_idx]
                rhs = e_old[active_idx] + e_harvest[active_idx] + TOL
                if np.any(lhs > rhs):
                    i = int(active_idx[np.argmax(lhs - rhs)])
                    print("Energy causality violation:")
                    print(ep, step, i, e_old[i], e_harvest[i], e_tx[i], e_new[i], active[i])
                    raise AssertionError("E_tx exceeds available energy.")

                expected = np.clip(
                    e_old[active_idx] + e_harvest[active_idx] - e_tx[active_idx], 0.0, e_max
                )
                if np.any(np.abs(e_new[active_idx] - expected) > TOL):
                    i = int(active_idx[np.argmax(np.abs(e_new[active_idx] - expected))])
                    print("Battery update mismatch:")
                    print(ep, step, i, e_old[i], e_harvest[i], e_tx[i], e_new[i], active[i])
                    raise AssertionError("Battery update incorrect.")

            if inactive_idx.size > 0:
                if np.any(np.abs(e_harvest[inactive_idx]) > TOL) or np.any(
                    np.abs(e_tx[inactive_idx]) > TOL
                ):
                    i = int(inactive_idx[0])
                    print("Inactive energy violation:")
                    print(ep, step, i, e_old[i], e_harvest[i], e_tx[i], e_new[i], active[i])
                    raise AssertionError("Inactive nodes must have zero energy activity.")

            if terminated or truncated:
                break

    print("PASS: energy causality holds for all tested steps.")


if __name__ == "__main__":
    main()
