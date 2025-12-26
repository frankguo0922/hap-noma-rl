# Copilot instructions (hap-noma-rl)

## Big picture
- This repo trains RL agents (Stable-Baselines3 SAC/PPO) to maximize energy efficiency (EE) in a HAP-assisted WPCN uplink with dynamic users and NOMA pairing.
- Key boundary: `train/` (algorithms + logging) drives a Gymnasium environment in `envs/` (physics + reward).

## Key entry points (run as modules)
- Install deps: `pip install -r requirements.txt` ([requirements.txt](../requirements.txt))
- Train SAC: `python -m train.train_sac` ([train/train_sac.py](../train/train_sac.py))
- Train PPO: `python -m train.train_ppo` ([train/train_ppo.py](../train/train_ppo.py))
- Evaluate: `python -m train.eval --model models/sac_hap_wpcn_noma.zip --episodes 20` ([train/eval.py](../train/eval.py))
- Plot: `python -m scripts.plot_ee_vs_steps` (reads `logs/ee_vs_steps.csv`, writes `figures/sac_convergence_ee.png`) ([scripts/plot_ee_vs_steps.py](../scripts/plot_ee_vs_steps.py))
- Sanity check: `python -m scripts.check_energy_causality` ([scripts/check_energy_causality.py](../scripts/check_energy_causality.py))

## Configuration conventions
- All runs load YAML (default: `configs/default.yaml`) and split into:
  - `env`: forwarded into `EnvConfig` via `build_env_config(...)` ([envs/hap_wpcn_noma_env.py](../envs/hap_wpcn_noma_env.py))
  - `train`: SB3 hyperparams used by SAC/PPO trainers ([configs/default.yaml](../configs/default.yaml))
- Common stability knob: if SINR becomes extremely small, increase `env.channel.gain_scale` (README note) ([README.md](../README.md)).
- `env.noise_mode`: `"thermal"` computes noise from `N0_dBm_perHz`, `B_Hz`, `noise_figure_dB`; otherwise uses fixed `env.noise_power` ([envs/hap_wpcn_noma_env.py](../envs/hap_wpcn_noma_env.py)).

## Environment API (important when changing features/reward)
- Env class: `HapWpcnNomaEnv` ([envs/hap_wpcn_noma_env.py](../envs/hap_wpcn_noma_env.py))
- Observation: `Box(shape=(n_wd, feat_dim))` with columns:
  - `[active_mask, channel_gain, sin(aoa), cos(aoa), battery_energy, (optional) distance]`
  - `feat_dim = 5 (+1 if include_distance)`.
- Action: continuous `Box(shape=(1 + n_wd + n_wd,))` parsed as:
  - `a_tau`: mapped to `tau0 in [tau_min, tau_max]` via affine+clip ([envs/utils.py](../envs/utils.py))
  - `a_beam`: mapped into {0,1,2} using thresholds (-0.33, 0.33); 0 means “don’t transmit”
  - `a_p`: mapped to power in `[0, p_max]` via affine+clip
- Dynamics: users join/leave each step via `p_join/p_leave`; channel gains resampled each step for active users ([envs/channel.py](../envs/channel.py)).

## NOMA/SINR implementation notes
- Pairing is per-beam by descending channel gain: (strong, weak) pairs; odd user count yields (strong, None) ([envs/grouping.py](../envs/grouping.py)).
- Rates use `tau1 * log2(1+SINR)`; SIC uses `sic_threshold` and `residual_factor` ([envs/sinr.py](../envs/sinr.py)).

## Rewards, constraints, and logging (don’t break dashboards)
- Reward is based on EE per transmitting user: `reward_raw = ee_sum / max(n_tx_users, 1)` then scaled/penalized by `reward_scale`, `mu_energy`, `mu_infeasible` ([envs/hap_wpcn_noma_env.py](../envs/hap_wpcn_noma_env.py)).
- Energy causality is enforced by clipping powers to `available/tau1`; diagnostics are emitted via `info`:
  - `battery_before`, `harvested_energy`, `tx_energy`, `battery_after`, `infeasible_count`, `energy_penalty`, `sinr_mean`, `n_tx_users`, `total_power`, `ee_avg`, etc.
- SAC training appends `[step, mean(ee_avg)]` rows to `logs/ee_vs_steps.csv` (used by plotting script) ([train/train_sac.py](../train/train_sac.py)).

## Working with models/artifacts
- Trainers save to `models/sac_hap_wpcn_noma.zip` or `models/ppo_hap_wpcn_noma.zip` ([train/train_sac.py](../train/train_sac.py), [train/train_ppo.py](../train/train_ppo.py)).
- `train.eval` auto-detects model type by filename suffix, then falls back to trying SAC then PPO ([train/eval.py](../train/eval.py)).
