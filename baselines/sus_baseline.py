"""SUS-based Hybrid SDMA/NOMA baseline (Amer et al. 2025 inspired).

This is a simplified, myopic heuristic:
- fixed tau0
- greedy SUS selection (no combinatorial search)
- fixed NOMA power split (no r optimization)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from envs.hap_wpcn_noma_env import HapWpcnNomaEnv
from envs.sinr import compute_rates
from envs.wpcn import harvest_energy


def _select_strong_users(active_idx: np.ndarray, gains: np.ndarray, aoa: np.ndarray) -> Tuple[int | None, int | None]:
    if active_idx.size == 0:
        return None, None
    # First strong: max gain
    first = int(active_idx[np.argmax(gains[active_idx])])
    if active_idx.size == 1:
        return first, None

    # Second strong: high gain + angular separation from first
    delta = np.abs(aoa[active_idx] - aoa[first])
    score = gains[active_idx] * (1.0 + 0.5 * np.sin(delta))
    score[active_idx == first] = -np.inf
    second_idx = int(active_idx[np.argmax(score)])
    return first, second_idx


def _assign_beams(aoa: np.ndarray, idx: int | None) -> int:
    if idx is None:
        return 0
    return 1 if aoa[idx] < 0.0 else 2


def _pick_weak_user(active_idx: np.ndarray, gains: np.ndarray, beam: np.ndarray, beam_id: int, exclude: set) -> int | None:
    candidates = [i for i in active_idx if beam[i] == beam_id and i not in exclude]
    if not candidates:
        return None
    cand_arr = np.array(candidates, dtype=int)
    return int(cand_arr[np.argmin(gains[cand_arr])])


def build_sus_action(env: HapWpcnNomaEnv, tau0: float, weak_ratio: float = 0.7) -> np.ndarray:
    active_idx = np.where(env.active)[0]
    gains = env.channel.gain
    aoa = env.channel.aoa

    strong1, strong2 = _select_strong_users(active_idx, gains, aoa)

    beam = np.zeros(env.max_wd, dtype=np.int64)
    if strong1 is not None:
        beam[strong1] = _assign_beams(aoa, strong1)
    if strong2 is not None:
        beam[strong2] = _assign_beams(aoa, strong2)

    used = {i for i in (strong1, strong2) if i is not None}
    for beam_id in (1, 2):
        weak = _pick_weak_user(active_idx, gains, beam, beam_id, used)
        if weak is not None:
            beam[weak] = beam_id
            used.add(weak)

    tau1 = 1.0 - tau0
    harvested = harvest_energy(gains, env.active, tau0, env.cfg.p_wet, env.cfg.eta)
    available = env.energy + harvested
    power = np.zeros(env.max_wd, dtype=np.float32)
    if tau1 > 0.0:
        max_power = np.minimum(available / tau1, env.cfg.p_max)
        for beam_id in (1, 2):
            idx = np.where(beam == beam_id)[0]
            if idx.size == 0:
                continue
            if idx.size == 1:
                i = int(idx[0])
                power[i] = max_power[i]
                continue
            gains_idx = gains[idx]
            strong = int(idx[np.argmax(gains_idx)])
            weak = int(idx[np.argmin(gains_idx)])
            pair_budget = max_power[weak] + max_power[strong]
            power[weak] = min(max_power[weak], pair_budget * weak_ratio)
            power[strong] = min(max_power[strong], pair_budget * (1.0 - weak_ratio))

    power[~env.active] = 0.0
    power[beam == 0] = 0.0

    a_tau = 2.0 * (tau0 - env.cfg.tau_min) / (env.cfg.tau_max - env.cfg.tau_min) - 1.0
    a_tau = float(np.clip(a_tau, -1.0, 1.0))
    a_beam = np.full(env.max_wd, -0.5, dtype=np.float32)
    a_beam[beam == 1] = 0.0
    a_beam[beam == 2] = 0.5
    a_p = 2.0 * (power / env.cfg.p_max) - 1.0
    a_p = np.clip(a_p, -1.0, 1.0).astype(np.float32)

    action = np.concatenate(([a_tau], a_beam, a_p)).astype(np.float32)
    return action


def _compute_noise_power(env: HapWpcnNomaEnv) -> float:
    if env.cfg.noise_mode == "thermal":
        n0_w_per_hz = 10 ** ((env.cfg.N0_dBm_perHz - 30.0) / 10.0)
        nf = 10 ** (env.cfg.noise_figure_dB / 10.0)
        return n0_w_per_hz * env.cfg.B_Hz * nf
    return env.cfg.noise_power


def sus_baseline_step(env: HapWpcnNomaEnv, weak_ratio: float = 0.7) -> Tuple[np.ndarray, float, Dict[str, float]]:
    sus_tau0 = getattr(env.cfg, "sus_tau0", None)
    if sus_tau0 is not None:
        tau0 = float(sus_tau0)
    else:
        tau0_attr = getattr(env.cfg, "tau0_baseline", None)
        tau0 = float(tau0_attr) if tau0_attr is not None else 0.1

    action = build_sus_action(env, tau0, weak_ratio=weak_ratio)
    a_beam = action[1 : 1 + env.max_wd]
    a_p = action[1 + env.max_wd : 1 + 2 * env.max_wd]
    beam = np.zeros(env.max_wd, dtype=np.int64)
    beam[(a_beam >= -0.33) & (a_beam < 0.33)] = 1
    beam[a_beam >= 0.33] = 2
    power = np.clip(0.5 * (a_p + 1.0), 0.0, 1.0) * env.cfg.p_max
    power[~env.active] = 0.0
    power[beam == 0] = 0.0
    tau1 = 1.0 - tau0
    if tau1 <= 0.0:
        power[:] = 0.0

    noise_power = _compute_noise_power(env)
    rates, _, _, _, _, _ = compute_rates(
        active=env.active,
        beam=beam,
        power=power,
        gain=env.channel.gain,
        aoa=env.channel.aoa,
        tau1=tau1,
        noise=noise_power,
        inter_k=env.cfg.inter_beam_k,
        inter_scale=env.cfg.inter_beam_scale,
        sic_threshold=env.cfg.sic_threshold,
        residual_factor=env.cfg.residual_factor,
        sinr_cap=env.cfg.sinr_cap,
        noise_floor=env.cfg.noise_floor,
    )
    sum_se = float(np.sum(rates))
    total_power = env.cfg.p_wet * tau0 + float(np.sum(power) * tau1) + env.cfg.p_c
    ee_sum = sum_se / (total_power + 1e-9)
    info = {"tau0": float(tau0), "solver_used": False, "note": "fixed_tau0_no_grid"}
    return action, float(ee_sum), info
