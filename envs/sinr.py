"""SINR and rate calculations for uplink NOMA."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from envs.grouping import pair_by_gain


def inter_beam_interference(
    idx: int,
    beam: np.ndarray,
    active: np.ndarray,
    aoa: np.ndarray,
    power: np.ndarray,
    gain: np.ndarray,
    inter_k: float,
    inter_scale: float,
) -> float:
    beam_id = beam[idx]
    if beam_id not in (1, 2):
        return 0.0
    other_beam = 2 if beam_id == 1 else 1
    other_idx = np.where(active & (beam == other_beam))[0]
    if other_idx.size == 0:
        return 0.0
    delta = np.abs(aoa[idx] - aoa[other_idx])
    coef = np.exp(-inter_k * delta)
    return float(np.sum(gain[other_idx] * power[other_idx] * coef) * inter_scale)


def compute_rates(
    active: np.ndarray,
    beam: np.ndarray,
    power: np.ndarray,
    gain: np.ndarray,
    aoa: np.ndarray,
    tau1: float,
    noise: float,
    inter_k: float,
    inter_scale: float,
    sic_threshold: float,
    residual_factor: float,
    sinr_cap: float,
    noise_floor: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, List[Tuple[int, Optional[int]]]]
]:
    rates = np.zeros_like(power, dtype=np.float32)
    sinr_vals = np.zeros_like(power, dtype=np.float32)
    sinr_raw_vals = np.zeros_like(power, dtype=np.float32)
    inter_vals = np.zeros_like(power, dtype=np.float32)
    log2_vals = np.zeros_like(power, dtype=np.float32)
    if tau1 <= 0.0:
        return rates, sinr_vals, sinr_raw_vals, inter_vals, log2_vals, {1: [], 2: []}

    pairs = pair_by_gain(gain, active, beam)

    for beam_id, pair_list in pairs.items():
        for strong, weak in pair_list:
            inter_s = inter_beam_interference(
                strong, beam, active, aoa, power, gain, inter_k, inter_scale
            )
            if weak is None:
                denom_s = max(noise + inter_s, noise_floor)
                sinr_raw = gain[strong] * power[strong] / denom_s
                sinr_clip = float(np.clip(sinr_raw, 0.0, sinr_cap))
                rates[strong] = tau1 * np.log2(1.0 + sinr_clip)
                sinr_vals[strong] = sinr_clip
                sinr_raw_vals[strong] = sinr_raw
                inter_vals[strong] = inter_s
                log2_vals[strong] = np.log2(1.0 + sinr_clip)
                continue

            inter_w = inter_beam_interference(
                weak, beam, active, aoa, power, gain, inter_k, inter_scale
            )
            signal_w = gain[weak] * power[weak]
            interf_w = noise + inter_w + gain[strong] * power[strong]
            denom_w = max(interf_w, noise_floor) + 1e-9
            sinr_w_raw = signal_w / denom_w
            sinr_w_clip = float(np.clip(sinr_w_raw, 0.0, sinr_cap))
            rates[weak] = tau1 * np.log2(1.0 + sinr_w_clip)
            sinr_vals[weak] = sinr_w_clip
            sinr_raw_vals[weak] = sinr_w_raw
            inter_vals[weak] = inter_w
            log2_vals[weak] = np.log2(1.0 + sinr_w_clip)

            strong_signal = gain[strong] * power[strong]
            weak_signal = gain[weak] * power[weak]
            if strong_signal >= sic_threshold * weak_signal:
                residual = residual_factor * weak_signal * 0.5
            else:
                residual = residual_factor * weak_signal
            denom_s = max(noise + inter_s + residual, noise_floor) + 1e-9
            sinr_s_raw = strong_signal / denom_s
            sinr_s_clip = float(np.clip(sinr_s_raw, 0.0, sinr_cap))
            rates[strong] = tau1 * np.log2(1.0 + sinr_s_clip)
            sinr_vals[strong] = sinr_s_clip
            sinr_raw_vals[strong] = sinr_s_raw
            inter_vals[strong] = inter_s
            log2_vals[strong] = np.log2(1.0 + sinr_s_clip)

    return rates, sinr_vals, sinr_raw_vals, inter_vals, log2_vals, pairs
