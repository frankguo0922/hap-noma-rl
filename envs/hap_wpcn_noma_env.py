"""Gymnasium environment for HAP WPCN NOMA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

from envs.channel import ChannelConfig, ChannelModel
from envs.grouping import assign_beams_by_aoa
from envs.sinr import compute_rates
from envs.utils import map_action_to_beam, map_action_to_power, map_action_to_tau0
from envs.wpcn import harvest_energy


@dataclass
class EnvConfig:
    n_wd: int = 50
    max_wd: int = 50
    p_join: float = 0.05
    p_leave: float = 0.05
    init_active_prob: float = 0.5

    p_wet: float = 5.0
    eta: float = 0.6
    p_max: float = 1.0
    p_c: float = 0.1
    e_max: float = 5.0
    power_action_scale: float = 1.0

    tau_min: float = 0.1
    tau_max: float = 0.9
    tau0_mode: str = "grid"

    noise_power: float = 1e-3
    noise_mode: str = "fixed"
    bandwidth: float = 1.0
    B_Hz: float = 1e6
    N0_dBm_perHz: float = -174.0
    noise_figure_dB: float = 0.0
    inter_beam_k: float = 2.0
    inter_beam_scale: float = 0.1
    sic_threshold: float = 1.2
    residual_factor: float = 0.05
    sinr_cap: float = 100.0
    noise_floor: float = 0.0

    reward_scale: float = 1.0
    mu_energy: float = 0.5
    mu_infeasible: float = 0.0
    reward_nu: float = 0.0
    sinr_target_db: float = 12.0
    sinr_margin_db: float = 5.0
    w_sinr0: float = 0.25
    w_sinr_min: float = 0.05
    w_sinr_T: float = 10000.0
    total_timesteps: int = 0
    is_training: bool = True
    curriculum: Dict[str, Any] = field(default_factory=dict)
    reward_ref_ee: float = 20.0
    reward_norm_mode: str = "ratio"
    reward_baseline: float = 0.0
    solver_k: int = 10
    solver_m: int = 60
    tau0_candidates: int = 10
    solver_k: int = 10
    solver_m: int = 60
    tau0_candidates: int = 10

    include_distance: bool = True
    max_steps: int = 200

    channel: ChannelConfig = field(default_factory=ChannelConfig)
    use_solver: bool = True
    tau0_baseline: float | None = None


def build_env_config(cfg: Optional[Dict[str, Any]] = None) -> EnvConfig:
    cfg = cfg or {}
    channel_cfg = cfg.get("channel", {})
    if isinstance(channel_cfg, dict):
        channel_cfg = ChannelConfig(**channel_cfg)
    if "max_wd" not in cfg and "n_wd" in cfg:
        cfg = {**cfg, "max_wd": cfg["n_wd"]}
    return EnvConfig(**{**cfg, "channel": channel_cfg})


class HapWpcnNomaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.n_wd = self.cfg.n_wd
        self.max_wd = max(self.cfg.max_wd, self.n_wd)
        self.rng = np.random.default_rng(seed)

        self.channel = ChannelModel(self.max_wd, self.cfg.channel, self.rng)

        self.active = np.zeros(self.max_wd, dtype=bool)
        self.energy = np.zeros(self.max_wd, dtype=np.float32)
        self.step_count = 0

        self.include_distance = self.cfg.include_distance
        self.feat_dim = 5 + (1 if self.include_distance else 0)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.max_wd, self.feat_dim), dtype=np.float32
        )
        action_dim = 1 + self.max_wd + self.max_wd
        low = np.full(action_dim, -1.0, dtype=np.float32)
        high = np.full(action_dim, 1.0, dtype=np.float32)
        low[0] = 0.0
        high[0] = 1.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _reset_state(self) -> None:
        self.active = np.zeros(self.max_wd, dtype=bool)
        active_mask = self.rng.random(self.n_wd) < self.cfg.init_active_prob
        self.active[: self.n_wd] = active_mask
        active_idx = np.where(self.active)[0]
        if active_idx.size > 0:
            self.channel.sample_positions(active_idx)
            self.channel.sample_gain(active_idx)
            self.energy[active_idx] = self.rng.uniform(0.2, self.cfg.e_max, size=active_idx.size)
        inactive_idx = np.where(~self.active)[0]
        if inactive_idx.size > 0:
            self.channel.distance[inactive_idx] = 0.0
            self.channel.aoa[inactive_idx] = 0.0
            self.channel.gain[inactive_idx] = 0.0
            self.energy[inactive_idx] = 0.0

    def _get_obs(self) -> np.ndarray:
        active_mask = self.active.astype(np.float32)
        sin_aoa = np.sin(self.channel.aoa)
        cos_aoa = np.cos(self.channel.aoa)
        feats = [active_mask, self.channel.gain, sin_aoa, cos_aoa, self.energy]
        if self.include_distance:
            feats.append(self.channel.distance)
        obs = np.stack(feats, axis=1).astype(np.float32)
        return obs

    def _apply_activity_dynamics(self) -> None:
        valid_idx = np.arange(self.n_wd)
        active_idx = valid_idx[self.active[: self.n_wd]]
        inactive_idx = valid_idx[~self.active[: self.n_wd]]

        if active_idx.size > 0:
            leave_mask = self.rng.random(active_idx.size) < self.cfg.p_leave
            leaving = active_idx[leave_mask]
            self.active[leaving] = False
            self.channel.distance[leaving] = 0.0
            self.channel.aoa[leaving] = 0.0
            self.channel.gain[leaving] = 0.0
            self.energy[leaving] = 0.0

        if inactive_idx.size > 0:
            join_mask = self.rng.random(inactive_idx.size) < self.cfg.p_join
            joining = inactive_idx[join_mask]
            if joining.size > 0:
                self.active[joining] = True
                self.channel.sample_positions(joining)
                self.channel.sample_gain(joining)
                self.energy[joining] = self.rng.uniform(0.2, self.cfg.e_max, size=joining.size)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel.rng = self.rng
        self.step_count = 0
        self._reset_state()
        obs = self._get_obs()
        info = {"sum_rate": 0.0, "ee": 0.0, "tau0": 0.0, "infeasible_count": 0}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).flatten()
        a_tau = float(np.clip(action[0], 0.0, 1.0))
        a_beam = action[1 : 1 + self.max_wd]
        a_tail = action[1 + self.max_wd : 1 + 2 * self.max_wd]
        tau0_agent = self.cfg.tau_min + a_tau * (self.cfg.tau_max - self.cfg.tau_min)

        a_sel = a_tail[: self.max_wd - 2]
        a_r1 = a_tail[self.max_wd - 2]
        a_r2 = a_tail[self.max_wd - 1]
        r1 = float(1.0 / (1.0 + np.exp(-a_r1)))
        r2 = float(1.0 / (1.0 + np.exp(-a_r2)))

        curriculum_cfg = self.cfg.curriculum or {}
        if self.cfg.is_training and curriculum_cfg.get("enabled", False):
            total_steps = float(curriculum_cfg.get("total_steps", self.cfg.total_timesteps or 1))
            progress = float(np.clip(self.step_count / max(total_steps, 1.0), 0.0, 1.0))
        else:
            progress = 1.0

        solver_curr_enabled = (
            self.cfg.is_training and curriculum_cfg.get("solver_enabled", False)
        )
        if solver_curr_enabled:
            k_start = int(curriculum_cfg.get("solver_k_start", self.cfg.solver_k))
            k_end = int(curriculum_cfg.get("solver_k_end", self.cfg.solver_k))
            m_start = int(curriculum_cfg.get("solver_m_start", self.cfg.solver_m))
            m_end = int(curriculum_cfg.get("solver_m_end", self.cfg.solver_m))
            tau_start = int(curriculum_cfg.get("tau0_candidates_start", self.cfg.tau0_candidates))
            tau_end = int(curriculum_cfg.get("tau0_candidates_end", self.cfg.tau0_candidates))
            rand_start = float(curriculum_cfg.get("random_combo_prob_start", 0.0))
            rand_end = float(curriculum_cfg.get("random_combo_prob_end", 0.0))
            solver_k_eff = int(round(k_start + (k_end - k_start) * progress))
            solver_m_eff = int(round(m_start + (m_end - m_start) * progress))
            tau0_count_eff = int(round(tau_start + (tau_end - tau_start) * progress))
            random_combo_prob = rand_start + (rand_end - rand_start) * progress
        else:
            solver_k_eff = self.cfg.solver_k
            solver_m_eff = self.cfg.solver_m
            tau0_count_eff = self.cfg.tau0_candidates
            random_combo_prob = 0.0

        # Upper-layer selection: top-K active users by selection score.
        sel_scores = np.full(self.max_wd, -np.inf, dtype=np.float32)
        sel_scores[: self.max_wd - 2] = a_sel
        active_idx = np.where(self.active)[0]
        k_candidates = int(max(1, solver_k_eff))
        if active_idx.size > 0:
            order = active_idx[np.argsort(sel_scores[active_idx])[::-1]]
            candidates = order[: min(k_candidates, order.size)]
        else:
            candidates = np.array([], dtype=int)

        beam_choice = np.where(a_beam >= 0.0, 1, 2)
        battery_before = self.energy.copy()

        best_combo = np.array([], dtype=int)
        best_tau0 = self.cfg.tau_min
        best_sum_se = 0.0
        sampled = 0
        best_ee = -np.inf

        self.channel.update(np.where(self.active)[0])

        effective_gain = self.channel.gain

        if self.cfg.noise_mode == "thermal":
            n0_w_per_hz = 10 ** ((self.cfg.N0_dBm_perHz - 30.0) / 10.0)
            nf = 10 ** (self.cfg.noise_figure_dB / 10.0)
            noise_power = n0_w_per_hz * self.cfg.B_Hz * nf
        else:
            noise_power = self.cfg.noise_power

        if self.cfg.use_solver and candidates.size >= 1:
            import itertools

            best_beam = np.zeros(self.max_wd, dtype=np.int64)
            best_power = np.zeros(self.max_wd, dtype=np.float32)
            rng = self.rng
            total_combos = 0
            if candidates.size >= 4:
                total_combos = int(math.comb(int(candidates.size), 4))

            max_enum = max(1, solver_m_eff)
            if candidates.size <= 8 and total_combos <= max_enum:
                combos = list(itertools.combinations(candidates.tolist(), 4))
            else:
                rng = self.rng
                combos = []
                beam1_cand = [i for i in candidates if beam_choice[i] == 1]
                beam2_cand = [i for i in candidates if beam_choice[i] == 2]
                if random_combo_prob > 0.0 and rng.random() < random_combo_prob:
                    for _ in range(max_enum):
                        pick = rng.choice(candidates, size=4, replace=False)
                        combo = tuple(sorted([int(x) for x in pick]))
                        combos.append(combo)
                else:
                    half = max(1, max_enum // 2)
                    if len(beam1_cand) >= 2 and len(beam2_cand) >= 2:
                        for _ in range(half):
                            pick1 = rng.choice(beam1_cand, size=2, replace=False)
                            pick2 = rng.choice(beam2_cand, size=2, replace=False)
                            combo = tuple(sorted([int(x) for x in np.concatenate([pick1, pick2])]))
                            combos.append(combo)
                    for _ in range(max_enum - len(combos)):
                        pick = rng.choice(candidates, size=4, replace=False)
                        combo = tuple(sorted([int(x) for x in pick]))
                        combos.append(combo)
                combos = list(dict.fromkeys(combos))
                sampled = len(combos)

            if self.cfg.tau0_mode == "agent":
                tau0_candidates = np.array([tau0_agent], dtype=np.float32)
            else:
                tau0_candidates = np.linspace(
                    self.cfg.tau_min, self.cfg.tau_max, int(max(2, tau0_count_eff))
                )
            battery_before = self.energy.copy()

            for combo in combos:
                combo_idx = np.array(combo, dtype=int)
                beam = np.zeros(self.max_wd, dtype=np.int64)
                beam[combo_idx] = beam_choice[combo_idx]
                for beam_id in (1, 2):
                    idx = np.where(beam == beam_id)[0]
                    if idx.size > 2:
                        top2 = idx[np.argsort(sel_scores[idx])[::-1][:2]]
                        drop_idx = idx[~np.isin(idx, top2)]
                        beam[drop_idx] = 0

                for tau0 in tau0_candidates:
                    tau1 = 1.0 - tau0
                    harvested = harvest_energy(
                        effective_gain, self.active, tau0, self.cfg.p_wet, self.cfg.eta
                    )
                    available = battery_before + harvested
                    power = np.zeros(self.max_wd, dtype=np.float32)
                    if tau1 > 0.0:
                        max_power = np.minimum(available / tau1, self.cfg.p_max)
                        for beam_id, r in ((1, r1), (2, r2)):
                            idx = np.where(beam == beam_id)[0]
                            if idx.size == 0:
                                continue
                            if idx.size == 1:
                                i = int(idx[0])
                                power[i] = min(max_power[i], self.cfg.p_max)
                                continue
                            gains_idx = self.channel.gain[idx]
                            strong = int(idx[np.argmax(gains_idx)])
                            weak = int(idx[np.argmin(gains_idx)])
                            p_weak = min(max_power[weak], self.cfg.p_max * r)
                            p_strong = min(max_power[strong], self.cfg.p_max * (1.0 - r))
                            power[weak] = p_weak
                            power[strong] = p_strong
                        power = np.clip(power, 0.0, self.cfg.p_max)
                    power[beam == 0] = 0.0
                    if tau1 <= 0.0:
                        power[:] = 0.0

                    rates, _, _, _, _, _ = compute_rates(
                        active=self.active,
                        beam=beam,
                        power=power,
                        gain=effective_gain,
                        aoa=self.channel.aoa,
                        tau1=tau1,
                        noise=noise_power,
                        inter_k=self.cfg.inter_beam_k,
                        inter_scale=self.cfg.inter_beam_scale,
                        sic_threshold=self.cfg.sic_threshold,
                        residual_factor=self.cfg.residual_factor,
                        sinr_cap=self.cfg.sinr_cap,
                        noise_floor=self.cfg.noise_floor,
                    )
                    sum_se = float(np.sum(rates))
                    total_power = self.cfg.p_wet * tau0 + float(np.sum(power) * tau1) + self.cfg.p_c
                    ee_sum = sum_se / (total_power + 1e-9)
                    if ee_sum > best_ee:
                        best_ee = ee_sum
                        best_tau0 = tau0
                        best_power = power
                        best_sum_se = sum_se
                        best_combo = combo_idx
                        best_beam = beam.copy()

            beam = best_beam
            power = best_power
            tau0 = best_tau0
            tau1 = 1.0 - tau0
            sum_se = best_sum_se
        else:
            scheduled = candidates[: min(4, candidates.size)] if candidates.size > 0 else np.array([], dtype=int)
            beam = np.zeros(self.max_wd, dtype=np.int64)
            if scheduled.size > 0:
                beam[scheduled] = beam_choice[scheduled]
            for beam_id in (1, 2):
                idx = np.where(beam == beam_id)[0]
                if idx.size > 2:
                    top2 = idx[np.argsort(sel_scores[idx])[::-1][:2]]
                    drop_idx = idx[~np.isin(idx, top2)]
                    beam[drop_idx] = 0
            beam[~self.active] = 0

            if self.cfg.tau0_mode == "agent":
                tau0 = tau0_agent
            else:
                tau0 = float(self.cfg.tau0_baseline) if self.cfg.tau0_baseline is not None else map_action_to_tau0(
                    a_tau, self.cfg.tau_min, self.cfg.tau_max
                )
            tau1 = 1.0 - tau0
            battery_before = self.energy.copy()
            harvested = harvest_energy(effective_gain, self.active, tau0, self.cfg.p_wet, self.cfg.eta)
            available = battery_before + harvested
            power = np.zeros(self.max_wd, dtype=np.float32)
            if tau1 > 0.0:
                max_power = np.minimum(available / tau1, self.cfg.p_max)
                for beam_id, r in ((1, r1), (2, r2)):
                    idx = np.where(beam == beam_id)[0]
                    if idx.size == 0:
                        continue
                    if idx.size == 1:
                        i = int(idx[0])
                        power[i] = min(max_power[i], self.cfg.p_max)
                        continue
                    gains_idx = self.channel.gain[idx]
                    strong = int(idx[np.argmax(gains_idx)])
                    weak = int(idx[np.argmin(gains_idx)])
                    p_weak = min(max_power[weak], self.cfg.p_max * r)
                    p_strong = min(max_power[strong], self.cfg.p_max * (1.0 - r))
                    power[weak] = p_weak
                    power[strong] = p_strong
                power = np.clip(power, 0.0, self.cfg.p_max)
            power[beam == 0] = 0.0
            if tau1 <= 0.0:
                power[:] = 0.0

        if candidates.size == 0:
            beam = np.zeros(self.max_wd, dtype=np.int64)
            power = np.zeros(self.max_wd, dtype=np.float32)
            tau0 = tau0_agent if self.cfg.tau0_mode == "agent" else map_action_to_tau0(
                a_tau, self.cfg.tau_min, self.cfg.tau_max
            )
            tau1 = 1.0 - tau0

        if np.all(beam == 0):
            beam[:] = 0

        # Low-level solver handled in combo search above.

        harvested = harvest_energy(effective_gain, self.active, tau0, self.cfg.p_wet, self.cfg.eta)
        available = battery_before + harvested

        infeasible_penalty = 0.0
        infeasible_count = 0
        energy_penalty = 0.0
        if tau1 > 0.0:
            max_power = available / tau1
            max_power = np.minimum(max_power, self.cfg.p_max)
            over_mask = power > max_power
            if np.any(over_mask):
                energy_violation = (power[over_mask] - max_power[over_mask]) * tau1
                energy_penalty = float(np.sum(energy_violation))
                infeasible_penalty = float(np.sum(power[over_mask] - max_power[over_mask]))
                infeasible_count = int(np.sum(over_mask))
                power[over_mask] = max_power[over_mask]
        else:
            power[:] = 0.0

        rates, sinr_vals, sinr_raw_vals, inter_vals, log2_vals, pairs = compute_rates(
            active=self.active,
            beam=beam,
            power=power,
            gain=effective_gain,
            aoa=self.channel.aoa,
            tau1=tau1,
            noise=noise_power,
            inter_k=self.cfg.inter_beam_k,
            inter_scale=self.cfg.inter_beam_scale,
            sic_threshold=self.cfg.sic_threshold,
            residual_factor=self.cfg.residual_factor,
            sinr_cap=self.cfg.sinr_cap,
            noise_floor=self.cfg.noise_floor,
        )

        if self.cfg.use_solver and best_ee >= 0.0:
            sum_se = best_sum_se
        else:
            sum_se = float(np.sum(rates))
        sum_rate_bps = self.cfg.bandwidth * sum_se
        tx_energy = power * tau1
        uplink_energy = float(np.sum(tx_energy))
        power_wet = self.cfg.p_wet * tau0
        power_ul = uplink_energy
        power_c = self.cfg.p_c
        total_power = power_wet + power_ul + power_c
        ee_sum = sum_se / (total_power + 1e-9)
        ee_sum = float(np.nan_to_num(ee_sum, nan=0.0, posinf=0.0, neginf=0.0))

        tx_mask = self.active & (power > 0.0)
        n_tx_users = int(np.sum(tx_mask))

        if np.any(tx_mask):
            sinr_active_raw = sinr_raw_vals[tx_mask]
            sinr_active_clip = sinr_vals[tx_mask]
            sinr_db_list = (10.0 * np.log10(sinr_active_raw + 1e-12)).astype(np.float32)
            sinr_mean = float(np.mean(sinr_active_raw))
            sinr_p50 = float(np.percentile(sinr_active_raw, 50))
            sinr_p90 = float(np.percentile(sinr_active_raw, 90))
            log2_mean = float(np.mean(log2_vals[tx_mask]))
            p_tx_mean = float(np.mean(power[tx_mask]))
            inter_mean = float(np.mean(inter_vals[tx_mask]))
            rx_signal_mean = float(np.mean(effective_gain[tx_mask] * power[tx_mask]))
        else:
            sinr_db_list = np.array([], dtype=np.float32)
            sinr_mean = 0.0
            sinr_p50 = 0.0
            sinr_p90 = 0.0
            log2_mean = 0.0
            p_tx_mean = 0.0
            inter_mean = 0.0
            rx_signal_mean = 0.0
            sinr_active_clip = np.array([], dtype=np.float32)

        if sinr_active_clip.size > 0:
            sinr_clip_mean = float(np.mean(sinr_active_clip))
        else:
            sinr_clip_mean = 0.0
        sinr_db = 10.0 * np.log10(sinr_mean + 1e-12)
        qos_violation = 0.0
        reward_raw = ee_sum
        if self.cfg.reward_norm_mode == "log":
            reward_norm = float(np.log1p(reward_raw / max(self.cfg.reward_ref_ee, 1e-9)))
        else:
            reward_norm = float(reward_raw / max(self.cfg.reward_ref_ee, 1e-9))
        reward_norm = max(0.0, reward_norm - float(self.cfg.reward_baseline))
        reward = (
            self.cfg.reward_scale * reward_norm
            - self.cfg.mu_energy * energy_penalty
            - self.cfg.mu_infeasible * infeasible_penalty
            - self.cfg.reward_nu * qos_violation
        )
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))

        energy_after = battery_before.copy()
        if np.any(self.active):
            energy_after[self.active] = np.clip(
                available[self.active] - tx_energy[self.active], 0.0, self.cfg.e_max
            )
        self.energy = energy_after

        h_gain_mean = float(np.mean(effective_gain[self.active])) if np.any(self.active) else 0.0
        ee_avg = (sum_se / max(n_tx_users, 1)) / (total_power + 1e-9)
        se_per_tx_user = sum_se / max(n_tx_users, 1)

        info = {
            "sum_se": sum_se,
            "sum_rate_bps": sum_rate_bps,
            "ee": ee_sum,
            "ee_sum": ee_sum,
            "ee_avg": ee_avg,
            "tau0": tau0,
            "tau0_selected": tau0,
            "tau1": tau1,
            "uplink_energy": uplink_energy,
            "total_power": total_power,
            "power_wet": power_wet,
            "power_ul": power_ul,
            "power_c": power_c,
            "reward_scale": float(self.cfg.reward_scale),
            "reward_raw": reward_raw,
            "reward_norm": reward_norm,
            "reward": reward,
            "r1": r1,
            "r2": r2,
            "sinr_mean": sinr_mean,
            "sinr_mean_tx": sinr_mean,
            "sinr_db_mean_tx": sinr_db,
            "sinr_db_tx_list": sinr_db_list,
            "sinr_p50": sinr_p50,
            "sinr_p90": sinr_p90,
            "log2_mean_tx": log2_mean,
            "se_per_tx_user": se_per_tx_user,
            "h_gain_mean": h_gain_mean,
            "p_tx_mean": p_tx_mean,
            "noise_power": float(noise_power),
            "noise_power_used": float(noise_power),
            "N0_W_perHz": float(n0_w_per_hz) if self.cfg.noise_mode == "thermal" else 0.0,
            "noise_power_W": float(noise_power),
            "inter_beam_interference_mean": inter_mean,
            "interferences_mean": inter_mean,
            "rx_signal_power_mean": rx_signal_mean,
            "n_tx_users": n_tx_users,
            "scheduled_users": np.where(beam > 0)[0],
            "beam1_users": np.where(beam == 1)[0],
            "beam2_users": np.where(beam == 2)[0],
            "K": int(k_candidates),
            "M": int(sampled),
            "best_combo": best_combo,
            "best_tau0": tau0,
            "solver_used": bool(self.cfg.use_solver),
            "infeasible_penalty": infeasible_penalty,
            "infeasible_count": infeasible_count,
            "energy_penalty": energy_penalty,
            "harvested_energy": harvested,
            "tx_energy": tx_energy,
            "battery_before": battery_before,
            "battery_after": self.energy.copy(),
            "pairs": pairs,
        }

        self._apply_activity_dynamics()
        obs = self._get_obs()

        terminated = False
        truncated = self.step_count >= self.cfg.max_steps
        return obs, reward, terminated, truncated, info
