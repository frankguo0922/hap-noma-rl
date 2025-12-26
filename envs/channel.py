"""Channel and geometry model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChannelConfig:
    d_min: float = 30.0
    d_max: float = 200.0
    pathloss_exp: float = 2.2
    beta0: float = 1e-3
    d0: float = 1.0
    gain_scale: float = 1.0
    ar_coef: float = 0.0
    aoa_jitter_std: float = 0.0


class ChannelModel:
    def __init__(self, n_wd: int, cfg: ChannelConfig, rng: np.random.Generator):
        self.n_wd = n_wd
        self.cfg = cfg
        self.rng = rng
        self.distance = np.zeros(n_wd, dtype=np.float32)
        self.aoa = np.zeros(n_wd, dtype=np.float32)
        self.gain = np.zeros(n_wd, dtype=np.float32)

    def _pathloss(self, distance: np.ndarray) -> np.ndarray:
        return self.cfg.beta0 * (distance / self.cfg.d0) ** (-self.cfg.pathloss_exp)

    def _small_scale(self, size: int) -> np.ndarray:
        return self.rng.exponential(scale=1.0, size=size).astype(np.float32)

    def sample_positions(self, idx: np.ndarray) -> None:
        self.distance[idx] = self.rng.uniform(self.cfg.d_min, self.cfg.d_max, size=idx.size)
        self.aoa[idx] = self.rng.uniform(-np.pi, np.pi, size=idx.size)

    def sample_gain(self, idx: np.ndarray) -> None:
        pathloss = self._pathloss(self.distance[idx])
        fading = self._small_scale(idx.size)
        new_gain = self.cfg.gain_scale * pathloss * fading
        if self.cfg.ar_coef > 0.0:
            self.gain[idx] = (
                self.cfg.ar_coef * self.gain[idx] + (1.0 - self.cfg.ar_coef) * new_gain
            )
        else:
            self.gain[idx] = new_gain

    def update(self, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        if self.cfg.aoa_jitter_std > 0.0:
            self.aoa[idx] += self.rng.normal(0.0, self.cfg.aoa_jitter_std, size=idx.size)
            self.aoa[idx] = np.clip(self.aoa[idx], -np.pi, np.pi)
        self.sample_gain(idx)
