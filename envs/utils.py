"""Utility helpers for action parsing and mapping."""
from __future__ import annotations

import numpy as np


def map_action_to_tau0(a_tau: float, tau_min: float, tau_max: float) -> float:
    a_norm = 0.5 * (float(a_tau) + 1.0)
    a_norm = float(np.clip(a_norm, 0.0, 1.0))
    return tau_min + a_norm * (tau_max - tau_min)


def map_action_to_beam(a_beam: np.ndarray) -> np.ndarray:
    beam = np.zeros_like(a_beam, dtype=np.int64)
    beam[(a_beam >= -0.33) & (a_beam < 0.33)] = 1
    beam[a_beam >= 0.33] = 2
    return beam


def map_action_to_power(a_p: np.ndarray, p_max: float) -> np.ndarray:
    a_norm = 0.5 * (a_p + 1.0)
    a_norm = np.clip(a_norm, 0.0, 1.0)
    return a_norm * p_max
