"""WPCN energy model."""
from __future__ import annotations

import numpy as np


def harvest_energy(
    gain: np.ndarray, active: np.ndarray, tau0: float, p_wet: float, eta: float
) -> np.ndarray:
    harvested = eta * p_wet * gain * tau0
    harvested = np.where(active, harvested, 0.0)
    return harvested.astype(np.float32)


def energy_cost(power: np.ndarray, tau1: float) -> np.ndarray:
    return power * tau1
