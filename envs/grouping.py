"""Grouping and pairing rules."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np


def assign_beams_by_aoa(aoa: np.ndarray, active: np.ndarray) -> np.ndarray:
    beam = np.zeros_like(aoa, dtype=np.int64)
    if np.any(active):
        beam[active & (aoa < 0.0)] = 1
        beam[active & (aoa >= 0.0)] = 2
    return beam


def pair_by_gain(
    gain: np.ndarray, active: np.ndarray, beam: np.ndarray
) -> Dict[int, List[Tuple[int, Optional[int]]]]:
    pairs: Dict[int, List[Tuple[int, Optional[int]]]] = {1: [], 2: []}
    for beam_id in (1, 2):
        idx = np.where(active & (beam == beam_id))[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(gain[idx])[::-1]]
        for i in range(0, order.size, 2):
            strong = int(order[i])
            weak = int(order[i + 1]) if i + 1 < order.size else None
            pairs[beam_id].append((strong, weak))
    return pairs
