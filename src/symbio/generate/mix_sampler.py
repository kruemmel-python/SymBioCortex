"""Hilfsfunktionen zur Mischung von Sprach- und Neologismus-Sampling."""

from __future__ import annotations

import numpy as np


def mix_probs(p_vocab: np.ndarray, p_neologism: float, neo_rate: float) -> tuple[np.ndarray, float]:
    """Mische Wahrscheinlichkeiten bekannter Tokens mit einem Neologismus-Gate."""

    neo_gate = float(np.clip(neo_rate, 0.0, 1.0))
    alpha = float(max(0.0, 1.0 - neo_gate))
    pv = (p_vocab * alpha).astype(float, copy=True)
    pv_sum = float(pv.sum())
    if pv_sum > 0.0:
        pv /= pv_sum
    neo_mass = neo_gate * float(np.clip(p_neologism, 0.0, 1.0))
    return pv, neo_mass


def sample_mixed(p_vocab: np.ndarray, neo_rate: float, rng: np.random.Generator) -> tuple[int | None, bool]:
    """Zieht entweder einen Vokabel-Index oder signalisiert einen Neologismus."""

    pv, neo_mass = mix_probs(p_vocab, p_neologism=1.0, neo_rate=neo_rate)
    r = float(rng.random())
    if r < neo_mass:
        return None, True
    if len(pv) == 0:
        return None, True
    idx = int(rng.choice(len(pv), p=pv))
    return idx, False


__all__ = ["mix_probs", "sample_mixed"]
