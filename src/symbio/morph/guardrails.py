"""Sanfte Leitplanken für morphologische Generatoren."""

from __future__ import annotations

import random
import re
from typing import Callable

_BAD_START = re.compile(r"^(ng|tsc|pfh|q[bcdfghjklmnpqrstvwxyz])", re.I)
_REPEAT4 = re.compile(r"(.)\1\1\1", re.I)

_PREFIX = ["ver", "be", "ent", "zer", "ur", "um", "miss"]
_SUFFIX = ["ung", "heit", "keit", "isch", "ieren", "bar", "haft", "sam", "los", "frei"]


def good_shape(word: str) -> bool:
    """Prüfe einfache Wohlgeformtheits-Kriterien."""

    if len(word) < 3 or len(word) > 24:
        return False
    if _BAD_START.search(word):
        return False
    if _REPEAT4.search(word):
        return False
    return True


def affix_boost(word: str, rnd: random.Random) -> str:
    """Gewichte gängige deutsche Affixe leicht nach oben."""

    if rnd.random() < 0.30:
        word = rnd.choice(_PREFIX) + word
    if rnd.random() < 0.40:
        word = word + rnd.choice(_SUFFIX)
    return word


def morph_wrapper(base_generator: Callable[[random.Random], str]) -> Callable[[random.Random], str]:
    """Verpacke einen Generator mit weichen Guardrails."""

    def wrapped(rnd: random.Random) -> str:
        for _ in range(10):
            candidate = base_generator(rnd)
            candidate = affix_boost(candidate, rnd)
            if good_shape(candidate):
                return candidate
        return base_generator(rnd)

    return wrapped


__all__ = ["affix_boost", "good_shape", "morph_wrapper"]
