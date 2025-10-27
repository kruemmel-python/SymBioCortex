"""Denken→Handeln Brücke."""

from __future__ import annotations

from typing import Sequence

from .biocortex import BioCortex
from .types import Pulse


def text_to_pulses(biocortex: BioCortex, text: str, field_shape: tuple[int, int]) -> list[Pulse]:
    """Leitet Text über Konzepte zu Puls-Events weiter."""

    concepts = biocortex.extract_concepts(text)
    pulses = biocortex.concepts_to_pulses(concepts, field_shape)
    for concept, pulse in zip(concepts, pulses):
        pulse.amplitude *= 1.0 + concept.pheromone
        pulse.spread = max(1.0, pulse.spread)
    return pulses


__all__ = ["text_to_pulses"]
