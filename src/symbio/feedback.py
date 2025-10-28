"""Handeln→Denken Rückkopplung."""

from __future__ import annotations

from typing import Iterable

from .biocortex import BioCortex
from .field import Field
from .types import Hotspot


def detect_hotspots(field: Field, threshold: float = 0.6) -> list[Hotspot]:
    """Ermittle Hotspots im Feld."""

    return field.hotspots(threshold)


def apply_feedback(biocortex: BioCortex, hotspots: Iterable[Hotspot]) -> dict:
    """Verstärkt Myzel-Kanten basierend auf Hotspots."""

    reinforcements = []
    for hotspot in hotspots:
        y, x = hotspot.position
        value = hotspot.value
        edges = sorted(
            biocortex.graph.pheromones.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not edges:
            continue
        ((a, b), pher) = edges[0]
        amount = value * (1.0 + biocortex.neuromod.dopamine)
        biocortex.graph.reinforce([a, b], amount=amount)
        biocortex.replay.add([a, b])
        biocortex.neuromod.apply_reward(value * 0.1)
        reinforcements.append(((a, b), amount))
    return {"reinforcements": reinforcements, "count": len(reinforcements)}


__all__ = ["detect_hotspots", "apply_feedback"]
