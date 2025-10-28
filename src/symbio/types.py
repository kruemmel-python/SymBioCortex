"""Basistypen und Dataklassen für SymBioCortex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

Coord = tuple[int, int]
Float = float


@dataclass(slots=True)
class Pulse:
    """Repräsentiert einen Puls, der in das Φ-Feld injiziert wird."""

    position: Coord
    amplitude: float
    spread: float
    tag: str


@dataclass(slots=True)
class Concept:
    """Abstraktes Konzept, abgeleitet aus Token-Sequenzen."""

    name: str
    strength: float = 1.0
    pheromone: float = 0.0


@dataclass(slots=True)
class Edge:
    """Kante im Myzel-Graphen."""

    a: int
    b: int
    weight: float = 1.0


@dataclass(slots=True)
class Hotspot:
    """Repräsentiert einen aktivierten Bereich im Φ-Feld."""

    position: Coord
    value: float
    tags: dict[str, float] = field(default_factory=dict)

    def top_tags(self, k: int = 3) -> list[tuple[str, float]]:
        """Gibt die stärksten Tags zurück."""

        return sorted(self.tags.items(), key=lambda item: item[1], reverse=True)[:k]

    def to_dict(self, top_k: int | None = None) -> dict:
        """Serialisiert den Hotspot für JSON-Ausgaben."""

        tags = self.top_tags(top_k) if top_k is not None else sorted(
            self.tags.items(), key=lambda item: item[1], reverse=True
        )
        return {
            "position": self.position,
            "value": self.value,
            "tags": [{"name": name, "strength": strength} for name, strength in tags],
        }


class Event(NamedTuple):
    """Ein Event im symbiotischen Zyklus."""

    kind: Literal["pulse", "feedback", "tick", "decay"]
    payload: Any


__all__ = [
    "Coord",
    "Float",
    "Pulse",
    "Concept",
    "Edge",
    "Hotspot",
    "Event",
]
