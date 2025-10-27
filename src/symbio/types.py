"""Basistypen und Dataklassen für SymBioCortex."""

from __future__ import annotations

from dataclasses import dataclass
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
    "Event",
]
