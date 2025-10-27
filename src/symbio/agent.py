"""Agenten-Definition ohne NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

Role = str

DEFAULT_ROLE_PARAMS = {
    "generalist": {"curiosity": 0.6, "cohesion": 0.4, "avoidance": 0.3, "deposit_sigma": 2.5},
    "scout": {"curiosity": 0.9, "cohesion": 0.2, "avoidance": 0.2, "deposit_sigma": 1.5},
    "harvester": {"curiosity": 0.3, "cohesion": 0.6, "avoidance": 0.4, "deposit_sigma": 3.0},
}


@dataclass(slots=True)
class Agent:
    """Repräsentiert einen Schwarm-Agenten."""

    position: tuple[float, float]
    velocity: tuple[float, float]
    role: Role
    battery: float
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.params:
            self.params = dict(DEFAULT_ROLE_PARAMS.get(self.role, DEFAULT_ROLE_PARAMS["generalist"]))

    def step_battery(self, drain: float = 0.01) -> None:
        self.battery = max(0.0, self.battery - drain)


__all__ = ["Agent", "DEFAULT_ROLE_PARAMS"]
