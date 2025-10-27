"""HPIO-Fassade ohne NumPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field

from .config import FieldConfig, SwarmConfig
from .field import Field
from .swarm import Swarm
from .types import Pulse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HPIO:
    """Verbindet Feld und Schwarm."""

    field_config: FieldConfig = dataclass_field(default_factory=FieldConfig)
    swarm_config: SwarmConfig = dataclass_field(default_factory=SwarmConfig)
    field: Field = dataclass_field(init=False)
    swarm: Swarm = dataclass_field(init=False)
    best_pos: tuple[int, int] | None = None
    best_val: float = float("-inf")

    def __post_init__(self) -> None:
        self.field = Field(self.field_config.shape)
        self.swarm = Swarm(
            field=self.field,
            n_agents=self.swarm_config.n_agents,
            boundary=self.swarm_config.boundary,
            seed=self.swarm_config.seed,
        )

    def inject_pulses(self, pulses: list[Pulse]) -> None:
        for pulse in pulses:
            self.field.inject_gaussian(pulse)

    def step(self) -> dict:
        metrics = self.swarm.step()
        best_val = float("-inf")
        best_pos = None
        for y, row in enumerate(self.field.phi):
            for x, value in enumerate(row):
                if value > best_val:
                    best_val = value
                    best_pos = (y, x)
        if best_pos and best_val > self.best_val:
            self.best_pos = best_pos
            self.best_val = best_val
        metrics.update({"best_pos": self.best_pos, "best_val": self.best_val})
        return metrics

    def relax_and_evaporate(self) -> None:
        self.field.relax(self.field_config.relax_alpha)
        self.field.evaporate(self.field_config.evaporate_rate)

    def polish(self, radius: int = 2) -> None:
        if self.best_pos is None:
            return
        y, x = self.best_pos
        h, w = self.field.shape
        for yy in range(max(0, y - radius), min(h, y + radius + 1)):
            for xx in range(max(0, x - radius), min(w, x + radius + 1)):
                self.field.phi[yy][xx] += self.field.phi[y][x] * 0.05


__all__ = ["HPIO"]
