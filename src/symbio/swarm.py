"""Schwarm-Logik ohne NumPy."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

from .agent import Agent, DEFAULT_ROLE_PARAMS
from .field import Field
from .types import Pulse

logger = logging.getLogger(__name__)


def vec_add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def vec_sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def vec_scale(a: tuple[float, float], s: float) -> tuple[float, float]:
    return (a[0] * s, a[1] * s)


def vec_length_squared(a: tuple[float, float]) -> float:
    return a[0] * a[0] + a[1] * a[1]


@dataclass(slots=True)
class Swarm:
    """Steuert Agenten auf dem Φ-Feld."""

    field: Field
    n_agents: int
    boundary: str = "reflect"
    seed: int = 0
    agents: list[Agent] = field(init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        roles = list(DEFAULT_ROLE_PARAMS)
        h, w = self.field.shape
        self.agents = []
        for i in range(self.n_agents):
            position = (rng.random() * (h - 1), rng.random() * (w - 1))
            velocity = (0.0, 0.0)
            role = roles[i % len(roles)]
            self.agents.append(Agent(position=position, velocity=velocity, role=role, battery=1.0))

    def _bilinear_sample(self, position: tuple[float, float]) -> float:
        y, x = position
        h, w = self.field.shape
        y0 = max(0, min(int(math.floor(y)), h - 1))
        x0 = max(0, min(int(math.floor(x)), w - 1))
        y1 = min(y0 + 1, h - 1)
        x1 = min(x0 + 1, w - 1)
        dy = y - y0
        dx = x - x0
        phi = self.field.phi
        return (
            phi[y0][x0] * (1 - dy) * (1 - dx)
            + phi[y1][x0] * dy * (1 - dx)
            + phi[y0][x1] * (1 - dy) * dx
            + phi[y1][x1] * dy * dx
        )

    def _gradient(self, position: tuple[float, float], epsilon: float = 1.0) -> tuple[float, float]:
        base = self._bilinear_sample(position)
        gy_pos = self._bilinear_sample(self._wrap_position((position[0] + epsilon, position[1])))
        gx_pos = self._bilinear_sample(self._wrap_position((position[0], position[1] + epsilon)))
        return ((gy_pos - base) / epsilon, (gx_pos - base) / epsilon)

    def _wrap_position(self, position: tuple[float, float]) -> tuple[float, float]:
        y, x = position
        h, w = self.field.shape
        if self.boundary == "reflect":
            if y < 0:
                y = -y
            if x < 0:
                x = -x
            if y >= h:
                y = 2 * (h - 1) - y
            if x >= w:
                x = 2 * (w - 1) - x
        elif self.boundary == "periodic":
            y = y % h
            x = x % w
        else:
            y = max(0.0, min(y, h - 1))
            x = max(0.0, min(x, w - 1))
        return (y, x)

    def step(self, dt: float = 1.0) -> dict:
        trails: list[tuple[tuple[float, float], str]] = []
        if not self.agents:
            return {"trails": [], "center": (0.0, 0.0), "mean_battery": 0.0}
        center_y = sum(agent.position[0] for agent in self.agents) / len(self.agents)
        center_x = sum(agent.position[1] for agent in self.agents) / len(self.agents)
        center = (center_y, center_x)
        for idx, agent in enumerate(self.agents):
            grad = self._gradient(agent.position)
            curiosity = vec_scale(grad, agent.params["curiosity"])
            cohesion = vec_scale(vec_sub(center, agent.position), agent.params["cohesion"])
            avoidance = (0.0, 0.0)
            for other in self.agents:
                if other is agent:
                    continue
                delta = vec_sub(agent.position, other.position)
                dist2 = vec_length_squared(delta)
                if dist2 < 4.0:
                    scale = agent.params["avoidance"] / max(dist2, 1e-3)
                    avoidance = vec_add(avoidance, vec_scale(delta, scale))
            rng = random.Random(self.seed + idx)
            noise = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
            velocity = vec_add(vec_scale(agent.velocity, 0.5), curiosity)
            velocity = vec_add(velocity, cohesion)
            velocity = vec_add(velocity, avoidance)
            velocity = vec_add(velocity, noise)
            agent.velocity = velocity
            agent.position = self._wrap_position(vec_add(agent.position, vec_scale(agent.velocity, dt)))
            agent.step_battery()
            sigma = agent.params["deposit_sigma"]
            pulse = Pulse(
                position=(int(agent.position[0]), int(agent.position[1])),
                amplitude=agent.battery,
                spread=sigma,
                tag=f"agent:{agent.role}",
            )
            self.field.inject_gaussian(pulse)
            trails.append(((agent.position[0], agent.position[1]), agent.role))
        mean_battery = sum(agent.battery for agent in self.agents) / len(self.agents)
        return {"trails": trails, "center": center, "mean_battery": mean_battery}


__all__ = ["Swarm"]
