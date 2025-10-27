"""2D-Feld Φ als Python-Listen."""

from __future__ import annotations

from typing import Iterable

from .types import Pulse
from .utils import gaussian_2d


class Field:
    """Repräsentiert das 2D-Feld Φ."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.phi = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]

    def relax(self, alpha: float = 0.1) -> None:
        """Einfache Diffusion mit Kreuznachbarn."""

        h, w = self.shape
        new_phi = [[0.0 for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                neighbors = []
                if y > 0:
                    neighbors.append(self.phi[y - 1][x])
                if y < h - 1:
                    neighbors.append(self.phi[y + 1][x])
                if x > 0:
                    neighbors.append(self.phi[y][x - 1])
                if x < w - 1:
                    neighbors.append(self.phi[y][x + 1])
                if neighbors:
                    avg = sum(neighbors) / len(neighbors)
                    new_phi[y][x] = self.phi[y][x] + alpha * (avg - self.phi[y][x])
                else:
                    new_phi[y][x] = self.phi[y][x]
        self.phi = new_phi

    def evaporate(self, rate: float) -> None:
        factor = max(0.0, 1.0 - rate)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                self.phi[y][x] *= factor

    def inject_gaussian(self, pulse: Pulse) -> None:
        gauss = gaussian_2d(self.shape, pulse.position, pulse.spread, pulse.amplitude)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                self.phi[y][x] += gauss[y][x]

    def hotspots(self, threshold: float = 0.5) -> list[tuple[int, int, float]]:
        return [
            (y, x, value)
            for y, row in enumerate(self.phi)
            for x, value in enumerate(row)
            if value >= threshold
        ]


__all__ = ["Field"]
