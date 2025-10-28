"""2D-Feld Φ als Python-Listen."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import Hotspot, Pulse
from .utils import gaussian_2d


class Field:
    """Repräsentiert das 2D-Feld Φ."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.phi = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
        self.imprint: list[list[dict[str, float]]] = [
            [defaultdict(float) for _ in range(shape[1])] for _ in range(shape[0])
        ]

    def relax(self, alpha: float = 0.1) -> None:
        """Einfache Diffusion mit Kreuznachbarn."""

        h, w = self.shape
        new_phi = [[0.0 for _ in range(w)] for _ in range(h)]
        new_imprint: list[list[defaultdict[str, float]]] = [
            [defaultdict(float) for _ in range(w)] for _ in range(h)
        ]
        for y in range(h):
            for x in range(w):
                neighbors = []
                neighbor_tags = []
                if y > 0:
                    neighbors.append(self.phi[y - 1][x])
                    neighbor_tags.append(self.imprint[y - 1][x])
                if y < h - 1:
                    neighbors.append(self.phi[y + 1][x])
                    neighbor_tags.append(self.imprint[y + 1][x])
                if x > 0:
                    neighbors.append(self.phi[y][x - 1])
                    neighbor_tags.append(self.imprint[y][x - 1])
                if x < w - 1:
                    neighbors.append(self.phi[y][x + 1])
                    neighbor_tags.append(self.imprint[y][x + 1])
                if neighbors:
                    avg = sum(neighbors) / len(neighbors)
                    new_phi[y][x] = self.phi[y][x] + alpha * (avg - self.phi[y][x])
                else:
                    new_phi[y][x] = self.phi[y][x]
                new_imprint[y][x] = self._diffuse_tags(
                    self.imprint[y][x], neighbor_tags, alpha
                )
        self.phi = new_phi
        self.imprint = new_imprint

    def evaporate(self, rate: float) -> None:
        factor = max(0.0, 1.0 - rate)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                self.phi[y][x] *= factor
                cell = self.imprint[y][x]
                for tag in list(cell.keys()):
                    cell[tag] *= factor
                    if cell[tag] < 1e-6:
                        del cell[tag]

    def inject_gaussian(self, pulse: Pulse) -> None:
        gauss = gaussian_2d(self.shape, pulse.position, pulse.spread, pulse.amplitude)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                self.phi[y][x] += gauss[y][x]
                if gauss[y][x] > 0.0:
                    self.imprint[y][x][pulse.tag] += gauss[y][x]

    def hotspots(self, threshold: float = 0.5) -> list[Hotspot]:
        hotspots: list[Hotspot] = []
        for y, row in enumerate(self.phi):
            for x, value in enumerate(row):
                if value >= threshold:
                    tags = dict(self.imprint[y][x])
                    hotspots.append(Hotspot(position=(y, x), value=value, tags=tags))
        return hotspots

    def _diffuse_tags(
        self,
        center: dict[str, float],
        neighbors: list[dict[str, float]],
        alpha: float,
    ) -> defaultdict[str, float]:
        if not neighbors:
            return defaultdict(float, center)
        tags = set(center.keys())
        for neighbor in neighbors:
            tags.update(neighbor.keys())
        updated: dict[str, float] = {}
        for tag in tags:
            neighbor_values = [neighbor.get(tag, 0.0) for neighbor in neighbors]
            avg = sum(neighbor_values) / len(neighbor_values)
            base = center.get(tag, 0.0)
            value = base + alpha * (avg - base)
            if value > 1e-6:
                updated[tag] = value
        return defaultdict(float, updated)


__all__ = ["Field"]
