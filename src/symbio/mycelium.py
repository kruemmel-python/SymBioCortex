"""Myzel-Graph mit STDP-ähnlichen Updates."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Sequence

from .types import Edge


@dataclass(slots=True)
class MyceliumGraph:
    """Gerichteter Graph zwischen Token-IDs."""

    weights: dict[tuple[int, int], float] = field(default_factory=dict)
    pheromones: dict[tuple[int, int], float] = field(default_factory=dict)
    a_plus: float = 0.1
    a_minus: float = 0.05
    decay: float = 0.01
    rng_seed: int = 7

    def update_edge(self, edge: Edge, pre: float, post: float) -> None:
        """Aktualisiere eine Kante basierend auf STDP."""

        delta = self.a_plus * pre * post - self.a_minus * self.decay
        key = (edge.a, edge.b)
        self.weights[key] = max(self.weights.get(key, 0.0) + delta, 0.0)
        self.pheromones[key] = max(self.pheromones.get(key, 0.0) + post, 0.0)

    def evaporate(self, rate: float) -> None:
        """Verdunste Pheromone und Gewichte leicht."""

        for mapping in (self.weights, self.pheromones):
            for key in list(mapping.keys()):
                mapping[key] *= max(0.0, 1.0 - rate)
                if mapping[key] < 1e-6:
                    del mapping[key]

    def reinforce(self, path: Sequence[int], amount: float = 1.0) -> None:
        """Verstärke einen Pfad proportionale zu amount."""

        for a, b in zip(path, path[1:]):
            key = (a, b)
            self.weights[key] = self.weights.get(key, 0.0) + amount
            self.pheromones[key] = self.pheromones.get(key, 0.0) + amount

    def top_k_successors(self, node: int, k: int = 3) -> list[int]:
        """Gibt die Top-K Nachfolger eines Knotens zurück."""

        successors = [
            (b, self.weights.get((node, b), 0.0) + self.pheromones.get((node, b), 0.0))
            for (a, b) in self.weights
            if a == node
        ]
        successors.sort(key=lambda item: item[1], reverse=True)
        return [b for b, _ in successors[:k]]

    def random_walk(self, seed: int, steps: int = 8, pher_bias: float = 1.0) -> list[int]:
        """Führe einen Pheromon-basierten Random-Walk aus."""

        rng = random.Random(self.rng_seed + seed)
        if not self.weights:
            return []
        start = rng.choice(list({a for a, _ in self.weights}))
        path = [start]
        current = start
        for _ in range(steps):
            options = [b for (a, b) in self.weights if a == current]
            if not options:
                break
            scores = [self.pheromones.get((current, b), 0.0) ** pher_bias + 1e-6 for b in options]
            total = sum(scores)
            probs = [score / total for score in scores]
            cumulative = 0.0
            pick = rng.random()
            for option, prob in zip(options, probs):
                cumulative += prob
                if pick <= cumulative:
                    current = option
                    path.append(option)
                    break
        return path


__all__ = ["MyceliumGraph"]
