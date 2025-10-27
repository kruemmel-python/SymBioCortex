"""Replay-Puffer für Konsolidierung."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfung
    from .lm_kn import KneserNeyLM


@dataclass(slots=True)
class ReplayBuffer:
    """FIFO-Puffer für Sequenzen."""

    capacity: int = 64
    buffer: Deque[list[int]] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.capacity)

    def add(self, sequence: Sequence[int]) -> None:
        self.buffer.append(list(sequence))

    def sample(self, n: int) -> list[list[int]]:
        seqs = list(self.buffer)
        if not seqs:
            return []
        rng = random.Random(42)
        return [rng.choice(seqs) for _ in range(min(n, len(seqs)))]

    def consolidate(self, model: "KneserNeyLM", n_steps: int = 4) -> None:
        """Füttere das Modell mit Replay-Sequenzen."""

        samples = self.sample(n_steps)
        if samples:
            model.train_sequences(samples)


__all__ = ["ReplayBuffer"]
