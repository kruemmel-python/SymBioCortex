"""Allgemeine Hilfsfunktionen ohne externe Abhängigkeiten."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    scale = max(temperature, 1e-6)
    shifted = [v / scale for v in values]
    max_val = max(shifted)
    exps = [math.exp(v - max_val) for v in shifted]
    total = sum(exps) or 1.0
    return [val / total for val in exps]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def deterministic_choice(options: Sequence[int], probs: Sequence[float], seed: int | None = None) -> int:
    rng = random.Random(seed)
    threshold = rng.random()
    cumulative = 0.0
    for option, prob in zip(options, probs):
        cumulative += prob
        if threshold <= cumulative:
            return option
    return options[-1]


def gaussian_2d(shape: tuple[int, int], center: tuple[float, float], sigma: float, amplitude: float) -> list[list[float]]:
    h, w = shape
    cy, cx = center
    sigma2 = max(sigma, 1e-3) ** 2
    field = []
    for y in range(h):
        row = []
        for x in range(w):
            dy = y - cy
            dx = x - cx
            dist2 = (dy * dy + dx * dx) / (2 * sigma2)
            row.append(amplitude * math.exp(-dist2))
        field.append(row)
    return field


def write_json(path: Path | str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def moving_average(values: Iterable[float], window: int) -> list[float]:
    seq = list(values)
    if not seq:
        return []
    window = max(1, window)
    result: list[float] = []
    for i in range(len(seq)):
        start = max(0, i - window + 1)
        segment = seq[start : i + 1]
        result.append(sum(segment) / len(segment))
    return result


__all__ = [
    "ensure_dir",
    "softmax",
    "normalize_text",
    "deterministic_choice",
    "gaussian_2d",
    "write_json",
    "read_json",
    "moving_average",
]
