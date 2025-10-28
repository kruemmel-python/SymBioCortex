"""Einfache Konfiguration mit Standardwerten."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field


@dataclass(slots=True)
class BioConfig:
    ngram_order: int = 3
    discount: float = 0.75
    replay_capacity: int = 64
    concept_top_k: int = 8
    neo_rate: float = 0.25
    gamma_bias: float = 1.4
    hpio_coupling_gain: float = 0.1
    hpio_best_val_threshold: float = 0.01


@dataclass(slots=True)
class FieldConfig:
    shape: tuple[int, int] = (64, 64)
    relax_alpha: float = 0.1
    evaporate_rate: float = 0.01


@dataclass(slots=True)
class SwarmConfig:
    n_agents: int = 16
    boundary: str = "reflect"
    seed: int = 13


@dataclass(slots=True)
class SymbioConfig:
    bio: BioConfig = dataclass_field(default_factory=BioConfig)
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    swarm: SwarmConfig = dataclass_field(default_factory=SwarmConfig)


DEFAULT_CONFIG = SymbioConfig()


__all__ = [
    "BioConfig",
    "FieldConfig",
    "SwarmConfig",
    "SymbioConfig",
    "DEFAULT_CONFIG",
]
