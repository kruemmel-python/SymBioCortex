"""Neuromodulatorische Signale."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NeuromodulatorState:
    """Hält modulierte Bias-Faktoren."""

    dopamine: float = 1.0
    serotonin: float = 1.0
    acetylcholine: float = 1.0

    def apply_reward(self, value: float) -> None:
        """Passe Dopamin basierend auf Belohnung an."""

        self.dopamine = max(0.1, min(5.0, self.dopamine + value))
        logger.debug("Dopamine updated to %.3f", self.dopamine)

    def apply_surprise(self, value: float) -> None:
        """Passe Serotonin bei Überraschung an."""

        self.serotonin = max(0.1, min(5.0, self.serotonin + value))
        logger.debug("Serotonin updated to %.3f", self.serotonin)

    def decay(self, rate: float = 0.01) -> None:
        """Langsame Rückkehr zum Baseline."""

        for attr in ("dopamine", "serotonin", "acetylcholine"):
            current = getattr(self, attr)
            updated = current + (1.0 - current) * rate
            setattr(self, attr, updated)
        logger.debug("Neuromodulatoren decayed to %s", self)


__all__ = ["NeuromodulatorState"]
