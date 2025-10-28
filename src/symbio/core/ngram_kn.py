"""Wrapper um den BioCortex für CLI-Sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from symbio.biocortex import BioCortex
from symbio.config import BioConfig, DEFAULT_CONFIG
from symbio.core.tokenize import detokenize


@dataclass
class KNTrigram:
    """Kleine Komfortklasse für das CLI-Sampling."""

    cortex: BioCortex

    @classmethod
    def load(cls, directory: str | Path, config: BioConfig | None = None) -> "KNTrigram":
        bio_config = config or DEFAULT_CONFIG.bio
        cortex = BioCortex.load(directory, config=bio_config)
        return cls(cortex=cortex)

    @property
    def vocab(self) -> dict[str, int]:
        return self.cortex.tokenizer.vocab

    @property
    def tokenizer(self):
        return self.cortex.tokenizer

    @property
    def lm(self):
        return self.cortex.lm

    def sample(
        self,
        prompt_tokens: Sequence[str],
        *,
        max_new: int,
        temperature: float,
        top_k: int | None = None,
        top_p: float = 0.95,
        neo_rate: float | None = None,
    ) -> str:
        """Erzeuge Text basierend auf dem Prompt."""

        prompt_text = detokenize(prompt_tokens)
        return self.cortex.generate(
            prompt_text,
            max_new_tokens=max_new,
            temperature=temperature,
            nucleus_p=top_p,
            neo_rate=neo_rate,
        )


__all__ = ["KNTrigram"]
