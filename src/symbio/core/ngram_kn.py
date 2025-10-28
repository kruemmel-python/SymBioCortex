"""Wrapper um den BioCortex für CLI-Sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from symbio.biocortex import BioCortex
from symbio.config import BioConfig, DEFAULT_CONFIG
from symbio.core.tokenize import detokenize, tokenize


@dataclass(slots=True)
class KNTrigram:
    cortex: BioCortex

    @classmethod
    def load(cls, directory: str | Path, config: BioConfig | None = None) -> "KNTrigram":
        bio_config = config or DEFAULT_CONFIG.bio
        cortex = BioCortex.load(directory, config=bio_config)
        return cls(cortex=cortex)

    def sample(
        self,
        prompt_tokens: Sequence[str],
        *,
        max_new: int,
        temperature: float,
        top_k: int | None = None,
        top_p: float = 0.95,
        neo_rate: float | None = None,
    ) -> list[str]:
        prompt_text = detokenize(prompt_tokens)
        generated = self.cortex.generate(
            prompt_text,
            max_new_tokens=max_new,
            temperature=temperature,
            nucleus_p=top_p,
            neo_rate=neo_rate,
        )
        return tokenize(generated)

    def average_log_prob(self, text: str) -> float:
        try:
            ids = self.cortex.tokenizer.encode(text)
        except KeyError:
            return -1e9
        if len(ids) < 2:
            return -1e9
        order = self.cortex.lm.order
        seq = [0] * (order - 1) + ids + [1]
        log_prob = 0.0
        for i in range(order - 1, len(seq)):
            context = tuple(seq[i - (order - 1) : i])
            token = seq[i]
            prob = self.cortex.lm._prob_kn(context, token, order)  # type: ignore[attr-defined]
            log_prob += math.log(max(prob, 1e-12))
        return log_prob / len(ids)


__all__ = ["KNTrigram"]
