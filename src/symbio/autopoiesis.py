"""Autopoietische Generierung neuer Sätze aus Feld-Hotspots."""

from __future__ import annotations

from typing import Iterable, Sequence

from .biocortex import BioCortex
from .types import Hotspot


def _compose_prompt(hotspot: Hotspot, top_k: int) -> str | None:
    tags = [
        name.strip()
        for name, _ in hotspot.top_tags(top_k)
        if name.strip() and ":" not in name
    ]
    if not tags:
        return None
    return " ".join(tags)


def synthesize_thoughts(
    biocortex: BioCortex,
    hotspots: Sequence[Hotspot],
    *,
    max_sentences: int = 5,
    top_k_tags: int = 3,
    max_new_tokens: int = 48,
) -> list[str]:
    """Erzeugt neue Sätze aus den stärksten Feldreaktionen."""

    if not hotspots:
        return []
    ranked = sorted(hotspots, key=lambda spot: spot.value, reverse=True)
    sentences: list[str] = []
    seen_prompts: set[str] = set()
    for hotspot in ranked:
        prompt = _compose_prompt(hotspot, top_k_tags)
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        text = biocortex.generate(prompt, max_new_tokens=max_new_tokens)
        sentences.append(text.strip())
        if len(sentences) >= max_sentences:
            break
    return sentences


def aggregate_prompts(hotspots: Iterable[Hotspot], top_k_tags: int = 3) -> list[str]:
    """Hilfsfunktion für Debugging und Analyse."""

    prompts: list[str] = []
    for hotspot in hotspots:
        prompt = _compose_prompt(hotspot, top_k_tags)
        if prompt:
            prompts.append(prompt)
    return prompts


__all__ = ["aggregate_prompts", "synthesize_thoughts"]
