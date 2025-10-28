"""Utilities zur Messung von Neologismen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class NeologyStats:
    """Einfache Statistik über neuartige Tokens."""

    total: int
    novel: int
    ratio: float  # novel / total


def build_corpus_lexicon(lines: Iterable[str]) -> set[str]:
    """Erzeuge eine einfache Wortliste aus einem Korpus."""

    lex: set[str] = set()
    for ln in lines:
        for word in ln.strip().split():
            token = word.lower().strip()
            if token:
                lex.add(token)
    return lex


def neology_ratio(tokens: Iterable[str], lexicon: set[str]) -> NeologyStats:
    """Bestimme Anteil und Anzahl neuer Tokens relativ zu einem Lexikon."""

    toks = [t for t in tokens if t.strip()]
    total = len(toks)
    novel = sum(1 for t in toks if t.lower() not in lexicon)
    return NeologyStats(total=total, novel=novel, ratio=(novel / total if total else 0.0))


__all__ = ["NeologyStats", "build_corpus_lexicon", "neology_ratio"]
