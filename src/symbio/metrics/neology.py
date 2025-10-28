from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class NeologyStats:
    total: int
    novel: int
    ratio: float


def build_corpus_lexicon(lines: Iterable[str]) -> set[str]:
    lex: set[str] = set()
    for ln in lines:
        for w in ln.strip().split():
            lex.add(w.lower())
    return lex


def neology_ratio(tokens: Iterable[str], lexicon: set[str]) -> NeologyStats:
    toks = [t for t in tokens if t.strip()]
    total = len(toks)
    novel = sum(1 for t in toks if t.lower() not in lexicon)
    return NeologyStats(total, novel, (novel / total if total else 0.0))


__all__ = ["NeologyStats", "build_corpus_lexicon", "neology_ratio"]
