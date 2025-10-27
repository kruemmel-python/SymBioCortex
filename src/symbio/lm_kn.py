"""Einfaches Kneser-Ney-N-Gramm-Sprachmodell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .utils import read_json, write_json


@dataclass(slots=True)
class KneserNeyLM:
    """Implementierung eines diskontierten Kneser-Ney-Modells."""

    order: int = 3
    discount: float = 0.75
    counts: list[dict[tuple[int, ...], dict[int, int]]] = field(default_factory=list)
    continuation: dict[int, set[tuple[int, ...]]] = field(default_factory=dict)
    vocabulary: set[int] = field(default_factory=set)

    def train_sequences(self, sequences: Sequence[Sequence[int]]) -> None:
        """Trainiert das Modell anhand von Sequenzen von Token-IDs."""

        self.counts = [dict() for _ in range(self.order)]
        self.continuation = {}
        self.vocabulary = set()
        bos = 0
        eos = 1
        for seq in sequences:
            tokens = [bos] * (self.order - 1) + list(seq) + [eos]
            self.vocabulary.update(tokens)
            for i in range(len(tokens)):
                for n in range(1, self.order + 1):
                    if i + n > len(tokens):
                        break
                    ngram = tuple(tokens[i : i + n])
                    context, token = ngram[:-1], ngram[-1]
                    context_dict = self.counts[n - 1].setdefault(context, {})
                    context_dict[token] = context_dict.get(token, 0) + 1
                    if n > 1:
                        self.continuation.setdefault(token, set()).add(context)
        for token in self.vocabulary:
            self.continuation.setdefault(token, set())

    def prob_next(self, context: Sequence[int], candidates: Iterable[int] | None = None) -> dict[int, float]:
        """Gibt eine Verteilung über das nächste Token zurück."""

        if not self.counts:
            raise RuntimeError("model not trained")
        context = tuple(context)[-(self.order - 1) :]
        if candidates is None:
            candidates = self.vocabulary
        probs = {token: self._prob_kn(tuple(context), token, self.order) for token in candidates}
        total = sum(probs.values())
        if total <= 0:
            uniform = 1.0 / max(len(probs), 1)
            return {token: uniform for token in probs}
        return {token: value / total for token, value in probs.items()}

    def _prob_kn(self, context: tuple[int, ...], token: int, order: int) -> float:
        if order == 1:
            total_contexts = sum(len(ctxs) for ctxs in self.continuation.values()) or 1
            return len(self.continuation.get(token, ())) / total_contexts
        context_counts = self.counts[order - 1].get(context, {})
        count = context_counts.get(token, 0)
        total = sum(context_counts.values())
        if total == 0:
            backoff_weight = 1.0
        else:
            unique_followers = len(context_counts)
            backoff_weight = (self.discount * unique_followers) / total
        lower = self._prob_kn(context[1:], token, order - 1)
        return max(count - self.discount, 0) / max(total, 1) + backoff_weight * lower

    def to_json(self) -> dict:
        """Serialisiert das Modell."""

        counts_serialized: list[dict[str, dict[str, int]]] = []
        for level in self.counts:
            level_ser: dict[str, dict[str, int]] = {}
            for context, tokens in level.items():
                key = ",".join(map(str, context))
                level_ser[key] = {str(tok): cnt for tok, cnt in tokens.items()}
            counts_serialized.append(level_ser)
        continuation_ser = {
            str(tok): [",".join(map(str, ctx)) for ctx in contexts] for tok, contexts in self.continuation.items()
        }
        return {
            "order": self.order,
            "discount": self.discount,
            "counts": counts_serialized,
            "continuation": continuation_ser,
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_json(cls, data: dict) -> "KneserNeyLM":
        model = cls(order=int(data["order"]), discount=float(data["discount"]))
        model.counts = []
        for level in data["counts"]:
            restored: dict[tuple[int, ...], dict[int, int]] = {}
            for context_str, tokens in level.items():
                context = tuple(int(x) for x in context_str.split(",") if x)
                restored[context] = {int(tok): int(cnt) for tok, cnt in tokens.items()}
            model.counts.append(restored)
        model.continuation = {
            int(tok): {tuple(int(x) for x in ctx.split(",") if x) for ctx in contexts}
            for tok, contexts in data["continuation"].items()
        }
        model.vocabulary = set(int(v) for v in data["vocabulary"])
        return model

    def save(self, path: str) -> None:
        write_json(path, self.to_json())

    @classmethod
    def load(cls, path: str) -> "KneserNeyLM":
        return cls.from_json(read_json(path))


__all__ = ["KneserNeyLM"]
