"""Deterministischer Bio-BPE Tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .utils import normalize_text, read_json, write_json


@dataclass(slots=True)
class BioBPETokenizer:
    """Einfacher Byte-Pair-Tokenizer mit deterministischen Merges."""

    merges: list[tuple[str, str]] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    def fit(self, texts: Sequence[str], vocab_size: int = 256) -> None:
        """Lerne Merge-Regeln basierend auf einem Korpus."""

        corpus = [list(normalize_text(text)) for text in texts]
        base_tokens = sorted({ch for text in corpus for ch in text} | {" ¢"})
        self.vocab = {token: idx for idx, token in enumerate(base_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.merges.clear()
        while len(self.vocab) < vocab_size:
            best_pair, freq = self._most_frequent_pair(corpus)
            if not best_pair or freq < 2:
                break
            self.merges.append(best_pair)
            merged = "".join(best_pair)
            new_token = "¤" + merged
            self.vocab[new_token] = len(self.vocab)
            self.id_to_token[self.vocab[new_token]] = new_token
            corpus = [self._merge_sequence(seq, best_pair, new_token) for seq in corpus]

    def _most_frequent_pair(self, corpus: Sequence[List[str]]) -> tuple[tuple[str, str] | None, int]:
        counts: dict[tuple[str, str], int] = {}
        for seq in corpus:
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            return None, 0
        best_pair = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        return best_pair, counts[best_pair]

    def _merge_sequence(self, seq: List[str], pair: tuple[str, str], new_token: str) -> List[str]:
        merged: list[str] = []
        skip = False
        for i, token in enumerate(seq):
            if skip:
                skip = False
                continue
            if i < len(seq) - 1 and (token, seq[i + 1]) == pair:
                merged.append(new_token)
                skip = True
            else:
                merged.append(token)
        return merged

    def encode(self, text: str) -> list[int]:
        """Kodiert Text in Token-IDs."""

        if not self.vocab:
            raise RuntimeError("Tokenizer is not fitted")
        seq = list(normalize_text(text))
        for pair in self.merges:
            new_token = self._merge_name(pair)
            seq = self._merge_sequence(seq, pair, new_token)
        return [self.vocab[token] for token in seq]

    def decode(self, ids: Iterable[int]) -> str:
        """Dekodiere IDs in Text."""

        tokens = [self.id_to_token[i] for i in ids]
        decoded = "".join(tok.replace("¤", "") for tok in tokens)
        return decoded.replace(" ¢", " ").strip()

    def to_json(self) -> dict:
        """Serialisiere den Tokenizer."""

        return {"merges": self.merges, "vocab": self.vocab}

    @classmethod
    def from_json(cls, data: dict) -> "BioBPETokenizer":
        """Lade den Tokenizer aus JSON-Daten."""

        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in data.get("merges", [])]
        tokenizer.vocab = {str(k): int(v) for k, v in data.get("vocab", {}).items()}
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.vocab.items()}
        return tokenizer

    def save(self, path: str) -> None:
        """Speichere den Tokenizer als JSON."""

        write_json(path, self.to_json())

    @classmethod
    def load(cls, path: str) -> "BioBPETokenizer":
        """Lade einen Tokenizer von einer Datei."""

        data = read_json(path)
        return cls.from_json(data)

    def _merge_name(self, pair: tuple[str, str]) -> str:
        return "¤" + "".join(pair)


__all__ = ["BioBPETokenizer"]
