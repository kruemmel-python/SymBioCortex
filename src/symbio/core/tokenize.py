"""Einfache Tokenisierungs-Helfer."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"\w+|[-\u2013\u2014]+|[.,!?;:()\[\]\{\}‹›«»„“‚‘\"']")

def tokenize(text: str) -> list[str]:
    """Zerlegt Text in eine einfache Tokenfolge."""

    return _TOKEN_RE.findall(text.strip()) if text else []

def detokenize(tokens: Sequence[str]) -> str:
    """Setzt Tokens wieder zu einem lesbaren Text zusammen."""

    out: list[str] = []
    for token in tokens:
        if not out:
            out.append(token)
            continue
        if re.fullmatch(r"[.,!?;:)\]\}]", token):
            out[-1] += token
        elif token in {"'", '\"', "“", "”"} and out[-1].endswith(token):
            out[-1] += token
        elif re.fullmatch(r"[-\u2013\u2014]+", token):
            out[-1] += token
        else:
            out.append(" " + token)
    return "".join(out)

def join_tokens(tokens: Iterable[str]) -> str:
    """Hilfsfunktion für Guardrails und Reranking."""

    return detokenize(list(tokens))


__all__ = ["detokenize", "join_tokens", "tokenize"]
