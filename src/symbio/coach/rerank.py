"""Reranking und Snapping für Kandidatentexte."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from symbio.core.tokenize import detokenize, tokenize
from symbio.metrics.neology import build_corpus_lexicon, neology_ratio
from symbio.tokenization import BioBPETokenizer
from symbio.lm_kn import KneserNeyLM

from .levenshtein import damerau_levenshtein

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


@dataclass(slots=True)
class RankWeights:
    w_fluency: float = 0.40
    w_semantic: float = 0.30
    w_form: float = 0.20
    w_neology: float = 0.10
    neo_target_low: float = 0.10
    neo_target_high: float = 0.35


@dataclass(slots=True)
class RankResult:
    text: str
    scores: Dict[str, float]
    total: float


def _content_words(tokens: Sequence[str], lexicon: set[str]) -> list[str]:
    return [t.lower() for t in tokens if _WORD_RE.fullmatch(t) and len(t) > 2]


def _tfidf_probe(words: Sequence[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for word in words:
        tf[word] = tf.get(word, 0.0) + 1.0
    total = sum(tf.values()) or 1.0
    return {word: (tf[word] / total) * idf.get(word, 1.0) for word in tf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va.dot(vb) / (na * nb))


def _form_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    end_punct = stripped[-1] in ".?!"
    first_cap = stripped[0].isupper() or stripped[0] in "„("
    tokens = stripped.split()
    avg_token = sum(len(tok) for tok in tokens) / max(len(tokens), 1) if tokens else 0.0
    within = 3.0 <= avg_token <= 9.0
    penal = any(len(tok) > 28 for tok in tokens)
    score = 0.0
    score += 0.35 if end_punct else 0.0
    score += 0.35 if first_cap else 0.0
    score += 0.30 if within else 0.0
    score -= 0.20 if penal else 0.0
    return max(0.0, min(1.0, score))


def _fluency_logp(model: KneserNeyLM, tokenizer: BioBPETokenizer, text: str) -> float:
    try:
        ids = tokenizer.encode(text)
    except KeyError:
        return -1e9
    if len(ids) < 2:
        return -1e9
    order = model.order
    seq = [0] * (order - 1) + ids + [1]
    log_prob = 0.0
    for i in range(order - 1, len(seq)):
        context = tuple(seq[i - (order - 1) : i])
        token = seq[i]
        prob = model._prob_kn(context, token, order)  # type: ignore[attr-defined]
        log_prob += math.log(max(prob, 1e-12))
    return log_prob / len(ids)


def _neology_penalty(ratio: float, low: float, high: float) -> float:
    if ratio == 0 and low == 0:
        return 1.0
    if ratio < low:
        delta = (low - ratio) / max(low, 1e-6)
        return float(max(0.6, 1.0 - 0.4 * delta))
    if ratio > high:
        delta = (ratio - high) / max(1.0 - high, 1e-6)
        return float(max(0.3, 1.0 - 0.7 * delta))
    return 1.0


def _build_idf(corpus_lines: Iterable[str]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    n_docs = 0
    for line in corpus_lines:
        n_docs += 1
        words = set(word.lower() for word in _WORD_RE.findall(line))
        for word in words:
            df[word] = df.get(word, 0) + 1
    return {word: math.log((1 + n_docs) / (1 + df[word])) + 1.0 for word in df}


def snap_tokens_to_lex(tokens: Sequence[str], lexicon: set[str], max_dist: int = 2, keep_ratio: float = 0.5) -> list[str]:
    snapped: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in lexicon or len(lower) < 3 or not _WORD_RE.fullmatch(lower):
            snapped.append(token)
            continue
        if random.random() < keep_ratio:
            snapped.append(token)
            continue
        best = None
        best_dist = max_dist + 1
        for candidate in lexicon:
            if abs(len(candidate) - len(lower)) > max_dist:
                continue
            dist = damerau_levenshtein(lower, candidate)
            if dist < best_dist:
                best, best_dist = candidate, dist
                if best_dist == 0:
                    break
        snapped.append(best if best is not None and best_dist <= max_dist else token)
    return snapped


def rerank_candidates(
    prompt: str,
    candidates: Sequence[str],
    kn_model: KneserNeyLM,
    tokenizer: BioBPETokenizer,
    corpus_path: str,
    weights: RankWeights,
    snap: bool = False,
) -> list[RankResult]:
    lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    lexicon = build_corpus_lexicon(lines)
    idf = _build_idf(lines)

    prompt_tokens = tokenize(prompt)
    prompt_words = _content_words(prompt_tokens, lexicon)
    prompt_vec = _tfidf_probe(prompt_words, idf)

    results: list[RankResult] = []
    for candidate in candidates:
        text = candidate
        tokens = tokenize(text)
        if snap:
            snapped = snap_tokens_to_lex(tokens, lexicon, max_dist=2, keep_ratio=0.6)
            text = detokenize(snapped)
            tokens = tokenize(text)

        flu = _fluency_logp(kn_model, tokenizer, text)
        sem = _cosine(_tfidf_probe(_content_words(tokens, lexicon), idf), prompt_vec)
        frm = _form_score(text)
        neo_stats = neology_ratio(tokens, lexicon)
        neo_adj = _neology_penalty(neo_stats.ratio, weights.neo_target_low, weights.neo_target_high)
        centered = flu - (-4.5)
        if centered <= -100:
            flu_norm = 0.0
        elif centered >= 100:
            flu_norm = 1.0
        else:
            flu_norm = 1.0 / (1.0 + math.exp(-5.0 * centered))

        result = RankResult(
            text=text,
            scores={
                "fluency": flu_norm,
                "semantic": sem,
                "form": frm,
                "neo_ratio": neo_stats.ratio,
                "neo_adj": neo_adj,
            },
            total=(
                weights.w_fluency * flu_norm
                + weights.w_semantic * sem
                + weights.w_form * frm
                + weights.w_neology * neo_adj
            ),
        )
        results.append(result)

    results.sort(key=lambda item: item.total, reverse=True)
    return results


def export_rank_results(results: Sequence[RankResult]) -> list[dict[str, object]]:
    return [
        {
            "total": result.total,
            "scores": result.scores,
            "text": result.text,
        }
        for result in results
    ]


__all__ = [
    "RankResult",
    "RankWeights",
    "export_rank_results",
    "rerank_candidates",
    "snap_tokens_to_lex",
]

