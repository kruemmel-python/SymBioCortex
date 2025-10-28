from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from symbio.core.tokenize import detokenize, tokenize
from symbio.core.ngram_kn import KNTrigram
from symbio.metrics.neology import build_corpus_lexicon, neology_ratio

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


def _content_words(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens if _WORD_RE.fullmatch(t) and len(t) > 2]


def _build_idf(lines: Iterable[str]) -> Dict[str, float]:
    import math

    df: Dict[str, int] = {}
    n_docs = 0
    for ln in lines:
        n_docs += 1
        for w in set(_WORD_RE.findall(ln.lower())):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((1 + n_docs) / (1 + df[w])) + 1.0 for w in df}


def _tfidf_vec(words: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for w in words:
        tf[w] = tf.get(w, 0.0) + 1.0
    n = sum(tf.values()) or 1.0
    return {w: (tf[w] / n) * idf.get(w, 1.0) for w in tf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va.dot(vb) / (na * nb))


def _form_score(text: str) -> float:
    s = text.strip()
    if not s:
        return 0.0
    end = s[-1] in ".?!"
    first_cap = s[0].isupper() or s[0] in "„("
    avg_tok = sum(len(t) for t in s.split()) / max(1, len(s.split()))
    within = 3.0 <= avg_tok <= 9.0
    penal = any(len(t) > 28 for t in s.split())
    score = 0.0
    score += 0.35 if end else 0.0
    score += 0.35 if first_cap else 0.0
    score += 0.30 if within else 0.0
    score -= 0.20 if penal else 0.0
    return max(0.0, min(1.0, score))


def _fluency_logp(model: KNTrigram, text: str) -> float:
    return model.average_log_prob(text)


def _neology_adj(ratio: float, low: float, high: float) -> float:
    if ratio < low:
        d = (low - ratio) / (low or 1e-6)
        return max(0.6, 1.0 - 0.4 * d)
    if ratio > high:
        d = (ratio - high) / max(1e-6, 1.0 - high)
        return max(0.3, 1.0 - 0.7 * d)
    return 1.0


def snap_tokens_to_lex(tokens: List[str], lexicon: set[str], max_dist: int = 2, keep_ratio: float = 0.6) -> List[str]:
    out: List[str] = []
    for t in tokens:
        w = t.lower()
        if w in lexicon or len(w) < 3 or not _WORD_RE.fullmatch(w):
            out.append(t)
            continue
        if random.random() < keep_ratio:
            out.append(t)
            continue
        best = None
        bestd = max_dist + 1
        for cand in lexicon:
            if not _WORD_RE.fullmatch(cand):
                continue
            if abs(len(cand) - len(w)) > max_dist:
                continue
            d = damerau_levenshtein(w, cand)
            if d < bestd:
                best, bestd = cand, d
                if bestd == 0:
                    break
        out.append(best if (best is not None and bestd <= max_dist) else t)
    return out


def rerank_candidates(
    prompt: str,
    candidates: List[str],
    kn_model: KNTrigram,
    corpus_path: str,
    weights: RankWeights,
    snap: bool = False,
) -> List[RankResult]:
    lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    lex = build_corpus_lexicon(lines)
    idf = _build_idf(lines)
    p_vec = _tfidf_vec(_content_words(tokenize(prompt)), idf)

    results: List[RankResult] = []
    for cand in candidates:
        text = cand
        toks = tokenize(text)
        if snap:
            snapped = snap_tokens_to_lex(toks, lex, max_dist=2, keep_ratio=0.6)
            text = detokenize(snapped)
            toks = tokenize(text)

        flu = _fluency_logp(kn_model, text)
        sem = _cosine(_tfidf_vec(_content_words(toks), idf), p_vec)
        frm = _form_score(text)
        neo_stats = neology_ratio(toks, lex)
        neo = neo_stats.ratio
        neo_adj = _neology_adj(neo, weights.neo_target_low, weights.neo_target_high)
        center = -4.5
        x = 5.0 * (flu - center)
        if x >= 60.0:
            flu_n = 1.0
        elif x <= -60.0:
            flu_n = 0.0
        else:
            flu_n = 1.0 / (1.0 + math.exp(-x))
        total = (
            weights.w_fluency * flu_n
            + weights.w_semantic * sem
            + weights.w_form * frm
            + weights.w_neology * neo_adj
        )
        results.append(
            RankResult(
                text=text,
                scores={
                    "fluency": flu_n,
                    "semantic": sem,
                    "form": frm,
                    "neo_ratio": neo,
                    "neo_adj": neo_adj,
                },
                total=total,
            )
        )
    results.sort(key=lambda r: r.total, reverse=True)
    return results


__all__ = ["RankResult", "RankWeights", "rerank_candidates", "snap_tokens_to_lex"]
