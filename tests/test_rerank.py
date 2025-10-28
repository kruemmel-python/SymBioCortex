from __future__ import annotations

import json
from pathlib import Path

from symbio.biocortex import BioCortex
from symbio.coach import (
    damerau_levenshtein,
    rerank_candidates,
    snap_tokens_to_lex,
    tune_rank_weights,
)
from symbio.coach.rerank import RankWeights
from symbio.config import DEFAULT_CONFIG


def test_damerau_distance_transposition() -> None:
    assert damerau_levenshtein("abcd", "abdc") == 1
    assert damerau_levenshtein("denken", "denken") == 0
    assert damerau_levenshtein("", "abc") == 3


def test_snap_tokens_to_lex(tmp_path: Path) -> None:
    lex = {"denken", "architektur"}
    tokens = ["Denkn", "Architekur", "ist"]
    snapped = snap_tokens_to_lex(tokens, lex, max_dist=2, keep_ratio=0.0)
    assert snapped[0].lower() == "denken"
    assert snapped[1].lower() == "architektur"


def test_rerank_prefers_closer_candidate(tmp_path: Path) -> None:
    config = DEFAULT_CONFIG.bio
    cortex = BioCortex(config=config)
    corpus = ["Hallo Welt.", "Hallo Denken."]
    cortex.partial_fit(corpus)
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("\n".join(corpus), encoding="utf-8")

    prompt = "Hallo Welt"
    candidates = ["Hallo Welt ist freundlich.", "Fremde Worte entstehen neu."]
    weights = RankWeights()
    ranked = rerank_candidates(
        prompt=prompt,
        candidates=candidates,
        kn_model=cortex.lm,
        tokenizer=cortex.tokenizer,
        corpus_path=str(corpus_path),
        weights=weights,
        snap=True,
    )
    assert ranked[0].text == candidates[0]


def test_tune_rank_weights(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    entry = {
        "prompt": "Hallo",
        "ranked": [
            {"scores": {"fluency": 0.9, "semantic": 0.8, "form": 0.7, "neo_adj": 0.5}, "text": "A"},
            {"scores": {"fluency": 0.2, "semantic": 0.4, "form": 0.6, "neo_adj": 0.9}, "text": "B"},
        ],
        "feedback": {"choice": 0, "liked": True},
    }
    entry2 = {
        "prompt": "Hallo",
        "ranked": [
            {"scores": {"fluency": 0.1, "semantic": 0.2, "form": 0.3, "neo_adj": 0.8}, "text": "C"},
            {"scores": {"fluency": 0.6, "semantic": 0.5, "form": 0.4, "neo_adj": 0.2}, "text": "D"},
        ],
        "feedback": {"choice": 0, "liked": False},
    }
    log_path.write_text("\n".join(json.dumps(e) for e in (entry, entry2)), encoding="utf-8")
    tuned = tune_rank_weights(log_path)
    assert tuned.w_fluency > 0.40
    assert tuned.w_neology < 0.10
