"""Feedback-basierte Gewichtsanpassung für das Reranking."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .rerank import RankWeights


def _collect_feedback(log_path: Path) -> Iterable[tuple[dict[str, float], bool]]:
    if not log_path.exists():
        return []
    records: list[tuple[dict[str, float], bool]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        feedback = payload.get("feedback")
        ranked = payload.get("ranked")
        if not isinstance(ranked, list) or not ranked:
            continue
        choice = 0
        liked = None
        if isinstance(feedback, dict):
            choice = int(feedback.get("choice", 0))
            like_val = feedback.get("liked")
            if isinstance(like_val, bool):
                liked = like_val
        if liked is None:
            continue
        if choice < 0 or choice >= len(ranked):
            continue
        scores = ranked[choice].get("scores")
        if not isinstance(scores, dict):
            continue
        records.append((scores, liked))
    return records


def tune_rank_weights(log_path: str | Path, base: RankWeights | None = None, step: float = 0.05) -> RankWeights:
    base = base or RankWeights()
    records = list(_collect_feedback(Path(log_path)))
    if not records:
        return base
    pos: dict[str, list[float]] = {"fluency": [], "semantic": [], "form": [], "neo_adj": []}
    neg: dict[str, list[float]] = {"fluency": [], "semantic": [], "form": [], "neo_adj": []}
    for scores, liked in records:
        target = pos if liked else neg
        for key in target:
            value = float(scores.get(key, 0.0))
            target[key].append(value)
    mapping = {
        "fluency": "w_fluency",
        "semantic": "w_semantic",
        "form": "w_form",
        "neo_adj": "w_neology",
    }
    updated = asdict(base)
    total = 0.0
    for score_key, weight_key in mapping.items():
        base_val = float(updated.get(weight_key, 0.0))
        pos_avg = sum(pos[score_key]) / len(pos[score_key]) if pos[score_key] else 0.0
        neg_avg = sum(neg[score_key]) / len(neg[score_key]) if neg[score_key] else 0.0
        delta = step * (pos_avg - neg_avg)
        new_val = max(0.0, min(1.0, base_val + delta))
        updated[weight_key] = new_val
        total += new_val
    if total > 0:
        for key in mapping.values():
            updated[key] = updated[key] / total
    return RankWeights(
        w_fluency=updated["w_fluency"],
        w_semantic=updated["w_semantic"],
        w_form=updated["w_form"],
        w_neology=updated["w_neology"],
        neo_target_low=base.neo_target_low,
        neo_target_high=base.neo_target_high,
    )


__all__ = ["tune_rank_weights"]
