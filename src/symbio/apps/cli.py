"""Kommandozeileninterface für SymBioCortex."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from symbio.coach.rerank import RankWeights, export_rank_results, rerank_candidates
from symbio.coach.tuner import tune_rank_weights
from symbio.core.ngram_kn import KNTrigram
from symbio.core.tokenize import tokenize

from symbio.biocortex import BioCortex
from symbio.config import DEFAULT_CONFIG, SymbioConfig
from symbio.hpio import HPIO
from symbio.logging_setup import configure_logging
from symbio.orchestrator import Orchestrator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="symbio", description="SymBioCortex CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Trainiere den BioCortex")
    train.add_argument("--data", nargs="+", required=True, help="Textdateien zum Training")
    train.add_argument("--model-dir", default="runs/model", help="Zielverzeichnis")

    generate = sub.add_parser("generate", help="Generiere Text")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new", type=int, default=64)
    generate.add_argument("--model-dir", default="runs/model")
    generate.add_argument("--corpus", default="datasets/sample_corpus.txt")
    generate.add_argument("--neo-rate", type=float, default=None, help="Anteil neuer Wörter [0-1]")
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--n-candidates", type=int, default=8)
    generate.add_argument("--snap", action="store_true", help="Sanftes Snapping aktivieren")
    generate.add_argument("--debug", action="store_true", help="Zeige Ranking-Scores")
    generate.add_argument("--log", default="runs/generate_log.jsonl", help="Feedback-Logdatei")
    generate.add_argument("--w-fluency", type=float, default=0.40)
    generate.add_argument("--w-semantic", type=float, default=0.30)
    generate.add_argument("--w-form", type=float, default=0.20)
    generate.add_argument("--w-neology", type=float, default=0.10)
    generate.add_argument("--neo-low", type=float, default=0.10)
    generate.add_argument("--neo-high", type=float, default=0.35)

    run = sub.add_parser("run", help="Führe eine komplette Episode aus")
    run.add_argument("--prompt", required=True)
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--model-dir", default="runs/model")
    run.add_argument("--save-run", default="runs/last_run.json")

    auto = sub.add_parser("autopoiesis", help="Generiere Sätze aus Feldreaktionen")
    auto.add_argument("--data", nargs="+", required=True, help="Datensätze für das Denken")
    auto.add_argument("--steps", type=int, default=200)
    auto.add_argument("--threshold", type=float, default=0.6)
    auto.add_argument("--max-sentences", type=int, default=5)
    auto.add_argument("--model-dir", default="runs/model")
    auto.add_argument("--save", help="Optionaler Pfad für JSON-Ergebnis")

    tune = sub.add_parser("tune", help="Passe Rerank-Gewichte anhand des Feedback-Logs an")
    tune.add_argument("--log", default="runs/generate_log.jsonl")
    tune.add_argument("--step", type=float, default=0.05)

    return parser


def load_texts(paths: Sequence[str]) -> list[str]:
    texts = []
    for path in paths:
        texts.append(Path(path).read_text(encoding="utf-8"))
    return texts


def cmd_train(args: argparse.Namespace, config: SymbioConfig) -> None:
    configure_logging()
    texts = load_texts(args.data)
    cortex = BioCortex(config=config.bio)
    cortex.partial_fit(texts)
    cortex.save(args.model_dir)
    LOGGER.info("BioCortex gespeichert in %s", args.model_dir)


def cmd_generate(args: argparse.Namespace, config: SymbioConfig) -> None:
    configure_logging()
    model = KNTrigram.load(args.model_dir, config=config.bio)
    prompt_tokens = tokenize(args.prompt)
    candidates: list[str] = []
    for _ in range(args.n_candidates):
        generated = model.sample(
            prompt_tokens,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            neo_rate=args.neo_rate,
        )
        candidates.append(generated)

    weights = RankWeights(
        w_fluency=args.w_fluency,
        w_semantic=args.w_semantic,
        w_form=args.w_form,
        w_neology=args.w_neology,
        neo_target_low=args.neo_low,
        neo_target_high=args.neo_high,
    )

    ranked = rerank_candidates(
        prompt=args.prompt,
        candidates=candidates,
        kn_model=model.lm,
        tokenizer=model.tokenizer,
        corpus_path=args.corpus,
        weights=weights,
        snap=args.snap,
    )

    best = ranked[0]
    print(best.text)

    if args.debug:
        debug_payload = [
            {
                "total": r.total,
                **r.scores,
                "text": r.text,
            }
            for r in ranked[: min(5, len(ranked))]
        ]
        print("\n[DEBUG top-k scores]")
        print(json.dumps(debug_payload, ensure_ascii=False, indent=2))

    if args.log:
        from datetime import datetime

        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "prompt": args.prompt,
            "params": {
                "max_new": args.max_new,
                "neo_rate": args.neo_rate,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "weights": asdict(weights),
                "snap": args.snap,
            },
            "ranked": export_rank_results(ranked),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def cmd_run(args: argparse.Namespace, config: SymbioConfig) -> None:
    configure_logging()
    cortex = BioCortex.load(args.model_dir, config=config.bio)
    hpio = HPIO(field_config=config.field, swarm_config=config.swarm)
    orchestrator = Orchestrator(cortex, hpio)
    summary = orchestrator.run_episode(args.prompt, steps=args.steps)
    Path(args.save_run).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Episode beendet: %s", summary)


def cmd_autopoiesis(args: argparse.Namespace, config: SymbioConfig) -> None:
    configure_logging()
    texts = load_texts(args.data)
    model_dir = Path(args.model_dir)
    if model_dir.exists():
        cortex = BioCortex.load(model_dir, config=config.bio)
        cortex.partial_fit(texts)
    else:
        cortex = BioCortex(config=config.bio)
        cortex.partial_fit(texts)
    hpio = HPIO(field_config=config.field, swarm_config=config.swarm)
    orchestrator = Orchestrator(cortex, hpio)
    result = orchestrator.autopoietic_cycle(
        texts,
        steps=args.steps,
        threshold=args.threshold,
        max_sentences=args.max_sentences,
    )
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.save:
        Path(args.save).write_text(output, encoding="utf-8")
        LOGGER.info("Autopoiesis-Ergebnis gespeichert in %s", args.save)
    print(output)


def cmd_tune(args: argparse.Namespace, config: SymbioConfig) -> None:
    configure_logging()
    tuned = tune_rank_weights(args.log, step=args.step)
    print(json.dumps({"weights": asdict(tuned)}, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = DEFAULT_CONFIG
    match args.command:
        case "train":
            cmd_train(args, config)
        case "generate":
            cmd_generate(args, config)
        case "run":
            cmd_run(args, config)
        case "autopoiesis":
            cmd_autopoiesis(args, config)
        case "tune":
            cmd_tune(args, config)
        case _:
            parser.error(f"Unbekannter Befehl: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
