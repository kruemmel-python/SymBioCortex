"""Kommandozeileninterface für SymBioCortex."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

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
    generate.add_argument("--neo-rate", type=float, default=None, help="Anteil neuer Wörter [0-1]")

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
    cortex = BioCortex.load(args.model_dir, config=config.bio)
    text = cortex.generate(args.prompt, max_new_tokens=args.max_new, neo_rate=args.neo_rate)
    print(text)


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
        case _:
            parser.error(f"Unbekannter Befehl: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
