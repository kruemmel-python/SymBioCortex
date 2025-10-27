"""Fassade für den BioCortex."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import BioConfig
from .lm_kn import KneserNeyLM
from .mycelium import MyceliumGraph
from .neuromod import NeuromodulatorState
from .replay import ReplayBuffer
from .tokenization import BioBPETokenizer
from .types import Concept, Edge, Pulse
from .utils import ensure_dir, softmax

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BioCortex:
    """Kapselt Tokenizer, Sprachmodell, Myzel und Replay."""

    config: BioConfig = field(default_factory=BioConfig)
    tokenizer: BioBPETokenizer = field(default_factory=BioBPETokenizer)
    lm: KneserNeyLM = field(init=False)
    graph: MyceliumGraph = field(default_factory=MyceliumGraph)
    neuromod: NeuromodulatorState = field(default_factory=NeuromodulatorState)
    replay: ReplayBuffer = field(init=False)
    _corpus: list[list[int]] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(1234))

    def __post_init__(self) -> None:
        self.lm = KneserNeyLM(order=self.config.ngram_order, discount=self.config.discount)
        self.replay = ReplayBuffer(capacity=self.config.replay_capacity)

    def partial_fit(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        if not self.tokenizer.vocab:
            self.tokenizer.fit(texts)
        sequences = [self.tokenizer.encode(text) for text in texts]
        self._corpus.extend(sequences)
        self.lm.train_sequences(self._corpus)
        for seq in sequences:
            self.replay.add(seq)
            for a, b in zip(seq, seq[1:]):
                self.graph.update_edge(edge=Edge(a, b), pre=1.0, post=1.0)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        nucleus_p: float = 0.9,
        temperature: float = 1.0,
    ) -> str:
        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        for _ in range(max_new_tokens):
            context = generated[-(self.lm.order - 1) :]
            probs = self.lm.prob_next(context)
            tokens_sorted = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            cumulative = 0.0
            nucleus: list[tuple[int, float]] = []
            for token, prob in tokens_sorted:
                nucleus.append((token, prob))
                cumulative += prob
                if cumulative >= nucleus_p:
                    break
            ids, weights = zip(*nucleus)
            scaled = [prob * self.neuromod.dopamine for prob in weights]
            distribution = softmax(scaled, temperature=temperature)
            choice = self._sample(ids, distribution)
            generated.append(choice)
            if choice == 1:
                break
        return self.tokenizer.decode(generated)

    def _sample(self, options: Sequence[int], probs: Sequence[float]) -> int:
        threshold = self.rng.random()
        cumulative = 0.0
        for option, prob in zip(options, probs):
            cumulative += prob
            if threshold <= cumulative:
                return option
        return options[-1]

    def extract_concepts(self, prompt: str) -> list[Concept]:
        tokens = self.tokenizer.encode(prompt)
        unique = sorted(set(tokens))
        concepts: list[Concept] = []
        total = len(tokens) or 1
        for token in unique:
            strength = tokens.count(token) / total
            successors = self.graph.top_k_successors(token, k=1)
            pher = max(
                [self.graph.pheromones.get((token, succ), 0.0) for succ in successors]
                or [self.graph.pheromones.get((token, token), 0.0)]
            )
            name = self.tokenizer.decode([token])
            concepts.append(Concept(name=name or f"tok{token}", strength=strength, pheromone=pher))
        concepts.sort(key=lambda c: c.strength, reverse=True)
        return concepts[: self.config.concept_top_k]

    def concepts_to_pulses(self, concepts: Sequence[Concept], field_shape: tuple[int, int]) -> list[Pulse]:
        h, w = field_shape
        pulses: list[Pulse] = []
        for concept in concepts:
            seed = abs(hash(concept.name))
            y = seed % h
            x = (seed // h) % w
            amplitude = 0.5 + concept.strength * (1.0 + concept.pheromone)
            spread = 1.5 + concept.pheromone
            pulses.append(Pulse(position=(y, x), amplitude=amplitude, spread=spread, tag=concept.name))
        return pulses

    def save(self, directory: str | Path) -> None:
        directory = ensure_dir(directory)
        self.tokenizer.save(str(Path(directory) / "tokenizer.json"))
        self.lm.save(str(Path(directory) / "language_model.kn.json"))
        graph_path = Path(directory) / "graph.json"
        data = {
            "weights": {f"{a},{b}": w for (a, b), w in self.graph.weights.items()},
            "pheromones": {f"{a},{b}": p for (a, b), p in self.graph.pheromones.items()},
        }
        graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, config: BioConfig | None = None) -> "BioCortex":
        config = config or BioConfig()
        instance = cls(config=config)
        directory = Path(directory)
        instance.tokenizer = BioBPETokenizer.load(str(directory / "tokenizer.json"))
        instance.lm = KneserNeyLM.load(str(directory / "language_model.kn.json"))
        data = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        instance.graph.weights = {tuple(map(int, key.split(","))): float(value) for key, value in data["weights"].items()}
        instance.graph.pheromones = {
            tuple(map(int, key.split(","))): float(value) for key, value in data["pheromones"].items()
        }
        return instance


__all__ = ["BioCortex"]
