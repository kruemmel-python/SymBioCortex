"""Orchestrator mit Event-Loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

from .biocortex import BioCortex
from .bridge import text_to_pulses
from .events import make_event
from .feedback import apply_feedback, detect_hotspots
from .hpio import HPIO
from .autopoiesis import synthesize_thoughts
from .types import Event

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Orchestrator:
    """Verknüpft BioCortex und HPIO über Events."""

    biocortex: BioCortex
    hpio: HPIO
    event_log: list[Event] = field(default_factory=list)

    def dispatch(self, event: Event) -> list[Event]:
        """Bearbeitet ein Event via Pattern Matching."""

        logger.debug("Dispatching event: %s", event)
        self.event_log.append(event)
        match event:
            case Event(kind="pulse", payload=pulses):
                self.hpio.inject_pulses(pulses)
                return []
            case Event(kind="tick", payload=payload):
                metrics = self.hpio.step()
                self.hpio.relax_and_evaporate()
                hotspots = detect_hotspots(self.hpio.field)
                if hotspots:
                    return [make_event("feedback", {"hotspots": hotspots, "metrics": metrics})]
                return []
            case Event(kind="feedback", payload=data):
                result = apply_feedback(self.biocortex, data.get("hotspots", []))
                logger.debug("Feedback applied: %s", result)
                return []
            case Event(kind="decay", payload=rate):
                self.biocortex.graph.evaporate(rate)
                self.biocortex.neuromod.decay()
                return []
            case _:
                logger.warning("Unknown event kind: %s", event.kind)
                return []

    def run_episode(self, prompt: str, steps: int = 50) -> dict:
        """Führt eine komplette Episode aus."""

        pulses = text_to_pulses(self.biocortex, prompt, self.hpio.field.shape)
        queue: list[Event] = [make_event("pulse", pulses)]
        for step in range(steps):
            queue.append(make_event("tick", {"step": step}))
            if step % 10 == 0:
                queue.append(make_event("decay", 0.05))
            while queue:
                event = queue.pop(0)
                new_events = self.dispatch(event)
                queue.extend(new_events)
        return {
            "events": len(self.event_log),
            "best_pos": self.hpio.best_pos,
            "best_val": self.hpio.best_val,
        }

    def autopoietic_cycle(
        self,
        texts: Sequence[str],
        *,
        steps: int = 100,
        threshold: float = 0.6,
        max_sentences: int = 5,
    ) -> dict:
        """Überführt Texte in Feldreaktionen und erzeugt neue Sätze daraus."""

        queue: list[Event] = []
        for text in texts:
            if not text.strip():
                continue
            pulses = text_to_pulses(self.biocortex, text, self.hpio.field.shape)
            queue.append(make_event("pulse", pulses))
        for step in range(steps):
            queue.append(make_event("tick", {"step": step}))
            if step % 10 == 0:
                queue.append(make_event("decay", 0.05))
            while queue:
                event = queue.pop(0)
                new_events = self.dispatch(event)
                queue.extend(new_events)
        hotspots = detect_hotspots(self.hpio.field, threshold)
        sentences = synthesize_thoughts(
            self.biocortex,
            hotspots,
            max_sentences=max_sentences,
        )
        return {
            "sentences": sentences,
            "hotspots": [hotspot.to_dict(top_k=5) for hotspot in hotspots],
        }


__all__ = ["Orchestrator"]
