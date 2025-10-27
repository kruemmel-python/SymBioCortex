"""Hilfsfunktionen für Event-Erzeugung und -Validierung."""

from __future__ import annotations

from typing import Any, Iterable

from .types import Event


def make_event(kind: Event.kind, payload: Any) -> Event:
    """Erzeuge ein Event und validiere den Inhalt.

    Args:
        kind: Typ des Events.
        payload: Beliebige Nutzlast.

    Returns:
        Ein validiertes :class:`Event`.

    Raises:
        ValueError: Falls die Nutzlast offensichtlich ungültig ist.
    """

    if kind == "pulse" and payload is None:
        raise ValueError("pulse events require a payload")
    if kind == "tick" and not isinstance(payload, dict | None.__class__):
        raise ValueError("tick payload must be a dict or None")
    return Event(kind=kind, payload=payload)


def iter_by_kind(events: Iterable[Event], kind: Event.kind) -> Iterable[Event]:
    """Filtere Events nach ihrem Typ."""

    for event in events:
        if event.kind == kind:
            yield event


def debug_match(event: Event) -> str:
    """Zeigt ein Pattern-Matching-Beispiel."""

    match event:
        case Event(kind="pulse", payload=payload):
            return f"pulse:{type(payload).__name__}"
        case Event(kind="feedback", payload=payload):
            return f"feedback:{payload!r}"
        case Event(kind="tick", payload=payload):
            return f"tick:{payload!r}"
        case Event(kind="decay", payload=payload):
            return f"decay:{payload!r}"
        case _:
            return "unknown"


__all__ = ["make_event", "iter_by_kind", "debug_match"]
