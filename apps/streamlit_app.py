"""Streamlit App für SymBioCortex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from symbio.biocortex import BioCortex
from symbio.coach.rerank import RankWeights, rerank_candidates
from symbio.config import DEFAULT_CONFIG
from symbio.feedback import detect_hotspots
from symbio.hpio import HPIO
from symbio.logging_setup import configure_logging
from symbio.orchestrator import Orchestrator

configure_logging()
st.set_page_config(page_title="SymBioCortex", layout="wide")


def _bootstrap_corpus() -> list[str]:
    """Lade ein kleines Initial-Korpus.

    Die Funktion sucht zunächst nach ``datasets/sample_corpus.txt`` im aktuellen
    Arbeitsverzeichnis und anschließend relativ zum Projektverzeichnis. Falls
    keine Datei gefunden wird, wird ein kurzer Fallback-Text genutzt, sodass der
    Tokenizer deterministisch initialisiert werden kann.
    """

    candidates = [
        Path("datasets") / "sample_corpus.txt",
        Path(__file__).resolve().parents[1] / "datasets" / "sample_corpus.txt",
        Path(__file__).resolve().parents[2] / "datasets" / "sample_corpus.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [candidate.read_text(encoding="utf-8")]
    fallback = (
        "SymBioCortex initialisiert den BioCortex mit einem kleinen Fallback-"
        "Text, um sofortige Interaktion zu ermöglichen."
    )
    return [fallback]


def _ensure_corpus_path() -> Path:
    """Bestimme den zu verwendenden Korpuspfad."""

    candidate = Path("datasets") / "sample_corpus.txt"
    if candidate.exists():
        return candidate
    fallback = Path("runs") / "streamlit_corpus.txt"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text("\n".join(_bootstrap_corpus()), encoding="utf-8")
    return fallback


st.title("SymBioCortex")
config = DEFAULT_CONFIG

if "cortex" not in st.session_state:
    cortex = BioCortex(config=config.bio)
    cortex.partial_fit(_bootstrap_corpus())
    st.session_state["cortex"] = cortex
if "hpio" not in st.session_state:
    st.session_state["hpio"] = HPIO(field_config=config.field, swarm_config=config.swarm)

cortex: BioCortex = st.session_state["cortex"]
hpio: HPIO = st.session_state["hpio"]

think_tab, act_tab, sym_tab = st.tabs(["Think", "Act", "Symbiosis"])

with think_tab:
    st.header("BioCortex")
    prompt = st.text_input("Prompt", "Symbiose des Denkens")
    neo_rate = st.slider("Neologismen-Rate", 0.0, 1.0, float(cortex.config.neo_rate), 0.05)
    neo_low, neo_high = st.slider("Neologismen-Zielband", 0.0, 1.0, (0.10, 0.35), 0.01)
    col_flu, col_sem, col_form, col_neo = st.columns(4)
    w_flu = col_flu.slider("Gewicht Fluency", 0.0, 1.0, 0.40, 0.05)
    w_sem = col_sem.slider("Gewicht Semantik", 0.0, 1.0, 0.30, 0.05)
    w_form = col_form.slider("Gewicht Form", 0.0, 1.0, 0.20, 0.05)
    w_neo = col_neo.slider("Gewicht Neologie", 0.0, 1.0, 0.10, 0.05)
    st.caption(f"Summe der Gewichte: {w_flu + w_sem + w_form + w_neo:.2f}")
    n_candidates = st.slider("Anzahl Kandidaten", 1, 16, 8)
    snap = st.checkbox("Sanftes Snapping", value=False)
    corpus_path = _ensure_corpus_path()
    ready = bool(cortex.tokenizer.vocab)
    if st.button("Generate", key="generate", disabled=not ready):
        candidates: list[str] = []
        for _ in range(n_candidates):
            candidates.append(cortex.generate(prompt, max_new_tokens=48, neo_rate=neo_rate))
        weights = RankWeights(
            w_fluency=w_flu,
            w_semantic=w_sem,
            w_form=w_form,
            w_neology=w_neo,
            neo_target_low=neo_low,
            neo_target_high=neo_high,
        )
        ranked = rerank_candidates(
            prompt=prompt,
            candidates=candidates,
            kn_model=cortex.lm,
            tokenizer=cortex.tokenizer,
            corpus_path=str(corpus_path),
            weights=weights,
            snap=snap,
        )
        best = ranked[0]
        st.subheader("Top-Ausgabe")
        st.write(best.text)
        st.write(
            f"Scores – Fluency: {best.scores['fluency']:.3f}, "
            f"Semantik: {best.scores['semantic']:.3f}, "
            f"Form: {best.scores['form']:.3f}, "
            f"Neologie: {best.scores['neo_ratio']:.3f} (adjusted {best.scores['neo_adj']:.3f})"
        )
        df = pd.DataFrame(
            [
                {
                    "total": r.total,
                    **r.scores,
                    "text": r.text,
                }
                for r in ranked
            ]
        )
        st.dataframe(df)
    if not ready:
        st.info("Bitte zunächst ein Trainingskorpus laden, um Texte zu generieren.")
    graph_edges = list(cortex.graph.weights.items())
    if graph_edges:
        g = nx.DiGraph()
        for (a, b), weight in graph_edges:
            g.add_edge(a, b, weight=weight)
        pos = nx.spring_layout(g, seed=42)
        fig, ax = plt.subplots(figsize=(5, 4))
        nx.draw(g, pos, ax=ax, with_labels=True, node_size=300, arrows=True)
        st.pyplot(fig)

with act_tab:
    st.header("HPIO Feld")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(hpio.field.phi, cmap="magma")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
    if st.button("Swarm Step", key="swarm_step"):
        metrics = hpio.step()
        hpio.relax_and_evaporate()
        st.write(metrics)

with sym_tab:
    st.header("Symbiotische Episode")
    prompt = st.text_input("Symbio Prompt", "Die Architektur des Denkens")
    steps = st.slider("Steps", min_value=10, max_value=200, value=50, step=10)
    if st.button("Run Episode", key="episode", disabled=not ready):
        orchestrator = Orchestrator(cortex, hpio)
        summary = orchestrator.run_episode(prompt, steps=steps)
        st.write(summary)
        hotspots = detect_hotspots(hpio.field)
        if hotspots:
            df = pd.DataFrame(hotspots, columns=["y", "x", "value"])
            st.dataframe(df)
        Path("runs/streamlit_last.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not ready:
        st.warning("Die Symbiose benötigt ein trainiertes Modell. Bitte zuerst trainieren.")
