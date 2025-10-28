"""Streamlit App für SymBioCortex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from symbio.biocortex import BioCortex
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
    ready = bool(cortex.tokenizer.vocab)
    if st.button("Generate", key="generate", disabled=not ready):
        output = cortex.generate(prompt, max_new_tokens=48, neo_rate=neo_rate)
        stats = cortex.last_neology
        if stats:
            st.write(f"Neologismen: {stats.novel}/{stats.total} ({stats.ratio:.1%})")
        st.write("Output:", output)
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
