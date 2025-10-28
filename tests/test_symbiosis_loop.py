from symbio.biocortex import BioCortex
from symbio.config import SymbioConfig, BioConfig, FieldConfig, SwarmConfig
from symbio.hpio import HPIO
from symbio.orchestrator import Orchestrator


def test_symbiosis_generates_feedback(tmp_path):
    config = SymbioConfig(
        bio=BioConfig(ngram_order=2, replay_capacity=8, concept_top_k=4),
        field=FieldConfig(shape=(12, 12), relax_alpha=0.1, evaporate_rate=0.05),
        swarm=SwarmConfig(n_agents=3, boundary="periodic", seed=2),
    )
    cortex = BioCortex(config=config.bio)
    cortex.partial_fit(["Die Architektur des Denkens verbindet Pulse"])
    hpio = HPIO(field_config=config.field, swarm_config=config.swarm)
    orchestrator = Orchestrator(cortex, hpio)
    summary = orchestrator.run_episode("Die Architektur des Denkens", steps=15)
    assert summary["events"] > 0
    assert hpio.best_val > float("-inf")
    assert cortex.graph.weights


def test_autopoietic_cycle_creates_sentences():
    config = SymbioConfig(
        bio=BioConfig(ngram_order=2, replay_capacity=16, concept_top_k=4),
        field=FieldConfig(shape=(10, 10), relax_alpha=0.1, evaporate_rate=0.05),
        swarm=SwarmConfig(n_agents=3, boundary="periodic", seed=3),
    )
    corpus = [
        "Bioinspirierte Architektur",
        "Feldreaktionen erzeugen neue Gedanken",
    ]
    cortex = BioCortex(config=config.bio)
    cortex.partial_fit(corpus)
    hpio = HPIO(field_config=config.field, swarm_config=config.swarm)
    orchestrator = Orchestrator(cortex, hpio)
    result = orchestrator.autopoietic_cycle(corpus, steps=20, threshold=0.1, max_sentences=2)
    assert result["sentences"]
    assert all(isinstance(sentence, str) and sentence for sentence in result["sentences"])
