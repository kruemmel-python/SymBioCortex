import random

import numpy as np

from symbio.biocortex import BioCortex
from symbio.metrics.neology import build_corpus_lexicon, neology_ratio
from symbio.generate.mix_sampler import mix_probs, sample_mixed
from symbio.morph.guardrails import affix_boost, good_shape, morph_wrapper


def test_build_corpus_lexicon_lowercases_and_deduplicates():
    lexicon = build_corpus_lexicon(["Hallo Welt", "Neue Welten", ""])
    assert lexicon == {"hallo", "welt", "neue", "welten"}


def test_neology_ratio_counts_novel():
    lexicon = {"test"}
    stats = neology_ratio(["Test", "Neu"], lexicon)
    assert stats.total == 2
    assert stats.novel == 1
    assert stats.ratio == 0.5


def test_mix_sampler_extremes():
    probs = np.array([0.7, 0.3])
    mixed, gate = mix_probs(probs, p_neologism=1.0, neo_rate=0.5)
    assert np.isclose(mixed.sum(), 1.0)
    assert 0.0 < gate <= 1.0

    rng = np.random.default_rng(123)
    idx, is_neo = sample_mixed(np.array([0.5, 0.5]), neo_rate=0.0, rng=rng)
    assert is_neo is False
    assert idx in {0, 1}

    rng = np.random.default_rng(321)
    idx, is_neo = sample_mixed(np.array([0.5, 0.5]), neo_rate=1.0, rng=rng)
    assert is_neo is True
    assert idx is None


def test_guardrails_maintain_good_shape():
    rnd = random.Random(42)

    def base_generator(r: random.Random) -> str:
        return "ngwort" if r.random() < 0.5 else "klar"

    wrapped = morph_wrapper(base_generator)
    word = wrapped(rnd)
    assert good_shape(word)
    boosted = affix_boost("kern", random.Random(1))
    prefixes = ("ver", "be", "ent", "zer", "ur", "um", "miss")
    suffixes = ("ung", "heit", "keit", "isch", "ieren", "bar", "haft", "sam", "los", "frei")
    assert boosted != "kern"
    assert boosted.startswith(prefixes) or boosted.endswith(suffixes)


def test_biocortex_records_neology_stats():
    cortex = BioCortex()
    cortex.partial_fit(["Das System lernt neue Wörter"])
    cortex.rng.seed(123)
    text = cortex.generate("das", max_new_tokens=4, neo_rate=1.0)
    assert text.startswith("das")
    assert cortex.last_neology is not None
    assert cortex.last_neology.total >= 1
    assert cortex.last_neology.novel == cortex.last_neology.total
