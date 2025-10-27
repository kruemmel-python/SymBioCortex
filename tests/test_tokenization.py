import pytest

from symbio.tokenization import BioBPETokenizer


def test_tokenizer_roundtrip():
    tokenizer = BioBPETokenizer()
    corpus = ["BioCortex ist adaptiv", "HPIO reagiert"]
    tokenizer.fit(corpus, vocab_size=64)
    text = "BioCortex reagiert adaptiv"
    ids = tokenizer.encode(text)
    reconstructed = tokenizer.decode(ids)
    assert isinstance(ids, list)
    assert reconstructed
    assert "biocortex" in reconstructed
