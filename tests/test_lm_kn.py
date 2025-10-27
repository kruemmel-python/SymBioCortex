from symbio.lm_kn import KneserNeyLM


def test_probabilities_normalize():
    lm = KneserNeyLM(order=3, discount=0.5)
    sequences = [[1, 2, 3], [1, 2, 4], [2, 3, 4]]
    lm.train_sequences(sequences)
    probs = lm.prob_next([1, 2])
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(prob >= 0 for prob in probs.values())
