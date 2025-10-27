from symbio.mycelium import MyceliumGraph
from symbio.types import Edge


def test_reinforce_increases_weight():
    graph = MyceliumGraph()
    edge = Edge(1, 2)
    graph.update_edge(edge, pre=1.0, post=1.0)
    before = graph.weights[(1, 2)]
    graph.reinforce([1, 2, 3], amount=0.5)
    after = graph.weights[(1, 2)]
    assert after > before
    graph.evaporate(0.1)
    assert (1, 2) in graph.weights
