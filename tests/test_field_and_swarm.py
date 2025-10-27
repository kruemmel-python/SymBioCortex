from symbio.field import Field
from symbio.swarm import Swarm
from symbio.types import Pulse


def total_energy(field: Field) -> float:
    return sum(sum(row) for row in field.phi)


def test_field_injection_and_relax():
    field = Field((16, 16))
    before = total_energy(field)
    pulse = Pulse(position=(8, 8), amplitude=1.0, spread=2.0, tag="test")
    field.inject_gaussian(pulse)
    assert total_energy(field) > before
    field.relax(alpha=0.2)
    field.evaporate(0.1)
    assert max(max(row) for row in field.phi) < 1.5


def test_swarm_step_metrics():
    field = Field((16, 16))
    swarm = Swarm(field=field, n_agents=4, boundary="periodic", seed=1)
    metrics = swarm.step()
    assert "trails" in metrics and metrics["trails"]
    assert 0.0 <= metrics["mean_battery"] <= 1.0
