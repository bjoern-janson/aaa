import pytest

from aaa_v0.contracts import LatentTask, ProbeObservation
from aaa_v0.environment import BudgetExceeded, MetaProbeEnvironment
from aaa_v0.tasks import probe_signature


def test_only_informative_probe_reveals_theta_and_budget_is_enforced():
    task = LatentTask("x", "D1", z=0, theta=3, q_star=2, surface_seed=1, phase_index=0)
    env = MetaProbeEnvironment(task, q_count=4, max_budget=4)
    q2 = probe_signature(2, 4)
    q0 = probe_signature(0, 4)
    assert env.probe(q2) == ProbeObservation(q2, True, 3)
    assert env.probe(q0) == ProbeObservation(q0, False, None)
    env.probe(q0)
    env.probe(q0)
    with pytest.raises(BudgetExceeded):
        env.probe(q0)
