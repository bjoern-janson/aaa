import inspect

from aaa_v0.agents import FixedMetaAgent, LocalAgent, RevisingMetaAgent
from aaa_v0.contracts import ProbeObservation, SolvedHistoryView, SurfaceView
from aaa_v0.serialization import canonical_json_bytes, sha256_json
from aaa_v0.tasks import context_signature, probe_signature


def _history_record(ctx_idx=1, q_idx=2, theta=3):
    width = 4
    surface = SurfaceView(
        task_id="h",
        family="D1",
        context_signature=context_signature(ctx_idx, width),
        context_token=f"context-{ctx_idx}",
        probe_signatures=tuple(probe_signature(i, width) for i in range(width)),
        probe_tokens=tuple(f"probe-{i}" for i in range(width)),
    )
    return SolvedHistoryView(surface, probe_signature(q_idx, width), theta)


def _future_view(ctx_idx=1):
    width = 4
    return SurfaceView(
        task_id="f",
        family="D3",
        context_signature=context_signature(ctx_idx, width),
        context_token=f"context-{ctx_idx}",
        probe_signatures=tuple(probe_signature(i, width) for i in range(width)),
        probe_tokens=tuple(f"probe-{i}" for i in range(width)),
    )


def test_public_agent_interface_has_no_latent_parameter_names_and_state_is_deterministic():
    forbidden = {"z", "q_star", "theta", "phi"}
    methods = ["observe_history", "start_task", "choose_probe", "observe_probe", "guess_theta", "state_dict", "preferred_probe"]
    for cls in (LocalAgent, FixedMetaAgent, RevisingMetaAgent):
        for name in methods:
            params = set(inspect.signature(getattr(cls, name)).parameters)
            assert not forbidden & params
        agent = cls(theta_count=4)
        assert canonical_json_bytes(agent.state_dict()) == canonical_json_bytes(agent.state_dict())


def test_fixed_stays_brittle_while_revising_updates_mapping_after_mismatch():
    record = _history_record()
    ctx = record.surface.context_signature
    old_probe = record.informative_probe_signature
    new_probe = probe_signature(3, 4)
    view = _future_view()

    fixed = FixedMetaAgent(theta_count=4)
    revising = RevisingMetaAgent(theta_count=4)
    fixed.observe_history((record,))
    revising.observe_history((record,))

    for agent in (fixed, revising):
        agent.start_task(view)
        assert agent.choose_probe() == old_probe
        agent.observe_probe(ProbeObservation(old_probe, False, None))
        # Simulate local search discovering the new informative role.
        while True:
            probe = agent.choose_probe()
            if probe == new_probe:
                agent.observe_probe(ProbeObservation(probe, True, 1))
                break
            agent.observe_probe(ProbeObservation(probe, False, None))

    assert fixed.preferred_probe(ctx) == old_probe
    assert revising.preferred_probe(ctx) == new_probe
    assert sha256_json(fixed.state_dict()) != sha256_json(revising.state_dict())


def test_local_ignores_passive_history_and_keeps_no_cross_task_mapping():
    local_a = LocalAgent(theta_count=4)
    local_b = LocalAgent(theta_count=4)
    local_a.observe_history((_history_record(0, 0),))
    local_b.observe_history((_history_record(0, 3), _history_record(1, 2)))
    assert local_a.preferred_probe(context_signature(0, 4)) is None
    assert local_b.preferred_probe(context_signature(0, 4)) is None
    assert local_a.state_dict() == local_b.state_dict()
