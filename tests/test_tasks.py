from dataclasses import asdict

from aaa_v0.contracts import AssayConfig, HistoryEpisode
from aaa_v0.provenance import FutureSeedReveal, seed_commitment
from aaa_v0.tasks import (
    context_signature,
    make_future_tasks,
    probe_signature,
    render_history_episode,
    render_task,
)


def _reveal(seed=101):
    return FutureSeedReveal(seed=seed, commitment=seed_commitment(seed))


def test_future_families_pair_d1_d2_and_invalidate_d3():
    cfg = AssayConfig.exploratory_default()
    phi = (2, 0, 3, 1)
    families = make_future_tasks(cfg, _reveal(), phi=phi)
    assert set(families) == {"D1", "D2", "D3"}
    assert len(families["D1"]) == cfg.future_tasks_per_family
    assert all(t.family == "D1" for t in families["D1"])
    for d1, d2 in zip(families["D1"], families["D2"], strict=True):
        assert (d1.z, d1.theta, d1.q_star) == (d2.z, d2.theta, d2.q_star)
        assert d1.surface_seed != d2.surface_seed
    for i, task in enumerate(families["D3"]):
        if i < cfg.d3_change_after:
            assert task.q_star == phi[task.z]
        else:
            assert task.q_star != phi[task.z]


def test_d2_changes_surface_but_preserves_structural_signatures_without_latent_fields():
    cfg = AssayConfig.exploratory_default()
    d1, d2 = [
        fam[0] for fam in (
            make_future_tasks(cfg, _reveal(102), (2, 0, 3, 1))["D1"],
            make_future_tasks(cfg, _reveal(102), (2, 0, 3, 1))["D2"],
        )
    ]
    v1 = render_task(d1, cfg)
    v2 = render_task(d2, cfg)
    assert v1.context_signature == v2.context_signature
    assert set(v1.probe_signatures) == set(v2.probe_signatures)
    assert v1.probe_tokens != v2.probe_tokens
    assert v1.probe_signatures != v2.probe_signatures
    keys = set(asdict(v2))
    assert not {"theta", "q_star", "phi"} & keys


def test_passive_history_exposes_observable_informative_role_and_resolved_target():
    cfg = AssayConfig.exploratory_default()
    ep = HistoryEpisode("h-0000", z=1, theta=3, q_star=2, surface_seed=7, resource_units=1)
    view = render_history_episode(ep, cfg)
    assert view.surface.context_signature == context_signature(1, 4)
    assert view.informative_probe_signature == probe_signature(2, 4)
    assert view.resolved_theta == 3
    assert "q_star" not in asdict(view.surface)
