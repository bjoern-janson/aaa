from dataclasses import FrozenInstanceError

import pytest

from aaa_v0.contracts import AssayConfig, HistoryEpisode
from aaa_v0.serialization import canonical_json_bytes, sha256_json


def test_assay_config_is_frozen_and_canonical():
    cfg = AssayConfig.exploratory_default()
    first = canonical_json_bytes(cfg.to_dict())
    second = canonical_json_bytes(dict(reversed(list(cfg.to_dict().items()))))
    assert first == second
    assert sha256_json(cfg.to_dict()) == sha256_json(cfg.to_dict())
    with pytest.raises(FrozenInstanceError):
        cfg.z_count = 99


def test_assay_config_rejects_degenerate_or_unbalanceable_v0_shapes():
    with pytest.raises(ValueError, match="two context/probe"):
        AssayConfig(1, 1, 4, 8, 32, 4, 8)
    with pytest.raises(ValueError, match="two target"):
        AssayConfig(4, 4, 1, 8, 32, 4, 8)
    with pytest.raises(ValueError, match="history_repeats"):
        AssayConfig(4, 4, 3, 8, 32, 4, 8)
    with pytest.raises(ValueError, match="post-change"):
        AssayConfig(4, 4, 4, 8, 8, 4, 8)


def test_history_episode_serializes_declared_fields():
    ep = HistoryEpisode(
        episode_id="h-000",
        z=0,
        theta=2,
        q_star=1,
        surface_seed=7,
        resource_units=1,
    )
    assert ep.to_dict() == {
        "episode_id": "h-000",
        "z": 0,
        "theta": 2,
        "q_star": 1,
        "surface_seed": 7,
        "resource_units": 1,
    }


def test_sequence_balance_rule_validates_tolerances_and_limits():
    from aaa_v0.contracts import SequenceBalanceRule

    rule = SequenceBalanceRule(0.5, 0.5, 0.1, 3, 100)
    assert rule.lag_window == 3
    with pytest.raises(ValueError):
        SequenceBalanceRule(-0.1, 0.5, 0.1, 3, 100)
    with pytest.raises(ValueError):
        SequenceBalanceRule(0.5, 0.5, 0.1, 0, 100)
