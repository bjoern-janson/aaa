import pytest

from aaa_v0.agents import FixedMetaAgent
from aaa_v0.contracts import AssayConfig, SequenceBalanceRule
from aaa_v0.runner import ProtocolStop, assert_match_gate, run_reference_pair
from aaa_v0.serialization import canonical_json_bytes


def test_match_gate_is_per_family_and_fails_closed():
    with pytest.raises(ProtocolStop) as exc:
        assert_match_gate({"D1": 0.25, "D2": 0.0, "D3": 0.0}, tolerance=0.10)
    assert exc.value.code == "MATCH_FAILURE"
    report = assert_match_gate({"D1": 0.0, "D2": 0.0, "D3": 0.0}, tolerance=0.0)
    assert report.ok


def test_reference_run_is_deterministic_and_future_seed_only_changes_future_outputs():
    cfg = AssayConfig.exploratory_default()
    rule = SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)
    a = run_reference_pair(cfg, rule, history_seed=17, future_seed=101, agent_cls=FixedMetaAgent, match_tolerance=0.0)
    b = run_reference_pair(cfg, rule, history_seed=17, future_seed=101, agent_cls=FixedMetaAgent, match_tolerance=0.0)
    c = run_reference_pair(cfg, rule, history_seed=17, future_seed=102, agent_cls=FixedMetaAgent, match_tolerance=0.0)
    assert canonical_json_bytes(a.to_dict()) == canonical_json_bytes(b.to_dict())
    assert a.config_hash == c.config_hash
    assert a.history_pair_hash == c.history_pair_hash
    assert a.raw_trial_records_hash != c.raw_trial_records_hash
    assert a.future_seed_commitment != c.future_seed_commitment


def test_balance_construction_failure_is_protocol_stop():
    cfg = AssayConfig.exploratory_default()
    impossible = SequenceBalanceRule(0.0, 0.0, 0.0, 3, 1)
    with pytest.raises(ProtocolStop) as exc:
        run_reference_pair(cfg, impossible, 17, 101, FixedMetaAgent, 0.0)
    assert exc.value.code == "BALANCE_FAILURE"


def test_run_bundle_does_not_expose_inflated_score_fields():
    cfg = AssayConfig.exploratory_default()
    rule = SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)
    bundle = run_reference_pair(cfg, rule, 17, 101, FixedMetaAgent, 0.0)
    keys = set(bundle.to_dict())
    assert "aaa_score" not in keys
    assert "intelligence" not in keys
    assert "self_improvement" not in keys
