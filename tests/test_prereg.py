import json
from dataclasses import replace

import pytest

from aaa_v0.agents import FixedMetaAgent, LocalAgent, RevisingMetaAgent
from aaa_v0.contracts import AssayConfig, SequenceBalanceRule
from aaa_v0.prereg import (
    SPEC_COMMIT,
    Preregistration,
    RecoveryRule,
    TransferRule,
    freeze_preregistration,
    seed_identity,
    validate_confirmatory_controls,
)
from aaa_v0.runner import run_reference_pair
from aaa_v0.serialization import canonical_json_bytes, sha256_json


def _config_and_rule():
    return AssayConfig.exploratory_default(), SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)


def _rules():
    # Software-fixture thresholds only; not scientific preregistration values.
    return (
        TransferRule(min_positive_budget_points=1, min_mean_delta=0.01, max_zero_budget_abs_delta=0.0),
        TransferRule(min_positive_budget_points=1, min_mean_delta=0.01, max_zero_budget_abs_delta=0.0),
        RecoveryRule(max_post_change_old_rule_adherence=0.50, min_post_change_competence=0.99, evaluation_start_offset=4),
    )


def _bundles(cfg, seq):
    return {
        "LOCAL": run_reference_pair(cfg, seq, 17, 101, LocalAgent, 0.0),
        "FIXED_META": run_reference_pair(cfg, seq, 17, 101, FixedMetaAgent, 0.0),
        "REVISING_META": run_reference_pair(cfg, seq, 17, 101, RevisingMetaAgent, 0.0),
    }


def test_preregistration_roundtrip_and_identity_are_stable():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    prereg = freeze_preregistration(
        config=cfg,
        sequence_balance_rule=seq,
        d1_rule=d1,
        d2_rule=d2,
        d3_rule=d3,
        exploratory_seed_hashes=(seed_identity("history", 1), seed_identity("future", 2)),
        confirmatory_history_seeds=(101,),
        confirmatory_future_seeds=(202,),
        spec_commit=SPEC_COMMIT,
    )
    encoded = canonical_json_bytes(prereg.to_dict())
    decoded = Preregistration.from_dict(json.loads(encoded))
    assert canonical_json_bytes(decoded.to_dict()) == encoded
    assert sha256_json(decoded.to_dict()) == sha256_json(prereg.to_dict())


def test_freeze_rejects_seed_overlap_and_wrong_spec_commit():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    with pytest.raises(ValueError, match="overlap"):
        freeze_preregistration(
            cfg, seq, d1, d2, d3,
            (seed_identity("history", 101),),
            (101,), (202,), SPEC_COMMIT,
        )
    with pytest.raises(ValueError, match="spec_commit"):
        freeze_preregistration(cfg, seq, d1, d2, d3, (), (101,), (202,), "bad")


def test_confirmatory_reference_signature_matches_required_matrix():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    prereg = freeze_preregistration(cfg, seq, d1, d2, d3, (), (17,), (101,), SPEC_COMMIT)
    report = validate_confirmatory_controls(
        prereg,
        _bundles(cfg, seq),
        sequence_balance_rule=seq,
        history_seeds=(17,),
        future_seeds=(101,),
    )
    assert report.status == "CONTROL_VALIDATED"
    assert report.next_phase_gate_passed is True
    assert report.aaa_evidence == "NONE"
    assert report.signature == (
        ("LOCAL", False, False, False),
        ("FIXED_META", True, True, False),
        ("REVISING_META", True, True, True),
    )


def test_confirmatory_validation_rejects_wrong_sequence_rule_or_schedule():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    prereg = freeze_preregistration(cfg, seq, d1, d2, d3, (), (17,), (101,), SPEC_COMMIT)
    bundles = _bundles(cfg, seq)
    different = SequenceBalanceRule(1.0, 2.0, 2.0, 3, 1000)
    with pytest.raises(ValueError, match="sequence"):
        validate_confirmatory_controls(prereg, bundles, sequence_balance_rule=different, history_seeds=(17,), future_seeds=(101,))
    with pytest.raises(ValueError, match="schedule"):
        validate_confirmatory_controls(prereg, bundles, sequence_balance_rule=seq, history_seeds=(18,), future_seeds=(101,))


def test_confirmatory_validation_binds_bundles_to_agent_seed_and_rule_provenance():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    prereg = freeze_preregistration(cfg, seq, d1, d2, d3, (), (17,), (101,), SPEC_COMMIT)
    bundles = _bundles(cfg, seq)

    bad_seed = dict(bundles)
    bad_seed["LOCAL"] = replace(
        bad_seed["LOCAL"], history_seed_identity=seed_identity("history", 999)
    )
    with pytest.raises(ValueError, match="history seed"):
        validate_confirmatory_controls(prereg, bad_seed, sequence_balance_rule=seq, history_seeds=(17,), future_seeds=(101,))

    bad_agent = dict(bundles)
    bad_agent["LOCAL"] = replace(bad_agent["LOCAL"], agent_name="FIXED_META")
    with pytest.raises(ValueError, match="agent identity"):
        validate_confirmatory_controls(prereg, bad_agent, sequence_balance_rule=seq, history_seeds=(17,), future_seeds=(101,))


def test_confirmatory_validation_rejects_unfrozen_multi_seed_aggregation():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    prereg = freeze_preregistration(cfg, seq, d1, d2, d3, (), (17, 18), (101,), SPEC_COMMIT)
    with pytest.raises(ValueError, match="multi-seed"):
        validate_confirmatory_controls(
            prereg,
            _bundles(cfg, seq),
            sequence_balance_rule=seq,
            history_seeds=(17, 18),
            future_seeds=(101,),
        )


def test_freeze_requires_one_common_zero_budget_match_tolerance():
    cfg, seq = _config_and_rule()
    d1, d2, d3 = _rules()
    mismatched_d2 = TransferRule(
        min_positive_budget_points=d2.min_positive_budget_points,
        min_mean_delta=d2.min_mean_delta,
        max_zero_budget_abs_delta=0.05,
    )
    with pytest.raises(ValueError, match="match tolerance"):
        freeze_preregistration(cfg, seq, d1, mismatched_d2, d3, (), (101,), (202,), SPEC_COMMIT)
