from dataclasses import replace

import pytest

from aaa_v0.provenance import ProtocolOrderError, ProtocolPhase, SeedCustody, seed_commitment


def test_future_seed_cannot_be_revealed_before_terminal_freeze():
    custody = SeedCustody(history_seed=11, sealed_future_seed=29)
    assert custody.phase is ProtocolPhase.HISTORY
    with pytest.raises(ProtocolOrderError):
        custody.reveal_future_seed()

    snapshot = custody.freeze_terminal(
        treatment_state_hash="a" * 64,
        control_state_hash="b" * 64,
        history_pair_hash="c" * 64,
        config_hash="d" * 64,
    )
    assert custody.phase is ProtocolPhase.TERMINAL_FROZEN
    reveal = custody.reveal_future_seed(snapshot)
    assert reveal.seed == 29
    assert reveal.commitment == seed_commitment(29)
    assert custody.phase is ProtocolPhase.FUTURE_REVEALED


def test_snapshot_tamper_and_cross_custody_are_rejected():
    custody = SeedCustody(history_seed=11, sealed_future_seed=29)
    snapshot = custody.freeze_terminal("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    tampered = replace(snapshot, config_hash="e" * 64)
    with pytest.raises(ProtocolOrderError):
        custody.reveal_future_seed(tampered)

    other = SeedCustody(history_seed=11, sealed_future_seed=31)
    other.freeze_terminal("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    with pytest.raises(ProtocolOrderError):
        other.reveal_future_seed(snapshot)
