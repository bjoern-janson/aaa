from collections import Counter

import pytest

from aaa_v0.contracts import AssayConfig, SequenceBalanceRule
from aaa_v0.histories import BalanceConstructionError, make_history_pair
from aaa_v0.balance import empirical_mutual_information


def test_history_pair_has_exact_marginals_and_only_target_dependency():
    cfg = AssayConfig.exploratory_default()
    rule = SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)
    pair = make_history_pair(cfg, seed=17, sequence_rule=rule)
    assert sorted(pair.treatment_phi) == list(range(cfg.q_count))
    assert [e.z for e in pair.treatment] == [e.z for e in pair.control]
    assert [e.theta for e in pair.treatment] == [e.theta for e in pair.control]
    assert [e.surface_seed for e in pair.treatment] == [e.surface_seed for e in pair.control]
    assert [e.resource_units for e in pair.treatment] == [e.resource_units for e in pair.control]
    assert sorted(e.q_star for e in pair.treatment) == sorted(e.q_star for e in pair.control)
    assert all(e.q_star == pair.treatment_phi[e.z] for e in pair.treatment)
    assert empirical_mutual_information((e.z, e.q_star) for e in pair.treatment) > 0.0
    assert empirical_mutual_information((e.z, e.q_star) for e in pair.control) == 0.0
    assert Counter((e.q_star, e.theta) for e in pair.treatment) == Counter(
        (e.q_star, e.theta) for e in pair.control
    )
    assert empirical_mutual_information((e.q_star, e.theta) for e in pair.treatment) == 0.0
    assert empirical_mutual_information((e.q_star, e.theta) for e in pair.control) == 0.0


def test_impossible_sequence_rule_fails_closed():
    cfg = AssayConfig.exploratory_default()
    impossible = SequenceBalanceRule(0.0, 0.0, 0.0, 3, 1)
    with pytest.raises(BalanceConstructionError):
        make_history_pair(cfg, seed=17, sequence_rule=impossible)
