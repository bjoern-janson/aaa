from dataclasses import replace

from aaa_v0.balance import audit_history_pair
from aaa_v0.contracts import AssayConfig, HistoryPair, SequenceBalanceRule
from aaa_v0.histories import make_history_pair
from aaa_v0.serialization import canonical_json_bytes


def _pair():
    cfg = AssayConfig.exploratory_default()
    rule = SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)
    return make_history_pair(cfg, 17, rule), rule


def test_balance_audit_accepts_valid_pair():
    pair, rule = _pair()
    report = audit_history_pair(pair, rule)
    assert report.ok
    assert report.mismatched_fields == ()
    assert report.treatment_mi > 0.0
    assert report.control_mi == 0.0


def test_redacted_histories_are_byte_identical():
    pair, _ = _pair()
    def redact(items):
        return [
            {
                "episode_id": e.episode_id,
                "z": e.z,
                "theta": e.theta,
                "surface_seed": e.surface_seed,
                "resource_units": e.resource_units,
            }
            for e in items
        ]
    assert canonical_json_bytes(redact(pair.treatment)) == canonical_json_bytes(redact(pair.control))


def test_balance_audit_rejects_theta_resource_and_order_mutations():
    pair, rule = _pair()
    base = list(pair.control)
    mutations = []
    changed_theta = base.copy()
    changed_theta[0] = replace(changed_theta[0], theta=(changed_theta[0].theta + 1) % 4)
    mutations.append(("non_treatment_sequence", changed_theta))
    changed_resource = base.copy()
    changed_resource[0] = replace(changed_resource[0], resource_units=2)
    mutations.append(("non_treatment_sequence", changed_resource))
    changed_order = base.copy()
    changed_order[0], changed_order[1] = changed_order[1], changed_order[0]
    mutations.append(("non_treatment_sequence", changed_order))
    for expected, control in mutations:
        bad = HistoryPair(pair.treatment, tuple(control), pair.treatment_phi)
        report = audit_history_pair(bad, rule)
        assert not report.ok
        assert expected in report.mismatched_fields
