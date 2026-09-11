from aaa_v0.contracts import TrialRecord
from aaa_v0.measurement import compute_paired_curve, compute_recovery_trajectory


def _r(condition, task, budget, correct, *, family="D1", old=None):
    return TrialRecord(
        condition=condition,
        agent_name="A",
        task_id=task,
        family=family,
        budget=budget,
        guess=1 if correct else 0,
        theta=1,
        correct=correct,
        chosen_probe_signature=None,
        old_rule_adherence=old,
    )


def test_paired_curve_preserves_intermediate_geometry_when_final_scores_match():
    records = (
        _r("treatment", "D1-0000", 0, False),
        _r("treatment", "D1-0000", 1, True),
        _r("treatment", "D1-0000", 2, True),
        _r("control", "D1-0000", 0, False),
        _r("control", "D1-0000", 1, False),
        _r("control", "D1-0000", 2, True),
    )
    curve = compute_paired_curve(records, family="D1")
    assert curve.budgets == (0, 1, 2)
    assert curve.treatment == (0.0, 1.0, 1.0)
    assert curve.control == (0.0, 0.0, 1.0)
    assert curve.delta == (0.0, 1.0, 0.0)
    assert curve.secondary_auc_delta == 1.0


def test_recovery_distinguishes_brittle_and_revising_with_equal_final_competence():
    brittle = []
    revising = []
    for i in range(4):
        for budget in (0, 1, 2):
            brittle.append(_r("treatment", f"D3-{i:04d}", budget, budget == 2, family="D3", old=(True if budget == 1 else None)))
            revising.append(_r("treatment", f"D3-{i:04d}", budget, budget == 2, family="D3", old=((i < 2) if budget == 1 else None)))
    b = compute_recovery_trajectory(tuple(brittle), change_after=0)
    r = compute_recovery_trajectory(tuple(revising), change_after=0)
    assert b.competence == r.competence
    assert b.old_rule_adherence != r.old_rule_adherence
    assert b.old_rule_adherence == (1.0, 1.0, 1.0, 1.0)
    assert r.old_rule_adherence == (1.0, 1.0, 0.0, 0.0)
