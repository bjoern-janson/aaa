from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from aaa_v0.contracts import TrialRecord


@dataclass(frozen=True)
class LearningCurve:
    budgets: tuple[int, ...]
    competence: tuple[float, ...]


@dataclass(frozen=True)
class PairedCurve:
    budgets: tuple[int, ...]
    treatment: tuple[float, ...]
    control: tuple[float, ...]
    delta: tuple[float, ...]
    secondary_auc_delta: float


@dataclass(frozen=True)
class RecoveryTrajectory:
    phase_indices: tuple[int, ...]
    old_rule_adherence: tuple[float, ...]
    competence: tuple[float, ...]
    cumulative_post_change_evidence: tuple[int, ...]


def compute_learning_curve(
    records: tuple[TrialRecord, ...],
    *,
    condition: str,
    family: str,
) -> LearningCurve:
    selected = [r for r in records if r.condition == condition and r.family == family]
    budgets = tuple(sorted({r.budget for r in selected}))
    competence = []
    for budget in budgets:
        vals = [float(r.correct) for r in selected if r.budget == budget]
        competence.append(sum(vals) / len(vals) if vals else 0.0)
    return LearningCurve(budgets, tuple(competence))


def _trapezoid(values: tuple[float, ...], budgets: tuple[int, ...]) -> float:
    total = 0.0
    for i in range(1, len(values)):
        width = budgets[i] - budgets[i - 1]
        total += width * (values[i] + values[i - 1]) / 2.0
    return total


def compute_paired_curve(records: tuple[TrialRecord, ...], *, family: str) -> PairedCurve:
    treatment_curve = compute_learning_curve(records, condition="treatment", family=family)
    control_curve = compute_learning_curve(records, condition="control", family=family)
    if treatment_curve.budgets != control_curve.budgets:
        raise ValueError("treatment/control budget grids differ")
    delta = tuple(t - c for t, c in zip(treatment_curve.competence, control_curve.competence, strict=True))
    return PairedCurve(
        budgets=treatment_curve.budgets,
        treatment=treatment_curve.competence,
        control=control_curve.competence,
        delta=delta,
        secondary_auc_delta=_trapezoid(delta, treatment_curve.budgets),
    )


def _phase_index(task_id: str) -> int:
    try:
        return int(task_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"task_id lacks numeric phase suffix: {task_id}") from exc


def compute_recovery_trajectory(
    records: tuple[TrialRecord, ...],
    *,
    change_after: int,
) -> RecoveryTrajectory:
    selected = [r for r in records if r.condition == "treatment" and r.family == "D3"]
    by_task: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in selected:
        by_task[record.task_id].append(record)
    ordered = sorted(by_task, key=_phase_index)

    phases: list[int] = []
    adherence: list[float] = []
    competence: list[float] = []
    cumulative: list[int] = []
    evidence = 0
    for task_id in ordered:
        task_records = sorted(by_task[task_id], key=lambda r: r.budget)
        phase = _phase_index(task_id)
        old_values = [r.old_rule_adherence for r in task_records if r.old_rule_adherence is not None]
        old = bool(old_values[0]) if old_values else False
        final_record = max(task_records, key=lambda r: r.budget)
        if phase >= change_after and old:
            evidence += 1
        phases.append(phase)
        adherence.append(float(old))
        competence.append(float(final_record.correct))
        cumulative.append(evidence)
    return RecoveryTrajectory(
        phase_indices=tuple(phases),
        old_rule_adherence=tuple(adherence),
        competence=tuple(competence),
        cumulative_post_change_evidence=tuple(cumulative),
    )
