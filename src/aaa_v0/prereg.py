from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from aaa_v0.contracts import AssayConfig, SequenceBalanceRule
from aaa_v0.runner import RunBundle
from aaa_v0.serialization import sha256_json

SPEC_COMMIT = "04f0f4380d4fad01386c1f61e95a1e65c24f8de0"


def seed_identity(kind: str, seed: int) -> str:
    if kind not in {"history", "future"}:
        raise ValueError("seed kind must be history or future")
    return sha256_json({"kind": kind, "seed": int(seed)})


@dataclass(frozen=True)
class TransferRule:
    min_positive_budget_points: int
    min_mean_delta: float
    max_zero_budget_abs_delta: float

    def __post_init__(self) -> None:
        if self.min_positive_budget_points < 0:
            raise ValueError("min_positive_budget_points must be non-negative")
        if self.max_zero_budget_abs_delta < 0:
            raise ValueError("max_zero_budget_abs_delta must be non-negative")


@dataclass(frozen=True)
class RecoveryRule:
    max_post_change_old_rule_adherence: float
    min_post_change_competence: float
    evaluation_start_offset: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_post_change_old_rule_adherence <= 1.0:
            raise ValueError("max adherence must lie in [0,1]")
        if not 0.0 <= self.min_post_change_competence <= 1.0:
            raise ValueError("min competence must lie in [0,1]")
        if self.evaluation_start_offset < 0:
            raise ValueError("evaluation_start_offset must be non-negative")


@dataclass(frozen=True)
class ExploratoryCalibrationReport:
    control_summaries: tuple[tuple[str, dict[str, object]], ...]
    candidate_rules: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "EXPLORATORY_UNSCORED",
            "next_phase_gate_passed": False,
            "aaa_evidence": "NONE",
            "control_summaries": dict(self.control_summaries),
            "candidate_rules": self.candidate_rules,
        }


@dataclass(frozen=True)
class Preregistration:
    config: AssayConfig
    sequence_balance_rule: SequenceBalanceRule
    d1_rule: TransferRule
    d2_rule: TransferRule
    d3_rule: RecoveryRule
    exploratory_seed_hashes: tuple[str, ...]
    confirmatory_history_seeds: tuple[int, ...]
    confirmatory_future_seeds: tuple[int, ...]
    spec_commit: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Preregistration":
        return cls(
            config=AssayConfig(**dict(value["config"])),  # type: ignore[arg-type]
            sequence_balance_rule=SequenceBalanceRule(**dict(value["sequence_balance_rule"])),  # type: ignore[arg-type]
            d1_rule=TransferRule(**dict(value["d1_rule"])),  # type: ignore[arg-type]
            d2_rule=TransferRule(**dict(value["d2_rule"])),  # type: ignore[arg-type]
            d3_rule=RecoveryRule(**dict(value["d3_rule"])),  # type: ignore[arg-type]
            exploratory_seed_hashes=tuple(value["exploratory_seed_hashes"]),  # type: ignore[arg-type]
            confirmatory_history_seeds=tuple(int(x) for x in value["confirmatory_history_seeds"]),  # type: ignore[arg-type]
            confirmatory_future_seeds=tuple(int(x) for x in value["confirmatory_future_seeds"]),  # type: ignore[arg-type]
            spec_commit=str(value["spec_commit"]),
        )


@dataclass(frozen=True)
class ConfirmatoryControlReport:
    status: str
    next_phase_gate_passed: bool
    aaa_evidence: str
    signature: tuple[tuple[str, bool, bool, bool], ...]
    preregistration_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def freeze_preregistration(
    config: AssayConfig,
    sequence_balance_rule: SequenceBalanceRule,
    d1_rule: TransferRule,
    d2_rule: TransferRule,
    d3_rule: RecoveryRule,
    exploratory_seed_hashes: tuple[str, ...],
    confirmatory_history_seeds: tuple[int, ...],
    confirmatory_future_seeds: tuple[int, ...],
    spec_commit: str,
) -> Preregistration:
    if spec_commit != SPEC_COMMIT:
        raise ValueError("spec_commit does not match frozen AAA-v0 design")
    if d1_rule.max_zero_budget_abs_delta != d2_rule.max_zero_budget_abs_delta:
        raise ValueError("D1 and D2 must freeze one common zero-budget match tolerance")
    exploratory = set(exploratory_seed_hashes)
    confirmatory = {
        *(seed_identity("history", seed) for seed in confirmatory_history_seeds),
        *(seed_identity("future", seed) for seed in confirmatory_future_seeds),
    }
    if exploratory & confirmatory:
        raise ValueError("exploratory/confirmatory seed overlap")
    return Preregistration(
        config=config,
        sequence_balance_rule=sequence_balance_rule,
        d1_rule=d1_rule,
        d2_rule=d2_rule,
        d3_rule=d3_rule,
        exploratory_seed_hashes=tuple(exploratory_seed_hashes),
        confirmatory_history_seeds=tuple(confirmatory_history_seeds),
        confirmatory_future_seeds=tuple(confirmatory_future_seeds),
        spec_commit=spec_commit,
    )


def _transfer_passes(curve, rule: TransferRule) -> bool:
    if not curve.budgets:
        return False
    zero_idx = curve.budgets.index(0) if 0 in curve.budgets else None
    if zero_idx is None or abs(curve.delta[zero_idx]) > rule.max_zero_budget_abs_delta:
        return False
    later = [d for b, d in zip(curve.budgets, curve.delta, strict=True) if b > 0]
    if not later:
        return False
    positive_points = sum(d > 0.0 for d in later)
    mean_delta = sum(later) / len(later)
    return positive_points >= rule.min_positive_budget_points and mean_delta >= rule.min_mean_delta


def _recovery_passes_with_config(bundle: RunBundle, rule: RecoveryRule, config: AssayConfig) -> bool:
    trajectory = bundle.d3_treatment_recovery
    cutoff = config.d3_change_after + rule.evaluation_start_offset
    selected = [
        (adherence, competence)
        for phase, adherence, competence in zip(
            trajectory.phase_indices,
            trajectory.old_rule_adherence,
            trajectory.competence,
            strict=True,
        )
        if phase >= cutoff
    ]
    if not selected:
        return False
    mean_adherence = sum(a for a, _ in selected) / len(selected)
    mean_competence = sum(c for _, c in selected) / len(selected)
    return (
        mean_adherence <= rule.max_post_change_old_rule_adherence
        and mean_competence >= rule.min_post_change_competence
    )


def validate_confirmatory_controls(
    preregistration: Preregistration,
    bundles: Mapping[str, RunBundle],
    *,
    sequence_balance_rule: SequenceBalanceRule,
    history_seeds: tuple[int, ...],
    future_seeds: tuple[int, ...],
) -> ConfirmatoryControlReport:
    if preregistration.spec_commit != SPEC_COMMIT:
        raise ValueError("spec_commit mismatch")
    if sequence_balance_rule != preregistration.sequence_balance_rule:
        raise ValueError("sequence balance rule mismatch")
    if tuple(history_seeds) != preregistration.confirmatory_history_seeds or tuple(future_seeds) != preregistration.confirmatory_future_seeds:
        raise ValueError("confirmatory seed schedule mismatch")
    confirmatory_ids = {
        *(seed_identity("history", seed) for seed in history_seeds),
        *(seed_identity("future", seed) for seed in future_seeds),
    }
    if confirmatory_ids & set(preregistration.exploratory_seed_hashes):
        raise ValueError("exploratory/confirmatory seed overlap")

    expected_names = ("LOCAL", "FIXED_META", "REVISING_META")
    if set(bundles) != set(expected_names):
        raise ValueError("confirmatory bundle set mismatch")
    config_hash = sha256_json(preregistration.config.to_dict())

    signature: list[tuple[str, bool, bool, bool]] = []
    lower_gate_failure = False
    for name in expected_names:
        bundle = bundles[name]
        status = bundle.protocol_status
        if not (status.balance_ok and status.custody_ok and status.match_ok):
            lower_gate_failure = True
        if bundle.config_hash != config_hash:
            raise ValueError("config hash mismatch")
        d1 = _transfer_passes(bundle.d1_paired_curve, preregistration.d1_rule)
        d2 = _transfer_passes(bundle.d2_paired_curve, preregistration.d2_rule)
        recovery = _recovery_passes_with_config(bundle, preregistration.d3_rule, preregistration.config)
        d3 = d1 and d2 and recovery
        signature.append((name, d1, d2, d3))

    frozen_signature = tuple(signature)
    required = (
        ("LOCAL", False, False, False),
        ("FIXED_META", True, True, False),
        ("REVISING_META", True, True, True),
    )
    ok = (not lower_gate_failure) and frozen_signature == required
    return ConfirmatoryControlReport(
        status="CONTROL_VALIDATED" if ok else "CONTROL_MISMATCH",
        next_phase_gate_passed=ok,
        aaa_evidence="NONE",
        signature=frozen_signature,
        preregistration_hash=sha256_json(preregistration.to_dict()),
    )
