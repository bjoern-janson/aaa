from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Iterable

from aaa_v0.contracts import HistoryEpisode, HistoryPair, SequenceBalanceRule
from aaa_v0.serialization import sha256_json


def empirical_mutual_information(pairs: Iterable[tuple[int, int]]) -> float:
    data = tuple(pairs)
    if not data:
        return 0.0
    joint = Counter(data)
    xs = Counter(x for x, _ in data)
    ys = Counter(y for _, y in data)
    n = len(data)
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        px = xs[x] / n
        py = ys[y] / n
        mi += pxy * math.log(pxy / (px * py))
    return 0.0 if abs(mi) < 1e-15 else mi


def _transition_distribution(values: tuple[int, ...]) -> dict[tuple[int, int], float]:
    if len(values) < 2:
        return {}
    counts = Counter(pairwise(values))
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _run_length_distribution(values: tuple[int, ...]) -> dict[int, float]:
    if not values:
        return {}
    lengths: list[int] = []
    current = values[0]
    run = 1
    for value in values[1:]:
        if value == current:
            run += 1
        else:
            lengths.append(run)
            current = value
            run = 1
    lengths.append(run)
    counts = Counter(lengths)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _l1(a: dict[object, float], b: dict[object, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _lagged_mi(zs: tuple[int, ...], qs: tuple[int, ...], lag: int) -> float:
    if lag > 0:
        pairs = zip(zs[:-lag], qs[lag:])
    else:
        k = -lag
        pairs = zip(zs[k:], qs[:-k])
    return empirical_mutual_information(pairs)


@dataclass(frozen=True)
class SequenceDiagnostics:
    q_transition_l1: float
    q_run_length_l1: float
    lagged_mi_delta: tuple[float, ...]


def sequence_diagnostics(
    treatment_q: tuple[int, ...],
    control_q: tuple[int, ...],
    z_sequence: tuple[int, ...],
    rule: SequenceBalanceRule,
) -> SequenceDiagostics:
    lags_neg = tuple(range(-rule.lag_window, 0))
    lags_pos = tuple(range(1, rule.lag_window + 1))
    lags = lags_neg + lags_pos
    return SequenceDiagnostics(
        q_transition_l1=_l1(
            _transition_distribution(treatment_q),
            _transition_distribution(control_q),
        ),
        q_run_length_l1=_l1(
            _run_length_distribution(treatment_q),
            _run_length_distribution(control_q),
        ),
        lagged_mi_delta=tuple(
            _lagged_mi(z_sequence, treatment_q, lag)
            - _lagged_mi(z_sequence, control_q, lag)
            for lag in lags
        ),
    )


def sequence_passes(diag: SequenceDiagnostics, rule: SequenceBalanceRule) -> bool:
    return (
        diag.q_transition_l1 <= rule.max_q_transition_l1
        and diag.q_run_length_l1 <= rule.max_q_run_length_l1
        and all(abs(v) <= rule.max_lagged_mi_delta for v in diag.lagged_mi_delta)
    )


@dataclass(frozen=True)
class BalanceReport:
    ok: bool
    mismatched_fields: tuple[str, ...]
    treatment_mi: float
    control_mi: float
    treatment_q_theta_mi: float
    control_q_theta_mi: float
    sequence: SequenceDiagnostics
    treatment_hash: str
    control_hash: str
    non_treatment_sequence_hash_treatment: str
    non_treatment_sequence_hash_control: str



def _redacted(items: tuple[HistoryEpisode, ...]) -> list[dict[str, object]]:
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


def audit_history_pair(pair: HistoryPair, rule: SequenceBalanceRule) -> BalanceReport:
    mismatches: list[str] = []
    treatment_redacted = _redacted(pair.treatment)
    control_redacted = _redacted(pair.control)
    treatment_non_hash = sha256_json(treatment_redacted)
    control_non_hash = sha256_json(control_redacted)
    if treatment_non_hash != control_non_hash:
        mismatches.append("non_treatment_sequence")

    tq = tuple(e.q_star for e in pair.treatment)
    cq = tuple(e.q_star for e in pair.control)
    if Counter(tq) != Counter(cq):
        mismatches.append("q_star_marginal")

    treatment_q_theta = Counter((e.q_star, e.theta) for e in pair.treatment)
    control_q_theta = Counter((e.q_star, e.theta) for e in pair.control)
    if treatment_q_theta != control_q_theta:
        mismatches.append("q_theta_joint")

    tmi = empirical_mutual_information((e.z, e.q_star) for e in pair.treatment)
    cmi = empirical_mutual_information((e.z, e.q_star) for e in pair.control)
    tqtheta_mi = empirical_mutual_information((e.q_star, e.theta) for e in pair.treatment)
    cqtheta_mi = empirical_mutual_information((e.q_star, e.theta) for e in pair.control)
    if not tmi > 0.0:
        mismatches.append("treatment_mi")
    if cmi != 0.0:
        mismatches.append("control_mi")

    zseq = tuple(e.z for e in pair.treatment)
    diag = sequence_diagnostics(tq, cq, zseq, rule)
    if not sequence_passes(diag, rule):
        mismatches.append("sequence_balance")

    return BalanceReport(
        ok=not mismatches,
        mismatched_fields=tuple(mismatches),
        treatment_mi=tmi,
        control_mi=cmi,
        treatment_q_theta_mi=tqtheta_mi,
        control_q_theta_mi=cqtheta_mi,
        sequence=diag,
        treatment_hash=sha256_json([e.to_dict() for e in pair.treatment]),
        control_hash=sha256_json([e.to_dict() for e in pair.control]),
        non_treatment_sequence_hash_treatment=treatment_non_hash,
        non_treatment_sequence_hash_control=control_non_hash,
    )
