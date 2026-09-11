from __future__ import annotations

import hashlib

import numpy as np

from aaa_v0.balance import sequence_diagnostics, sequence_passes
from aaa_v0.contracts import AssayConfig, HistoryEpisode, HistoryPair, SequenceBalanceRule


class BalanceConstructionError(RuntimeError):
    pass


def _seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"aaa-v0:{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _rng(base: int, label: str) -> np.random.Generator:
    return np.random.default_rng(_seed(base, label))


def make_history_pair(
    config: AssayConfig,
    seed: int,
    sequence_rule: SequenceBalanceRule,
) -> HistoryPair:
    phi_rng = _rng(seed, "history-phi")
    treatment_phi = tuple(int(x) for x in phi_rng.permutation(config.q_count))

    z_values = []
    for z in range(config.z_count):
        z_values.extend([z] * (config.history_repeats * config.q_count))
    z_values = np.array(z_values, dtype=int)
    order = _rng(seed, "history-order").permutation(len(z_values))
    z_ordered = tuple(int(x) for x in z_values[order])

    theta_rng = _rng(seed, "history-theta")
    thetas = tuple(int(x) for x in theta_rng.integers(0, config.theta_count, len(z_ordered)))
    surface_rng = _rng(seed, "history-surface")
    surface_seeds = tuple(int(x) for x in surface_rng.integers(0, 2**31 - 1, len(z_ordered)))

    treatment_q = tuple(treatment_phi[z] for z in z_ordered)
    positions_by_z = {
        z: [i for i, observed_z in enumerate(z_ordered) if observed_z == z]
        for z in range(config.z_count)
    }
    control_rng = _rng(seed, "history-control-q")

    for _ in range(sequence_rule.max_search_attempts):
        control_q_list = [0] * len(z_ordered)
        for z, positions in positions_by_z.items():
            multiset = np.repeat(np.arange(config.q_count), config.history_repeats)
            shuffled = control_rng.permutation(multiset)
            for pos, q in zip(positions, shuffled, strict=True):
                control_q_list[pos] = int(q)
        control_q = tuple(control_q_list)
        diag = sequence_diagnostics(treatment_q, control_q, z_ordered, sequence_rule)
        if sequence_passes(diag, sequence_rule):
            break
    else:
        raise BalanceConstructionError("no control assignment satisfied sequence balance rule")

    treatment = tuple(
        HistoryEpisode(
            episode_id=f"h-{i:04d}",
            z=z,
            theta=thetas[i],
            q_star=treatment_q[i],
            surface_seed=surface_seeds[i],
            resource_units=1,
        )
        for i, z in enumerate(z_ordered)
    )
    control = tuple(
        HistoryEpisode(
            episode_id=f"h-{i:04d}",
            z=z,
            theta=thetas[i],
            q_star=control_q[i],
            surface_seed=surface_seeds[i],
            resource_units=1,
        )
        for i, z in enumerate(z_ordered)
    )
    return HistoryPair(treatment=treatment, control=control, treatment_phi=treatment_phi)
