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

    # Build an exactly balanced (z, theta) shell. This prevents the q* treatment
    # from accidentally introducing a second q*--theta regularity in passive history.
    per_theta_per_z = (config.history_repeats * config.q_count) // config.theta_count
    theta_rng = _rng(seed, "history-theta")
    shells: list[tuple[int, int]] = []
    for z in range(config.z_count):
        theta_multiset = np.repeat(np.arange(config.theta_count), per_theta_per_z)
        for theta in theta_rng.permutation(theta_multiset):
            shells.append((z, int(theta)))
    order = _rng(seed, "history-order").permutation(len(shells))
    ordered_shells = tuple(shells[int(i)] for i in order)
    z_ordered = tuple(z for z, _ in ordered_shells)
    thetas = tuple(theta for _, theta in ordered_shells)

    surface_rng = _rng(seed, "history-surface")
    surface_seeds = tuple(int(x) for x in surface_rng.integers(0, 2**31 - 1, len(z_ordered)))

    treatment_q = tuple(treatment_phi[z] for z in z_ordered)
    positions_by_z_theta = {
        (z, theta): [
            i
            for i, (observed_z, observed_theta) in enumerate(ordered_shells)
            if observed_z == z and observed_theta == theta
        ]
        for z in range(config.z_count)
        for theta in range(config.theta_count)
    }
    q_repeats_per_stratum = config.history_repeats // config.theta_count
    control_rng = _rng(seed, "history-control-q")

    for _ in range(sequence_rule.max_search_attempts):
        control_q_list = [0] * len(z_ordered)
        for positions in positions_by_z_theta.values():
            multiset = np.repeat(np.arange(config.q_count), q_repeats_per_stratum)
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
