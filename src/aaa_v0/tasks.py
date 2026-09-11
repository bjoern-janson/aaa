from __future__ import annotations

import hashlib

import numpy as np

from aaa_v0.contracts import (
    AssayConfig,
    HistoryEpisode,
    LatentTask,
    SolvedHistoryView,
    SurfaceView,
)
from aaa_v0.provenance import FutureSeedReveal, seed_commitment


def _derived_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"aaa-v0:{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(base: int, label: str) -> np.random.Generator:
    return np.random.default_rng(_derived_seed(base, label))


def context_signature(z: int, width: int) -> tuple[int, ...]:
    if not 0 <= z < width:
        raise ValueError("context role out of range")
    return tuple(1 if i == z else 0 for i in range(width))


def probe_signature(q: int, width: int) -> tuple[int, ...]:
    if not 0 <= q < width:
        raise ValueError("probe role out of range")
    return tuple(1 if i == q else 0 for i in range(width))


def _nonidentity_permutation(width: int, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    perm = tuple(int(x) for x in rng.permutation(width))
    identity = tuple(range(width))
    if perm == identity and width > 1:
        perm = identity[1:] + identity[:1]
    return perm


def render_task(task: LatentTask, config: AssayConfig) -> SurfaceView:
    ctx_sig = context_signature(task.z, config.z_count)
    structural_probes = tuple(probe_signature(q, config.q_count) for q in range(config.q_count))
    if task.family == "D2":
        order = _nonidentity_permutation(config.q_count, task.surface_seed)
        rename = _nonidentity_permutation(config.q_count, task.surface_seed ^ 0xA5A5A5A5)
        probe_sigs = tuple(structural_probes[i] for i in order)
        probe_tokens = tuple(f"shift-probe-{rename[i]}" for i in order)
        context_token = f"shift-context-{(task.z + 1) % config.z_count}"
    else:
        probe_sigs = structural_probes
        probe_tokens = tuple(f"probe-{q}" for q in range(config.q_count))
        context_token = f"context-{task.z}"
    return SurfaceView(
        task_id=task.task_id,
        family=task.family,
        context_signature=ctx_sig,
        context_token=context_token,
        probe_signatures=probe_sigs,
        probe_tokens=probe_tokens,
    )


def render_history_episode(episode: HistoryEpisode, config: AssayConfig) -> SolvedHistoryView:
    latent = LatentTask(
        task_id=episode.episode_id,
        family="D1",
        z=episode.z,
        theta=episode.theta,
        q_star=episode.q_star,
        surface_seed=episode.surface_seed,
        phase_index=0,
    )
    return SolvedHistoryView(
        surface=render_task(latent, config),
        informative_probe_signature=probe_signature(episode.q_star, config.q_count),
        resolved_theta=episode.theta,
    )


def make_future_tasks(
    config: AssayConfig,
    reveal: FutureSeedReveal,
    phi: tuple[int, ...],
) -> dict[str, tuple[LatentTask, ...]]:
    if reveal.commitment != seed_commitment(reveal.seed):
        raise ValueError("future seed reveal does not match commitment")
    if len(phi) != config.z_count or sorted(phi) != list(range(config.q_count)):
        raise ValueError("phi must be a bijection from contexts to probe roles")

    n = config.future_tasks_per_family
    latent_rng = _rng(reveal.seed, "future-latent")
    offset = 1 + int(latent_rng.integers(0, config.q_count - 1))
    phi_prime = tuple((q + offset) % config.q_count for q in phi)

    paired_z = tuple(int(x) for x in latent_rng.integers(0, config.z_count, n))
    paired_theta = tuple(int(x) for x in latent_rng.integers(0, config.theta_count, n))
    d3_z = tuple(int(x) for x in latent_rng.integers(0, config.z_count, n))
    d3_theta = tuple(int(x) for x in latent_rng.integers(0, config.theta_count, n))

    d1_surface_rng = _rng(reveal.seed, "future-d1-surface")
    d2_surface_rng = _rng(reveal.seed, "future-d2-surface")
    d3_surface_rng = _rng(reveal.seed, "future-d3-surface")
    d1_seeds = [int(x) for x in d1_surface_rng.integers(1, 2**31 - 1, n)]
    d2_seeds = [int(x) for x in d2_surface_rng.integers(1, 2**31 - 1, n)]
    d3_seeds = [int(x) for x in d3_surface_rng.integers(1, 2**31 - 1, n)]
    for i in range(n):
        if d1_seeds[i] == d2_seeds[i]:
            d2_seeds[i] = 1 + (d2_seeds[i] % (2**31 - 2))
            if d2_seeds[i] == d1_seeds[i]:
                d2_seeds[i] = 1 + (d2_seeds[i] % (2**31 - 2))

    d1 = tuple(
        LatentTask(
            task_id=f"D1-{i:04d}",
            family="D1",
            z=z,
            theta=paired_theta[i],
            q_star=phi[z],
            surface_seed=d1_seeds[i],
            phase_index=i,
        )
        for i, z in enumerate(paired_z)
    )
    d2 = tuple(
        LatentTask(
            task_id=f"D2-{i:04d}",
            family="D2",
            z=z,
            theta=paired_theta[i],
            q_star=phi[z],
            surface_seed=d2_seeds[i],
            phase_index=i,
        )
        for i, z in enumerate(paired_z)
    )
    d3 = tuple(
        LatentTask(
            task_id=f"D3-{i:04d}",
            family="D3",
            z=z,
            theta=d3_theta[i],
            q_star=(phi[z] if i < config.d3_change_after else phi_prime[z]),
            surface_seed=d3_seeds[i],
            phase_index=i,
        )
        for i, z in enumerate(d3_z)
    )
    return {"D1": d1, "D2": d2, "D3": d3}
