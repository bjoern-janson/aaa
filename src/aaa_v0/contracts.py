from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class AssayConfig:
    z_count: int
    q_count: int
    theta_count: int
    history_repeats: int
    future_tasks_per_family: int
    max_budget: int
    d3_change_after: int

    def __post_init__(self) -> None:
        fields = (
            self.z_count,
            self.q_count,
            self.theta_count,
            self.history_repeats,
            self.future_tasks_per_family,
        )
        if any(value <= 0 for value in fields):
            raise ValueError("all count fields must be positive")
        if self.q_count != self.z_count:
            raise ValueError("AAA-v0 requires q_count == z_count")
        if self.q_count < 2:
            raise ValueError("AAA-v0 requires at least two context/probe roles")
        if self.theta_count < 2:
            raise ValueError("AAA-v0 requires at least two target values")
        if self.history_repeats % self.theta_count != 0:
            raise ValueError("AAA-v0 exact history balancing requires history_repeats % theta_count == 0")
        if self.max_budget < 1:
            raise ValueError("max_budget must be >= 1")
        if self.d3_change_after < 1:
            raise ValueError("d3_change_after must be >= 1")
        if self.d3_change_after >= self.future_tasks_per_family:
            raise ValueError("d3_change_after must leave at least one post-change D3 task")

    @classmethod
    def exploratory_default(cls) -> "AssayConfig":
        return cls(4, 4, 4, 8, 32, 4, 8)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceBalanceRule:
    max_q_transition_l1: float
    max_q_run_length_l1: float
    max_lagged_mi_delta: float
    lag_window: int
    max_search_attempts: int

    def __post_init__(self) -> None:
        if any(v < 0 for v in (self.max_q_transition_l1, self.max_q_run_length_l1, self.max_lagged_mi_delta)):
            raise ValueError("sequence tolerances must be non-negative")
        if self.lag_window < 1:
            raise ValueError("lag_window must be >= 1")
        if self.max_search_attempts < 1:
            raise ValueError("max_search_attempts must be >= 1")


@dataclass(frozen=True)
class HistoryEpisode:
    episode_id: str
    z: int
    theta: int
    q_star: int
    surface_seed: int
    resource_units: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryPair:
    treatment: tuple[HistoryEpisode, ...]
    control: tuple[HistoryEpisode, ...]
    treatment_phi: tuple[int, ...]


@dataclass(frozen=True)
class LatentTask:
    task_id: str
    family: Literal["D1", "D2", "D3"]
    z: int
    theta: int
    q_star: int
    surface_seed: int
    phase_index: int


@dataclass(frozen=True)
class SurfaceView:
    task_id: str
    family: str
    context_signature: tuple[int, ...]
    context_token: str
    probe_signatures: tuple[tuple[int, ...], ...]
    probe_tokens: tuple[str, ...]


@dataclass(frozen=True)
class SolvedHistoryView:
    surface: SurfaceView
    informative_probe_signature: tuple[int, ...]
    resolved_theta: int


@dataclass(frozen=True)
class ProbeObservation:
    probe_signature: tuple[int, ...]
    informative: bool
    value: int | None


@dataclass(frozen=True)
class TrialRecord:
    condition: Literal["treatment", "control"]
    agent_name: str
    task_id: str
    family: str
    budget: int
    guess: int
    theta: int
    correct: bool
    chosen_probe_signature: tuple[int, ...] | None
    old_rule_adherence: bool | None


@dataclass(frozen=True)
class ProtocolStatus:
    balance_ok: bool
    custody_ok: bool
    match_ok: bool
    controls_ok: bool
    stop_code: str | None
