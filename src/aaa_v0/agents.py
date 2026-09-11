from __future__ import annotations

from collections import Counter, defaultdict
from typing import Protocol

from aaa_v0.contracts import ProbeObservation, SolvedHistoryView, SurfaceView

Signature = tuple[int, ...]


class Agent(Protocol):
    def observe_history(self, records: tuple[SolvedHistoryView, ...]) -> None: ...
    def start_task(self, view: SurfaceView) -> None: ...
    def choose_probe(self) -> Signature: ...
    def observe_probe(self, observation: ProbeObservation) -> None: ...
    def guess_theta(self) -> int: ...
    def state_dict(self) -> dict[str, object]: ...
    def preferred_probe(self, context_signature: Signature) -> Signature | None: ...


class LocalAgent:
    name = "LOCAL"

    def __init__(self, theta_count: int) -> None:
        self.theta_count = theta_count
        self._view: SurfaceView | None = None
        self._attempted: list[Signature] = []
        self._observed_value: int | None = None

    def observe_history(self, records: tuple[SolvedHistoryView, ...]) -> None:
        return None

    def start_task(self, view: SurfaceView) -> None:
        self._view = view
        self._attempted = []
        self._observed_value = None

    def choose_probe(self) -> Signature:
        if self._view is None:
            raise RuntimeError("no active task")
        for signature in self._view.probe_signatures:
            if signature not in self._attempted:
                self._attempted.append(signature)
                return signature
        raise RuntimeError("no untried probes remain")

    def observe_probe(self, observation: ProbeObservation) -> None:
        if observation.informative:
            self._observed_value = observation.value

    def guess_theta(self) -> int:
        return self._observed_value if self._observed_value is not None else 0

    def preferred_probe(self, context_signature: Signature) -> Signature | None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "active_task": None if self._view is None else self._view.task_id,
            "attempted": [list(x) for x in self._attempted],
            "observed_value": self._observed_value,
        }


class FixedMetaAgent(LocalAgent):
    name = "FIXED_META"

    def __init__(self, theta_count: int) -> None:
        super().__init__(theta_count)
        self._mapping: dict[Signature, Signature] = {}

    def observe_history(self, records: tuple[SolvedHistoryView, ...]) -> None:
        counts: dict[Signature, Counter[Signature]] = defaultdict(Counter)
        for record in records:
            counts[record.surface.context_signature][record.informative_probe_signature] += 1
        mapping: dict[Signature, Signature] = {}
        for context, counter in counts.items():
            max_count = max(counter.values())
            winners = sorted(sig for sig, count in counter.items() if count == max_count)
            mapping[context] = winners[0]
        self._mapping = mapping

    def choose_probe(self) -> Signature:
        if self._view is None:
            raise RuntimeError("no active task")
        if not self._attempted:
            preferred = self._mapping.get(self._view.context_signature)
            if preferred is not None and preferred in self._view.probe_signatures:
                self._attempted.append(preferred)
                return preferred
        return super().choose_probe()

    def preferred_probe(self, context_signature: Signature) -> Signature | None:
        return self._mapping.get(context_signature)

    def state_dict(self) -> dict[str, object]:
        base = super().state_dict()
        base["name"] = self.name
        base["mapping"] = [
            {"context": list(context), "probe": list(probe)}
            for context, probe in sorted(self._mapping.items())
        ]
        return base


class RevisingMetaAgent(FixedMetaAgent):
    name = "REVISING_META"

    def __init__(self, theta_count: int) -> None:
        super().__init__(theta_count)
        self._preferred_failed = False

    def start_task(self, view: SurfaceView) -> None:
        super().start_task(view)
        self._preferred_failed = False

    def observe_probe(self, observation: ProbeObservation) -> None:
        if self._view is None:
            raise RuntimeError("no active task")
        preferred = self._mapping.get(self._view.context_signature)
        if preferred is not None and observation.probe_signature == preferred and not observation.informative:
            self._preferred_failed = True
        if observation.informative:
            self._observed_value = observation.value
            if self._preferred_failed:
                self._mapping[self._view.context_signature] = observation.probe_signature

    def state_dict(self) -> dict[str, object]:
        base = super().state_dict()
        base["name"] = self.name
        base["preferred_failed"] = self._preferred_failed
        return base
