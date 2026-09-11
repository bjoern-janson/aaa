from __future__ import annotations

from aaa_v0.contracts import LatentTask, ProbeObservation


class BudgetExceeded(RuntimeError):
    pass


def _signature_index(signature: tuple[int, ...], width: int) -> int:
    if len(signature) != width or sum(signature) != 1 or any(v not in (0, 1) for v in signature):
        raise ValueError("probe signature must be a one-hot structural role")
    return signature.index(1)


class MetaProbeEnvironment:
    def __init__(self, task: LatentTask, q_count: int, max_budget: int) -> None:
        self.task = task
        self.q_count = q_count
        self.max_budget = max_budget
        self.used_budget = 0

    def probe(self, signature: tuple[int, ...]) -> ProbeObservation:
        if self.used_budget >= self.max_budget:
            raise BudgetExceeded("probe budget exhausted")
        self.used_budget += 1
        q = _signature_index(signature, self.q_count)
        if q == self.task.q_star:
            return ProbeObservation(signature, True, self.task.theta)
        return ProbeObservation(signature, False, None)
