from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum

from aaa_v0.serialization import sha256_json


class ProtocolOrderError(RuntimeError):
    pass


class ProtocolPhase(Enum):
    HISTORY = "HISTORY"
    TERMINAL_FROZEN = "TERMINAL_FROZEN"
    FUTURE_REVEALED = "FUTURE_REVEALED"


def seed_commitment(seed: int) -> str:
    payload = b"aaa-v0-future-seed:" + str(seed).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TerminalSnapshot:
    treatment_state_hash: str
    control_state_hash: str
    history_pair_hash: str
    config_hash: str
    future_seed_commitment: str
    snapshot_hash: str

    @classmethod
    def build(
        cls,
        treatment_state_hash: str,
        control_state_hash: str,
        history_pair_hash: str,
        config_hash: str,
        future_seed_commitment: str,
    ) -> "TerminalSnapshot":
        core = {
            "treatment_state_hash": treatment_state_hash,
            "control_state_hash": control_state_hash,
            "history_pair_hash": history_pair_hash,
            "config_hash": config_hash,
            "future_seed_commitment": future_seed_commitment,
        }
        return cls(**core, snapshot_hash=sha256_json(core))

    def verify(self) -> bool:
        core = asdict(self)
        claimed = core.pop("snapshot_hash")
        return sha256_json(core) == claimed


@dataclass(frozen=True)
class FutureSeedReveal:
    seed: int
    commitment: str


class SeedCustody:
    def __init__(self, history_seed: int, sealed_future_seed: int) -> None:
        self.history_seed = history_seed
        self._future_seed = sealed_future_seed
        self.future_seed_commitment = seed_commitment(sealed_future_seed)
        self.phase = ProtocolPhase.HISTORY
        self._terminal_snapshot_hash: str | None = None

    def freeze_terminal(
        self,
        treatment_state_hash: str,
        control_state_hash: str,
        history_pair_hash: str,
        config_hash: str,
    ) -> TerminalSnapshot:
        if self.phase is not ProtocolPhase.HISTORY:
            raise ProtocolOrderError("terminal may be frozen exactly once from HISTORY")
        snapshot = TerminalSnapshot.build(
            treatment_state_hash,
            control_state_hash,
            history_pair_hash,
            config_hash,
            self.future_seed_commitment,
        )
        self._terminal_snapshot_hash = snapshot.snapshot_hash
        self.phase = ProtocolPhase.TERMINAL_FROZEN
        return snapshot

    def reveal_future_seed(self, snapshot: TerminalSnapshot | None = None) -> FutureSeedReveal:
        if self.phase is not ProtocolPhase.TERMINAL_FROZEN or snapshot is None:
            raise ProtocolOrderError("future seed is unavailable before terminal freeze")
        if not snapshot.verify():
            raise ProtocolOrderError("terminal snapshot failed integrity verification")
        if snapshot.snapshot_hash != self._terminal_snapshot_hash:
            raise ProtocolOrderError("terminal snapshot does not belong to this custody object")
        if snapshot.future_seed_commitment != self.future_seed_commitment:
            raise ProtocolOrderError("future seed commitment mismatch")
        if seed_commitment(self._future_seed) != self.future_seed_commitment:
            raise ProtocolOrderError("sealed future seed no longer matches its commitment")
        self.phase = ProtocolPhase.FUTURE_REVEALED
        return FutureSeedReveal(self._future_seed, self.future_seed_commitment)
