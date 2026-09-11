# AAA-v0 Assay Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, auditable AAA-v0 assay kernel that generates balanced treatment/control histories, enforces the terminal boundary, produces D1/D2/D3 future tasks, runs the three reference controls, records paired learning curves and corrective-recovery trajectories, and provides tooling for later exploratory calibration, preregistration freeze, and fresh confirmatory control validation—without executing or interpreting any frontier model.

**Architecture:** The implementation is a Python package split into immutable domain contracts, paired-history generation/audit, seed custody, future-task/rendering mechanics, reference agents, measurement, protocol orchestration, and preregistration/control-validation tooling. Scientific choices left open by the approved design are serialized configuration rather than hidden constants. This implementation phase builds and software-validates the machinery only; calibration, scientific preregistration, confirmatory control execution, and frontier execution remain later gated actions.

**Tech Stack:** Python 3.12+, standard library, NumPy 2.x, pytest 8.x. The kernel has no network dependency. JSON/JSONL are custody formats; SHA-256 is used for content addressing and provenance.

**Spec:** `docs/superpowers/specs/2026-09-11-aaa-v0-assay-design.md`

## Global Constraints

- Concept commit: `45061c6ea30edf4c16833c3deca0b13c2d98bb8f`.
- Approved assay-design commit: `04f0f4380d4fad01386c1f61e95a1e65c24f8de0`.
- The only manipulated acquisition-history regularity is the `Z`–`Q*` dependency: treatment has `I(Z;Q*) > 0`; control destroys that dependency while preserving the frozen marginals and sequence-balance contract.
- Treatment/control histories use the same ordered `z` sequence, same ordered `theta` sequence, same episode templates, and same presentation/resource schedule. `q*` assignments differ only as required by the treatment.
- Acquisition history is passive.
- Future targets and future task realizations become available only after the terminal boundary is frozen.
- Matching is bounded observable matching at `b=0`; it never authorizes hidden-state identity.
- Primary measurement is the full paired curve `Delta C(b)` at every frozen budget point. Area under the curve is secondary.
- D1 measures surface transfer; D2 measures transfer across a frozen surface-representation transformation; D3 measures corrective recovery after `phi -> phi_prime`. D3 success is not raw final competence.
- Required reference signature: `LOCAL = (-,-,-)`, `FIXED_META = (+,+,-)`, `REVISING_META = (+,+,+)`, with D3 signs referring only to the frozen recovery rule.
- Exploratory control calibration and confirmatory control validation use disjoint seeds and instances. Numerical thresholds are frozen between them in a preregistration artifact.
- `BALANCE_FAILURE`, `MATCH_FAILURE`, custody failure, and `CONTROL_MISMATCH` are hard stops.
- No frontier-model adapter, provider SDK, model execution, or empirical AAA interpretation is in scope.
- No implementation task modifies `README.md`, `formalization/AAA_CONCEPT_V0.md`, `formalization/CLAIM_HIERARCHY_V0.md`, or the approved assay-design spec.

---

## Planned file map

```text
pyproject.toml
src/aaa_v0/
    __init__.py
    contracts.py
    serialization.py
    histories.py
    balance.py
    provenance.py
    tasks.py
    environment.py
    agents.py
    measurement.py
    runner.py
    prereg.py
    cli.py

tests/
    test_contracts.py
    test_histories.py
    test_balance.py
    test_provenance.py
    test_tasks.py
    test_environment.py
    test_agents.py
    test_measurement.py
    test_runner.py
    test_prereg.py
    test_cli.py
```

The package deliberately contains no frontier/provider integration.

---

### Task 1: Project skeleton and immutable contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/aaa_v0/__init__.py`
- Create: `src/aaa_v0/contracts.py`
- Create: `src/aaa_v0/serialization.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: `AssayConfig`, `HistoryEpisode`, `HistoryPair`, `LatentTask`, `SurfaceView`, `SolvedHistoryView`, `ProbeObservation`, `TrialRecord`, `ProtocolStatus`, `canonical_json_bytes()`, `sha256_json()`.

- [ ] **Step 1: Add packaging metadata**

Create `pyproject.toml` with package name `aaa-v0`, Python `>=3.12`, dependency `numpy>=2.0,<3`, test dependency `pytest>=8,<9`, `src` package layout, and pytest `testpaths = ["tests"]`.

- [ ] **Step 2: Write failing contract tests**

Create `tests/test_contracts.py`:

```python
from dataclasses import FrozenInstanceError
from aaa_v0.contracts import AssayConfig, HistoryEpisode
from aaa_v0.serialization import canonical_json_bytes, sha256_json


def test_assay_config_is_frozen_and_canonical():
    cfg = AssayConfig.exploratory_default()
    first = canonical_json_bytes(cfg.to_dict())
    second = canonical_json_bytes(dict(reversed(list(cfg.to_dict().items()))))
    assert first == second
    assert sha256_json(cfg.to_dict()) == sha256_json(cfg.to_dict())
    try:
        cfg.z_count = 99
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("AssayConfig must be immutable")


def test_history_episode_serializes_declared_fields():
    ep = HistoryEpisode(
        episode_id="h-000",
        z=0,
        theta=2,
        q_star=1,
        surface_seed=7,
        resource_units=1,
    )
    assert ep.to_dict() == {
        "episode_id": "h-000",
        "z": 0,
        "theta": 2,
        "q_star": 1,
        "surface_seed": 7,
        "resource_units": 1,
    }
```

- [ ] **Step 3: Verify RED**

```bash
pytest tests/test_contracts.py -v
```

Expected: import failure because `aaa_v0.contracts` does not exist.

- [ ] **Step 4: Implement exact contract shapes**

In `contracts.py`, import `asdict`, `dataclass`, and `Literal`. Implement frozen dataclasses with `to_dict()` returning `asdict(self)` where the result is JSON-safe. Use these exact fields:

```python
@dataclass(frozen=True)
class AssayConfig:
    z_count: int
    q_count: int
    theta_count: int
    history_repeats: int
    future_tasks_per_family: int
    max_budget: int
    d3_change_after: int

    @classmethod
    def exploratory_default(cls) -> "AssayConfig":
        return cls(4, 4, 4, 8, 32, 4, 8)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


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
    match_ok: bool
    controls_ok: bool
    stop_code: str | None
```

`AssayConfig.__post_init__` rejects non-positive counts, `q_count != z_count`, `max_budget < 1`, and `d3_change_after < 1`.

- [ ] **Step 5: Implement deterministic serialization**

```python
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
```

- [ ] **Step 6: Verify GREEN**

```bash
pytest tests/test_contracts.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/aaa_v0/__init__.py src/aaa_v0/contracts.py src/aaa_v0/serialization.py tests/test_contracts.py
git commit -m "feat: add AAA-v0 immutable assay contracts"
```

---

### Task 2: Paired acquisition-history generator and balance audit

**Files:**
- Create: `src/aaa_v0/histories.py`
- Create: `src/aaa_v0/balance.py`
- Test: `tests/test_histories.py`
- Test: `tests/test_balance.py`

**Interfaces:**
- Produces: `make_history_pair(config: AssayConfig, seed: int) -> HistoryPair`, `empirical_mutual_information(pairs) -> float`, `BalanceReport`, `audit_history_pair(pair) -> BalanceReport`.

- [ ] **Step 1: Write failing paired-history tests**

```python
pair = make_history_pair(AssayConfig.exploratory_default(), seed=17)
assert sorted(pair.treatment_phi) == list(range(4))
assert [e.z for e in pair.treatment] == [e.z for e in pair.control]
assert [e.theta for e in pair.treatment] == [e.theta for e in pair.control]
assert [e.surface_seed for e in pair.treatment] == [e.surface_seed for e in pair.control]
assert [e.resource_units for e in pair.treatment] == [e.resource_units for e in pair.control]
assert sorted(e.q_star for e in pair.treatment) == sorted(e.q_star for e in pair.control)
assert all(e.q_star == pair.treatment_phi[e.z] for e in pair.treatment)
assert empirical_mutual_information((e.z, e.q_star) for e in pair.treatment) > 0.0
assert empirical_mutual_information((e.z, e.q_star) for e in pair.control) == 0.0
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_histories.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement exact finite balancing**

Generate `treatment_phi` as a seeded permutation of `range(q_count)`. This bijection plus equal `z` counts makes the treatment `Q*` marginal uniform. For each `z`, create exactly `history_repeats * q_count` episodes. Treatment uses `q_star = treatment_phi[z]`; control assigns every `q_star` exactly `history_repeats` times within each `z`, giving exact empirical independence. Generate `theta`, ordering, surface seeds, and `phi` from SHA-256 domain-separated RNG streams named `history-theta`, `history-order`, `history-surface`, and `history-phi`.

- [ ] **Step 4: Write failing balance-audit tests**

```python
report = audit_history_pair(pair)
assert report.ok
assert report.mismatched_fields == ()
assert report.treatment_mi > 0.0
assert report.control_mi == 0.0
```

Rebuild a control history with one changed `theta`, one changed `resource_units`, and one order swap in separate tests. Each must fail with a named mismatch.

- [ ] **Step 5: Implement balance audit**

Use:

```python
@dataclass(frozen=True)
class BalanceReport:
    ok: bool
    mismatched_fields: tuple[str, ...]
    treatment_mi: float
    control_mi: float
    treatment_hash: str
    control_hash: str
    non_treatment_sequence_hash_treatment: str
    non_treatment_sequence_hash_control: str
```

The non-treatment fingerprint covers ordered `(episode_id, z, theta, surface_seed, resource_units)` and excludes `q_star`. Exact fingerprint equality is required. Compute MI from finite counts using natural logarithms and coerce absolute values below `1e-15` to `0.0`.

- [ ] **Step 6: Add treatment-isolation regression test**

Redact `q_star` from both histories, serialize canonically, and assert byte equality. This test is the executable check that no other episode-level field carries condition.

- [ ] **Step 7: Verify GREEN**

```bash
pytest tests/test_histories.py tests/test_balance.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/aaa_v0/histories.py src/aaa_v0/balance.py tests/test_histories.py tests/test_balance.py
git commit -m "feat: add balanced AAA history intervention"
```

---

### Task 3: Terminal-boundary and seed custody

**Files:**
- Create: `src/aaa_v0/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces: `ProtocolPhase`, `ProtocolOrderError`, `SeedCustody`, `TerminalSnapshot`, `FutureSeedReveal`.

- [ ] **Step 1: Write failing custody tests**

```python
custody = SeedCustody(history_seed=11, sealed_future_seed=29)
assert custody.phase is ProtocolPhase.HISTORY
with pytest.raises(ProtocolOrderError):
    custody.reveal_future_seed()

snapshot = custody.freeze_terminal(
    treatment_state_hash="a" * 64,
    control_state_hash="b" * 64,
    history_pair_hash="c" * 64,
    config_hash="d" * 64,
)
assert custody.phase is ProtocolPhase.TERMINAL_FROZEN
reveal = custody.reveal_future_seed(snapshot)
assert reveal.seed == 29
assert custody.phase is ProtocolPhase.FUTURE_REVEALED
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_provenance.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement monotonic phase transition and commitments**

Use:

```python
class ProtocolPhase(Enum):
    HISTORY = "HISTORY"
    TERMINAL_FROZEN = "TERMINAL_FROZEN"
    FUTURE_REVEALED = "FUTURE_REVEALED"
```

The future-seed commitment is SHA-256 over `b"aaa-v0-future-seed:" + str(seed).encode("ascii")`. `TerminalSnapshot` stores treatment/control state hashes, history pair hash, config hash, future-seed commitment, and a snapshot hash over those fields. `reveal_future_seed()` verifies the caller's snapshot hash and commitment before returning `FutureSeedReveal(seed, commitment)`.

- [ ] **Step 4: Add tamper and cross-custody tests**

Changing any snapshot field or using another custody object's snapshot must raise `ProtocolOrderError`. The revealed seed must reproduce the commitment.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest tests/test_provenance.py -v
pytest -q
git add src/aaa_v0/provenance.py tests/test_provenance.py
git commit -m "feat: enforce AAA terminal seed boundary"
```

---

### Task 4: Future tasks, D1/D2/D3 rendering, and environment mechanics

**Files:**
- Create: `src/aaa_v0/tasks.py`
- Create: `src/aaa_v0/environment.py`
- Test: `tests/test_tasks.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: `make_future_tasks(config, reveal, phi) -> dict[str, tuple[LatentTask, ...]]`, `render_task(task, config) -> SurfaceView`, `MetaProbeEnvironment`.
- `SurfaceView` exposes observable structural signatures; it never exposes fields named `theta` or `q_star`.

- [ ] **Step 1: Write failing task-family tests**

```python
families = make_future_tasks(cfg, reveal, phi=(2, 0, 3, 1))
assert set(families) == {"D1", "D2", "D3"}
assert len(families["D1"]) == cfg.future_tasks_per_family
assert all(t.family == "D1" for t in families["D1"])
```

For paired D1/D2 indices, assert identical `(z, theta, q_star)` and different surface seeds. For D3, assert old `phi` before the change point and a deranged `phi_prime` afterward.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_tasks.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement future generation with domain-separated RNG**

Use SHA-256 stream labels `future-latent`, `future-d1-surface`, `future-d2-surface`, and `future-d3-surface`. Future `theta` values are uniform over `range(theta_count)` and independent of history generation. Construct `phi_prime` by a seeded cyclic offset in `1..q_count-1`, guaranteeing `phi_prime[z] != phi[z]` for every `z`.

- [ ] **Step 4: Implement observable structural signatures**

Define pure helpers:

```python
def context_signature(z: int, width: int) -> tuple[int, ...]:
    return tuple(1 if i == z else 0 for i in range(width))


def probe_signature(q: int, width: int) -> tuple[int, ...]:
    return tuple(1 if i == q else 0 for i in range(width))
```

These signatures are observable structure, not hidden state. D1 uses canonical surface tokens and canonical presentation order. D2 changes all context/probe token names and permutes the presentation order, while the structural signature remains attached to the corresponding visible item. Thus a literal token lookup fails, but a structural-role mapping can transfer.

- [ ] **Step 5: Write failing environment tests**

```python
task = LatentTask("x", "D1", z=0, theta=3, q_star=2, surface_seed=1, phase_index=0)
env = MetaProbeEnvironment(task, q_count=4, max_budget=4)
q2 = probe_signature(2, 4)
q0 = probe_signature(0, 4)
assert env.probe(q2) == ProbeObservation(q2, True, 3)
assert env.probe(q0) == ProbeObservation(q0, False, None)
```

Assert the fifth probe raises `BudgetExceeded`.

- [ ] **Step 6: Implement environment mechanics**

Map observable probe signatures back to the unique structural role index. Only `q_star` returns `(informative=True, value=theta)`; all other probes return `(False, None)`. Each probe call consumes one budget unit, including repeated probes.

- [ ] **Step 7: Add renderer-leakage tests**

Assert serialized `SurfaceView` has no key named `theta`, `q_star`, or `phi`. Observable context/probe structural signatures are allowed and expected. Assert D2 tokens and presentation order differ from D1 while the set of structural signatures is exactly preserved.

- [ ] **Step 8: Verify GREEN and commit**

```bash
pytest tests/test_tasks.py tests/test_environment.py -v
pytest -q
git add src/aaa_v0/tasks.py src/aaa_v0/environment.py tests/test_tasks.py tests/test_environment.py
git commit -m "feat: add AAA future task families"
```

---

### Task 5: Passive-history rendering and the three reference controls

**Files:**
- Create: `src/aaa_v0/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `Agent` protocol, `LocalAgent`, `FixedMetaAgent`, `RevisingMetaAgent`.
- Agent methods are exactly `observe_history(records)`, `start_task(view)`, `choose_probe()`, `observe_probe(observation)`, and `guess_theta()`.
- `records` are `tuple[SolvedHistoryView, ...]`; agents never receive latent `z`, `q_star`, or `phi`.

- [ ] **Step 1: Write failing no-omniscience tests**

Use `inspect.signature` to verify no public agent method parameter is named `z`, `q_star`, `theta`, or `phi`. Verify passive-history records expose only `SurfaceView`, the informative probe's observable structural signature, and the resolved historical target.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_agents.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement `LocalAgent`**

Ignore history. For each future task, probe visible probe signatures in presentation order. Before observing an informative value, `guess_theta()` returns `0`; afterward it returns the observed value. No cross-task mapping persists.

- [ ] **Step 4: Implement `FixedMetaAgent`**

During history, learn `context_signature -> informative_probe_signature`. In future tasks, probe the learned signature first. If that probe is uninformative, search remaining visible signatures in presentation order so local task competence can still recover, but never change the stored cross-task mapping. This ensures D3 can separate recovery dynamics from final-score recovery.

- [ ] **Step 5: Implement `RevisingMetaAgent`**

Start with the same learned map. If the preferred first probe is uninformative, search remaining probes. When an informative probe is found, replace the stored mapping for that context signature with the newly informative probe signature. Subsequent tasks in that context use the revised mapping first.

- [ ] **Step 6: Add qualitative behavior tests**

Use hand-authored history and future views to prove:

```python
assert local.mapping_size == 0
assert fixed.preferred_probe(ctx) == old_probe
assert revising.preferred_probe(ctx) == new_probe
```

after both meta agents encounter the same D3 mismatch and discover the new informative role. `FixedMetaAgent` must still report `old_probe`; `RevisingMetaAgent` must report `new_probe`.

- [ ] **Step 7: Verify GREEN and commit**

```bash
pytest tests/test_agents.py -v
pytest -q
git add src/aaa_v0/agents.py tests/test_agents.py
git commit -m "feat: add AAA reference controls"
```

---

### Task 6: Paired learning curves and D3 recovery observables

**Files:**
- Create: `src/aaa_v0/measurement.py`
- Test: `tests/test_measurement.py`

**Interfaces:**
- Produces: `LearningCurve`, `PairedCurve`, `RecoveryTrajectory`, `compute_learning_curve()`, `compute_paired_curve()`, `compute_recovery_trajectory()`.
- This module computes observables only; it does not authorize C1–C4.

- [ ] **Step 1: Write failing curve tests**

Create hand-authored `TrialRecord`s for two tasks at budgets `0,1,2`. Include two examples with equal final competence but different intermediate geometry. Assert exact treatment, control, and delta points.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_measurement.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement immutable measurement records**

```python
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
```

Compute competence as mean `correct` at each budget. Compute secondary AUC by trapezoidal integration across the integer budget grid.

- [ ] **Step 4: Add D3 semantic regression test**

Construct brittle and revising records with the same eventual competence but different first-probe old-rule adherence. Assert their `RecoveryTrajectory` values differ. This prevents D3 from collapsing into a final-score test.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest tests/test_measurement.py -v
pytest -q
git add src/aaa_v0/measurement.py tests/test_measurement.py
git commit -m "feat: measure paired AAA learning dynamics"
```

---

### Task 7: Protocol runner, zero-budget matching, and hard stops

**Files:**
- Create: `src/aaa_v0/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `ProtocolStop`, `MatchReport`, `RunBundle`, `run_reference_pair()`, `assert_match_gate()`.

- [ ] **Step 1: Write failing match-gate tests**

```python
with pytest.raises(ProtocolStop) as exc:
    assert_match_gate(treatment=0.75, control=0.50, tolerance=0.10)
assert exc.value.code == "MATCH_FAILURE"
```

Also assert `0.50` versus `0.50` passes at tolerance `0.0`.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_runner.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement paired future stepping**

For each shared latent task: render equivalent copies, call both agents' guesses at `b=0`, record them, then for budgets `1..B` allow exactly one probe per condition per budget step and record the new guess. One condition never receives extra probes or retries.

For each `TrialRecord.old_rule_adherence`, compare the first chosen probe signature on that task with the pre-D3 mapping recorded at the terminal boundary. Use `None` for D1/D2.

- [ ] **Step 4: Implement matching as a pre-interpretation gate**

Compute declared `b=0` competence separately for each condition and compare with an explicit tolerance passed into the runner. On failure raise `ProtocolStop("MATCH_FAILURE")` before transfer/recovery rule evaluation.

- [ ] **Step 5: Implement custody-rich `RunBundle`**

The serialized bundle includes exactly named top-level fields:

```text
config_hash
history_pair_hash
balance_report
terminal_snapshot_hash
future_seed_commitment
future_seed_reveal_hash
raw_trial_records_hash
d1_paired_curve
d2_paired_curve
d3_recovery_trajectory
protocol_status
```

No field is named `aaa_score`, `intelligence`, or `self_improvement`.

- [ ] **Step 6: Add deterministic end-to-end software test**

Run identical reference-agent/config/history/future seeds twice and assert byte-identical `RunBundle` canonical serialization. Change only the future seed; history/config/terminal commitment fields stay stable while future record hashes change.

- [ ] **Step 7: Verify GREEN and commit**

```bash
pytest tests/test_runner.py -v
pytest -q
git add src/aaa_v0/runner.py tests/test_runner.py
git commit -m "feat: add AAA protocol runner and stop gates"
```

---

### Task 8: Calibration report, preregistration contract, and confirmatory control gate

**Files:**
- Create: `src/aaa_v0/prereg.py`
- Test: `tests/test_prereg.py`

**Interfaces:**
- Produces: `TransferRule`, `RecoveryRule`, `ExploratoryCalibrationReport`, `Preregistration`, `ConfirmatoryControlReport`, `freeze_preregistration()`, `validate_confirmatory_controls()`.
- The implementation provides the mechanism; this phase does not perform the scientific preregistration freeze or confirmatory run.

- [ ] **Step 1: Write failing preregistration identity tests**

Assert canonical JSON round-trip stability, SHA-256 identity stability, and rejection when any confirmatory seed is also present in the exploratory seed set.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_prereg.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement explicit rule dataclasses**

```python
@dataclass(frozen=True)
class TransferRule:
    min_positive_budget_points: int
    min_mean_delta: float
    max_zero_budget_abs_delta: float


@dataclass(frozen=True)
class RecoveryRule:
    max_post_change_old_rule_adherence: float
    min_post_change_competence: float
    evaluation_start_offset: int


@dataclass(frozen=True)
class Preregistration:
    config: AssayConfig
    d1_rule: TransferRule
    d2_rule: TransferRule
    d3_rule: RecoveryRule
    exploratory_seed_hashes: tuple[str, ...]
    confirmatory_history_seeds: tuple[int, ...]
    confirmatory_future_seeds: tuple[int, ...]
    spec_commit: str
```

`spec_commit` must equal `04f0f4380d4fad01386c1f61e95a1e65c24f8de0`.

- [ ] **Step 4: Implement exploratory-report semantics**

`ExploratoryCalibrationReport.to_dict()` includes:

```json
{"status":"EXPLORATORY_UNSCORED","next_phase_gate_passed":false,"aaa_evidence":"NONE"}
```

It records raw control curves, recovery trajectories, and proposed rule values supplied by the caller. It emits no C1–C4 claim.

- [ ] **Step 5: Implement confirmatory signature gate**

Apply the frozen rules to fresh `LOCAL`, `FIXED_META`, and `REVISING_META` bundles and require:

```text
               D1   D2   D3
LOCAL           -    -    -
FIXED_META      +    +    -
REVISING_META   +    +    +
```

D3 uses only `RecoveryRule`. On any cell mismatch return:

```json
{"status":"CONTROL_MISMATCH","next_phase_gate_passed":false,"aaa_evidence":"NONE"}
```

If every balance, custody, match, and signature check succeeds, return:

```json
{"status":"CONTROL_VALIDATED","next_phase_gate_passed":true,"aaa_evidence":"NONE"}
```

`next_phase_gate_passed` means only that a separately approved frontier-execution protocol may be designed or invoked later. It does not authorize empirical interpretation by itself.

- [ ] **Step 6: Add anti-tuning tests**

Reject confirmatory bundles when exploratory/confirmatory seed identities overlap, `spec_commit` differs, config hashes differ, or any bundle has a failed balance/match/custody status.

- [ ] **Step 7: Verify GREEN and commit**

```bash
pytest tests/test_prereg.py -v
pytest -q
git add src/aaa_v0/prereg.py tests/test_prereg.py
git commit -m "feat: add AAA preregistration and control validation"
```

---

### Task 9: Offline CLI, manifests, and implementation-only verification

**Files:**
- Create: `src/aaa_v0/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Allowed subcommands: `audit-history`, `calibrate-controls`, `freeze-prereg`, `validate-controls`.
- A frontier/model execution subcommand is forbidden in this implementation.

- [ ] **Step 1: Write failing CLI-surface tests**

Run `python -m aaa_v0.cli --help` under subprocess and assert the four allowed names appear. Invoke `run-frontier` and assert argparse exits nonzero.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_cli.py -v
```

Expected: subprocess/import failure.

- [ ] **Step 3: Implement `audit-history`**

Arguments: config JSON path and history seed. Write canonical JSON containing config hash, history pair hash, treatment/control MI, named balance fields, and `status` equal to `BALANCED` or `BALANCE_FAILURE`. Exit nonzero on failure.

- [ ] **Step 4: Implement `calibrate-controls`**

Arguments: config JSON, explicit exploratory history seed list, explicit exploratory future seed list, output directory. Run only the three reference agents. Write raw JSONL records plus `exploratory_calibration.json` with status `EXPLORATORY_UNSCORED` and `aaa_evidence = NONE`.

- [ ] **Step 5: Implement `freeze-prereg` as a mechanical writer**

Arguments: exploratory report, explicit decision-rules JSON, explicit disjoint confirmatory seed lists. Validate disjointness; write `preregistration.json` and `preregistration.sha256`. The command never invents threshold values; every threshold comes from the supplied decision-rules JSON.

- [ ] **Step 6: Implement `validate-controls`**

Arguments: frozen preregistration and output directory. Run fresh confirmatory reference controls only. Write `confirmatory_controls.json`; exit nonzero on `BALANCE_FAILURE`, `MATCH_FAILURE`, custody failure, or `CONTROL_MISMATCH`.

- [ ] **Step 7: Add manifest-integrity tests**

Every output directory gets `MANIFEST.json` mapping relative file paths to SHA-256. Identical inputs produce identical hashes. Confirmatory output includes the preregistration hash.

- [ ] **Step 8: Run complete automated verification**

```bash
pytest -q
python -m aaa_v0.cli --help
```

Expected: all tests PASS; help exposes only the four allowed subcommands.

- [ ] **Step 9: Run scope-leak checks**

```bash
grep -R "OpenAI\|Anthropic\|Gemini" -n src/aaa_v0 tests || true
grep -R "aaa_score\|self_improvement" -n src/aaa_v0 tests || true
```

Expected: no provider integration and no scalar `aaa_score` or `self_improvement` output field. A match inside a negative test assertion is permitted only when the test is explicitly proving that the name is absent from produced output.

- [ ] **Step 10: Commit**

```bash
git add src/aaa_v0/cli.py tests/test_cli.py
git commit -m "feat: add AAA offline assay workflow"
```

---

## Final implementation gate

After Task 9, do not calibrate scientifically and do not run a frontier model. Run fresh verification and create `IMPLEMENTATION_STATE.json` from actual repository/runtime values. The writer must obtain the implementation SHA with `git rev-parse HEAD` and record these fixed provenance fields:

```json
{
  "concept_commit": "45061c6ea30edf4c16833c3deca0b13c2d98bb8f",
  "assay_design_commit": "04f0f4380d4fad01386c1f61e95a1e65c24f8de0",
  "implementation_head": "value returned by git rev-parse HEAD",
  "history_balance": "VERIFIED_ON_TEST_FIXTURES_ONLY",
  "seed_boundary": "VERIFIED_ON_TEST_FIXTURES_ONLY",
  "reference_controls": "SOFTWARE_VALIDATED_ONLY",
  "preregistration": "NOT_FROZEN",
  "confirmatory_controls": "NOT_RUN",
  "frontier_execution": "NOT_RUN",
  "aaa_evidence": "NONE"
}
```

The `implementation_head` string is populated programmatically at verification time, not manually copied from this plan. Add exact pytest pass/fail counts under a `unit_tests` object generated from that verification run.

`SOFTWARE_VALIDATED_ONLY` means deterministic tests exercise the intended code paths. It is not exploratory calibration and not the confirmatory scientific control gate.

If any automated test, balance audit, seed-boundary check, deterministic custody check, or scope-leak check fails, stop and repair the implementation before any later calibration action.

## Explicitly outside this plan

The following require later explicit approval and remain absent from this implementation phase:

- a frontier-model/provider adapter;
- a frontier-model prompt protocol;
- exploratory tuning using frontier-model behavior;
- the scientific act of freezing the official AAA-v0 preregistration thresholds;
- confirmatory scientific control execution;
- frontier-model execution;
- C1/C2/C3/C4 interpretation for a frontier model;
- claims about AGI, recursive self-improvement, ARC-AGI-4, or general corrigibility.
