# AAA-v0 Assay Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, auditable AAA-v0 assay kernel that generates balanced treatment/control histories, enforces the terminal boundary, produces D1/D2/D3 future tasks, runs the three reference controls, records paired learning curves and corrective-recovery trajectories, and provides tooling for later exploratory calibration, preregistration freeze, and fresh confirmatory control validation—without executing or interpreting any frontier model.

**Architecture:** The implementation is a Python package split into immutable domain contracts, paired-history construction/audit, seed custody, future-task/rendering mechanics, reference agents, measurement, protocol orchestration, and preregistration/control-validation tooling. Scientific choices left open by the approved design are explicit serialized contracts rather than hidden constants. This implementation phase builds and software-validates machinery only; exploratory calibration, the scientific preregistration freeze, confirmatory scientific controls, and frontier execution remain later gated actions.

**Tech Stack:** Python 3.12+, standard library, NumPy 2.x, pytest 8.x. No network dependency. JSON/JSONL are custody formats; SHA-256 is used for content addressing and provenance.

**Spec:** `docs/superpowers/specs/2026-09-11-aaa-v0-assay-design.md`

## Global Constraints

- Concept commit: `45061c6ea30edf4c16833c3deca0b13c2d98bb8f`.
- Approved assay-design commit: `04f0f4380d4fad01386c1f61e95a1e65c24f8de0`.
- The only intended acquisition-history treatment is the `Z`–`Q*` dependency: treatment has `I(Z;Q*) > 0`; control has exact empirical `I(Z;Q*) = 0` while preserving declared marginals and satisfying a preregistered sequence-balance rule.
- Treatment/control histories share the same ordered `z`, `theta`, episode-template, presentation-budget, and resource schedule. `q*` assignments differ only to instantiate/destroy the target relation and must pass sequence-cue audits.
- Acquisition history is passive.
- Future targets and future task realizations become available only after the terminal boundary is frozen.
- Matching is bounded observable matching at `b=0`; it never authorizes hidden-state identity.
- Primary measurement is the full paired curve `Delta C(b)` at every frozen budget point. AUC is secondary.
- D1 measures surface transfer; D2 measures transfer across a frozen surface-representation transformation; D3 measures corrective recovery after `phi -> phi_prime`. D3 success is not raw final competence.
- Required reference signature: `LOCAL = (-,-,-)`, `FIXED_META = (+,+,-)`, `REVISING_META = (+,+,+)`, with D3 signs referring only to the frozen recovery rule.
- Exploratory calibration and confirmatory validation use disjoint seeds/instances. All numerical rules—including sequence-balance tolerances—are frozen between them in a preregistration artifact.
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
- Produces: `AssayConfig`, `SequenceBalanceRule`, `HistoryEpisode`, `HistoryPair`, `LatentTask`, `SurfaceView`, `SolvedHistoryView`, `ProbeObservation`, `TrialRecord`, `ProtocolStatus`, `canonical_json_bytes()`, `sha256_json()`.

- [ ] **Step 1: Add packaging metadata**

Create `pyproject.toml` with package name `aaa-v0`, Python `>=3.12`, dependency `numpy>=2.0,<3`, test dependency `pytest>=8,<9`, `src` package layout, and pytest `testpaths = ["tests"]`.

- [ ] **Step 2: Write failing contract tests**

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
    ep = HistoryEpisode("h-000", 0, 2, 1, 7, 1)
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

Implement frozen dataclasses using `dataclasses.asdict` for JSON-safe `to_dict()` methods where needed:

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
class SequenceBalanceRule:
    max_q_transition_l1: float
    max_q_run_length_l1: float
    max_lagged_mi_delta: float
    lag_window: int
    max_search_attempts: int


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
```

`AssayConfig.__post_init__` rejects non-positive counts, `q_count != z_count`, `max_budget < 1`, and `d3_change_after < 1`. `SequenceBalanceRule.__post_init__` rejects negative tolerances, `lag_window < 1`, and `max_search_attempts < 1`.

- [ ] **Step 5: Implement deterministic serialization**

```python
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
```

- [ ] **Step 6: Verify GREEN and commit**

```bash
pytest tests/test_contracts.py -v
pytest -q
git add pyproject.toml src/aaa_v0/__init__.py src/aaa_v0/contracts.py src/aaa_v0/serialization.py tests/test_contracts.py
git commit -m "feat: add AAA-v0 immutable assay contracts"
```

---

### Task 2: Paired acquisition-history generator and sequence-aware balance audit

**Files:**
- Create: `src/aaa_v0/histories.py`
- Create: `src/aaa_v0/balance.py`
- Test: `tests/test_histories.py`
- Test: `tests/test_balance.py`

**Interfaces:**
- Produces: `make_history_pair(config, seed, sequence_rule) -> HistoryPair`, `BalanceConstructionError`, `empirical_mutual_information()`, `SequenceDiagnostics`, `BalanceReport`, `audit_history_pair(pair, sequence_rule) -> BalanceReport`.

- [ ] **Step 1: Write failing exact-marginal tests**

```python
cfg = AssayConfig.exploratory_default()
rule = SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)
pair = make_history_pair(cfg, seed=17, sequence_rule=rule)
assert sorted(pair.treatment_phi) == list(range(cfg.q_count))
assert [e.z for e in pair.treatment] == [e.z for e in pair.control]
assert [e.theta for e in pair.treatment] == [e.theta for e in pair.control]
assert [e.surface_seed for e in pair.treatment] == [e.surface_seed for e in pair.control]
assert [e.resource_units for e in pair.treatment] == [e.resource_units for e in pair.control]
assert sorted(e.q_star for e in pair.treatment) == sorted(e.q_star for e in pair.control)
assert all(e.q_star == pair.treatment_phi[e.z] for e in pair.treatment)
assert empirical_mutual_information((e.z, e.q_star) for e in pair.treatment) > 0.0
assert empirical_mutual_information((e.z, e.q_star) for e in pair.control) == 0.0
```

The permissive rule above is a software-test fixture only; it is not a scientific threshold.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_histories.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement exact marginal construction with constrained control shuffling**

Generate `treatment_phi` as a seeded permutation of `range(q_count)`. Make each `z` appear exactly `history_repeats * q_count` times. Treatment uses `q_star = phi[z]`. Control assigns every `q_star` exactly `history_repeats` times within each `z`, which makes finite-sample `I(Z;Q*) = 0` exactly.

Build one ordered episode shell containing `episode_id`, `z`, `theta`, `surface_seed`, and `resource_units`. Both conditions reuse that shell byte-for-byte; only `q_star` differs.

For the control assignment, deterministically shuffle the fixed within-`z` `q_star` multisets using a domain-separated RNG and rejection-sample until `sequence_diagnostics(treatment_q, control_q, z_sequence, rule)` passes. If no assignment passes within `rule.max_search_attempts`, raise `BalanceConstructionError` rather than relaxing the rule.

Use SHA-256 domain-separated RNG streams `history-theta`, `history-order`, `history-surface`, `history-phi`, and `history-control-q`.

- [ ] **Step 4: Define sequence diagnostics**

`SequenceDiagnostics` contains:

```python
@dataclass(frozen=True)
class SequenceDiagnostics:
    q_transition_l1: float
    q_run_length_l1: float
    lagged_mi_delta: tuple[float, ...]
```

Compute normalized `Q* -> Q*` transition matrices, normalized run-length histograms, and `I(Z_t; Q*_{t+k})` for every integer lag `k` in `[-lag_window, -1] U [1, lag_window]`. `q_transition_l1` and `q_run_length_l1` are treatment/control L1 distances; `lagged_mi_delta` is treatment minus control at each lag. The intended lag-zero difference is excluded because that is the treatment itself.

- [ ] **Step 5: Implement balance audit**

Use:

```python
@dataclass(frozen=True)
class BalanceReport:
    ok: bool
    mismatched_fields: tuple[str, ...]
    treatment_mi: float
    control_mi: float
    sequence: SequenceDiagnostics
    treatment_hash: str
    control_hash: str
    non_treatment_sequence_hash_treatment: str
    non_treatment_sequence_hash_control: str
```

`ok` requires exact non-treatment sequence equality, exact `Q*` marginal equality, treatment MI positive, control MI normalized to exactly `0.0`, and every sequence diagnostic within the supplied rule. Values with absolute magnitude below `1e-15` are normalized to zero.

- [ ] **Step 6: Write hard-failure regression tests**

Redact `q_star` and assert treatment/control canonical serialization is byte-identical. Separately mutate `theta`, `resource_units`, or order and assert `ok is False`. Supply a deliberately impossible all-zero sequence rule and low search budget, and assert construction raises `BalanceConstructionError` rather than silently proceeding.

- [ ] **Step 7: Verify GREEN and commit**

```bash
pytest tests/test_histories.py tests/test_balance.py -v
pytest -q
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

- [ ] **Step 3: Implement monotonic phase transition and commitment checking**

```python
class ProtocolPhase(Enum):
    HISTORY = "HISTORY"
    TERMINAL_FROZEN = "TERMINAL_FROZEN"
    FUTURE_REVEALED = "FUTURE_REVEALED"
```

Commit the future seed with SHA-256 over `b"aaa-v0-future-seed:" + str(seed).encode("ascii")`. `TerminalSnapshot` stores treatment/control state hashes, history pair hash, config hash, future-seed commitment, and a snapshot hash. `reveal_future_seed()` verifies snapshot identity and seed commitment before returning `FutureSeedReveal`.

- [ ] **Step 4: Add tamper/cross-custody tests**

Changing any snapshot field or presenting another custody object's snapshot raises `ProtocolOrderError`. The revealed seed must reproduce the pre-boundary commitment.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest tests/test_provenance.py -v
pytest -q
git add src/aaa_v0/provenance.py tests/test_provenance.py
git commit -m "feat: enforce AAA terminal seed boundary"
```

---

### Task 4: History/future rendering, D1/D2/D3 tasks, and environment mechanics

**Files:**
- Create: `src/aaa_v0/tasks.py`
- Create: `src/aaa_v0/environment.py`
- Test: `tests/test_tasks.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: `context_signature()`, `probe_signature()`, `render_history_episode()`, `render_task()`, `make_future_tasks()`, `MetaProbeEnvironment`.
- `SurfaceView` may expose observable structural signatures but never exposes fields named `theta`, `q_star`, or `phi`.

- [ ] **Step 1: Write failing task-family tests**

```python
families = make_future_tasks(cfg, reveal, phi=(2, 0, 3, 1))
assert set(families) == {"D1", "D2", "D3"}
assert len(families["D1"]) == cfg.future_tasks_per_family
assert all(t.family == "D1" for t in families["D1"])
```

For paired D1/D2 indices, assert identical `(z, theta, q_star)` and different surface seeds. For D3, assert old `phi` before `d3_change_after` and deranged `phi_prime` afterward.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_tasks.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement domain-separated future generation**

Use SHA-256 stream labels `future-latent`, `future-d1-surface`, `future-d2-surface`, and `future-d3-surface`. Future `theta` values are uniform over `range(theta_count)` and independent of history generation. Construct `phi_prime` by a seeded cyclic offset in `1..q_count-1`, so every context changes informative probe role.

- [ ] **Step 4: Implement structural signatures and renderers**

```python
def context_signature(z: int, width: int) -> tuple[int, ...]:
    return tuple(1 if i == z else 0 for i in range(width))


def probe_signature(q: int, width: int) -> tuple[int, ...]:
    return tuple(1 if i == q else 0 for i in range(width))
```

`render_history_episode()` returns a `SolvedHistoryView`: a D1-style `SurfaceView`, the observable informative probe signature, and resolved historical `theta`. This is the complete passive-history record given to agents.

D1 uses canonical context/probe token names and canonical presentation order. D2 renames every context/probe token and permutes probe presentation order using its surface seed, while the structural signature stays attached to the corresponding visible item. Thus literal token reuse fails but structural-role reuse remains possible.

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

Map observable probe signatures to unique structural roles. Only `q_star` returns `(informative=True, value=theta)`; other probes return `(False, None)`. Every probe consumes one budget unit, including repeats.

- [ ] **Step 7: Add leakage/shift tests**

Assert serialized `SurfaceView` has no key named `theta`, `q_star`, or `phi`. Observable signatures are allowed. Assert D2 tokens and presentation order differ from D1 while the set of structural signatures is preserved exactly. Assert `render_history_episode()` reveals the informative observable role but not latent field names.

- [ ] **Step 8: Verify GREEN and commit**

```bash
pytest tests/test_tasks.py tests/test_environment.py -v
pytest -q
git add src/aaa_v0/tasks.py src/aaa_v0/environment.py tests/test_tasks.py tests/test_environment.py
git commit -m "feat: add AAA task and rendering kernel"
```

---

### Task 5: Reference-agent protocol and three controls

**Files:**
- Create: `src/aaa_v0/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `Agent` protocol, `LocalAgent`, `FixedMetaAgent`, `RevisingMetaAgent`.
- Exact public methods: `observe_history(records)`, `start_task(view)`, `choose_probe()`, `observe_probe(observation)`, `guess_theta()`, `state_dict()`, `preferred_probe(context_signature)`.
- `state_dict()` is JSON-safe and is the sole source for terminal agent-state hashing.

- [ ] **Step 1: Write failing no-omniscience/interface tests**

Use `inspect.signature` to verify no public method parameter is named `z`, `q_star`, `theta`, or `phi`. Verify `state_dict()` canonical serialization is deterministic.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_agents.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement `LocalAgent`**

Ignore history. Probe visible signatures in presentation order. Before informative evidence, `guess_theta()` returns `0`; afterward it returns the observed value. `state_dict()` contains only name and current local state; no cross-task mapping persists.

- [ ] **Step 4: Implement `FixedMetaAgent`**

From passive history learn `context_signature -> informative_probe_signature`. Probe the learned signature first. After a mismatch, search remaining visible signatures so object-level competence can recover, but never modify the learned mapping. `preferred_probe()` always returns the frozen learned role.

- [ ] **Step 5: Implement `RevisingMetaAgent`**

Start with the same learned map. After a preferred probe fails, search remaining roles. Once an informative role is found, replace that context's mapping. Subsequent tasks use the revised role first.

- [ ] **Step 6: Add behavior/state tests**

After the same D3 mismatch and local discovery:

```python
assert local.preferred_probe(ctx) is None
assert fixed.preferred_probe(ctx) == old_probe
assert revising.preferred_probe(ctx) == new_probe
assert sha256_json(fixed.state_dict()) != sha256_json(revising.state_dict())
```

Also assert `LocalAgent.state_dict()` is unchanged by two different passive histories after local state is reset.

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
- This module computes observables only; it authorizes no C1–C4 claim.

- [ ] **Step 1: Write failing curve tests**

Create hand-authored `TrialRecord`s for two tasks at budgets `0,1,2`, including two cases with equal final competence but different intermediate geometry. Assert exact treatment, control, and delta points.

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

Competence is mean `correct` at each budget. Secondary AUC uses trapezoidal integration on the integer budget grid.

- [ ] **Step 4: Add D3 semantic regression test**

Construct brittle and revising treatment-arm records with equal eventual competence but different first-probe old-rule adherence. Assert their recovery trajectories differ. D3 recovery is computed from the history-exposed/treatment agent relative to its pre-change learned mapping; the H0 arm remains recorded as a descriptive control but does not define the recovery criterion.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest tests/test_measurement.py -v
pytest -q
git add src/aaa_v0/measurement.py tests/test_measurement.py
git commit -m "feat: measure paired AAA learning dynamics"
```

---

### Task 7: Protocol runner, per-family matching, and hard stops

**Files:**
- Create: `src/aaa_v0/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `ProtocolStop`, `MatchReport`, `RunBundle`, `run_reference_pair()`, `assert_match_gate()`.

- [ ] **Step 1: Write failing match-gate tests**

`MatchReport` records zero-budget treatment/control competence and absolute delta separately for D1, D2, and D3. All families must pass the same supplied tolerance.

```python
with pytest.raises(ProtocolStop) as exc:
    assert_match_gate({"D1": 0.25, "D2": 0.0, "D3": 0.0}, tolerance=0.10)
assert exc.value.code == "MATCH_FAILURE"
```

A report with every family delta `0.0` passes at tolerance `0.0`.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_runner.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement history exposure and terminal freeze**

Render each `HistoryEpisode` into `SolvedHistoryView`; expose H+ only to the treatment agent and H0 only to the control agent. Compute treatment/control terminal state hashes exclusively from `sha256_json(agent.state_dict())`. Freeze those hashes, history pair hash, config hash, and future-seed commitment before future reveal.

- [ ] **Step 4: Implement paired future stepping**

For each shared latent task, render equivalent copies, call both agents at `b=0`, then allow exactly one probe per condition per budget step `1..B`. Record the first chosen probe signature on every task. `old_rule_adherence` compares the treatment agent's first D3 probe with the pre-change terminal preferred probe for that context; D1/D2 use `None`.

- [ ] **Step 5: Implement matching before rule evaluation**

Compute `b=0` competence separately by D1/D2/D3. If any family exceeds tolerance, raise `ProtocolStop("MATCH_FAILURE")` before transfer/recovery evaluation.

- [ ] **Step 6: Implement custody-rich `RunBundle`**

Top-level serialized fields are:

```text
config_hash
history_pair_hash
balance_report
terminal_snapshot_hash
future_seed_commitment
future_seed_reveal_hash
raw_trial_records_hash
match_report
d1_paired_curve
d2_paired_curve
d3_treatment_recovery
d3_control_curve
protocol_status
```

No field is named `aaa_score`, `intelligence`, or `self_improvement`.

- [ ] **Step 7: Add deterministic end-to-end software tests**

Identical config/history/future seeds and agent classes produce byte-identical `RunBundle` serialization. Changing only future seed leaves history/config state hashes unchanged and changes future record hashes. An intentionally failed balance report prevents terminal exposure; an intentionally failed match report prevents rule evaluation.

- [ ] **Step 8: Verify GREEN and commit**

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
- This task implements machinery only; it does not perform the scientific preregistration freeze or confirmatory run.

- [ ] **Step 1: Write failing preregistration identity tests**

Assert canonical JSON round-trip stability, SHA-256 identity stability, and rejection when any confirmatory seed identity appears in the exploratory set.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_prereg.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement explicit frozen-rule shapes**

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
    sequence_balance_rule: SequenceBalanceRule
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

```json
{"status":"EXPLORATORY_UNSCORED","next_phase_gate_passed":false,"aaa_evidence":"NONE"}
```

The report records raw curves/recovery, balance diagnostics, and caller-supplied candidate rules. It emits no C1–C4 claim.

- [ ] **Step 5: Implement confirmatory signature gate**

Apply the frozen rules to fresh reference-control bundles and require exactly:

```text
               D1   D2   D3
LOCAL           -    -    -
FIXED_META      +    +    -
REVISING_META   +    +    +
```

D1/D2 signs use `TransferRule`; D3 signs use the treatment-arm `RecoveryTrajectory` and `RecoveryRule` only. On any failed balance/custody/match/signature cell:

```json
{"status":"CONTROL_MISMATCH","next_phase_gate_passed":false,"aaa_evidence":"NONE"}
```

On full success:

```json
{"status":"CONTROL_VALIDATED","next_phase_gate_passed":true,"aaa_evidence":"NONE"}
```

The gate means only that a separately approved frontier protocol may proceed later.

- [ ] **Step 6: Add anti-tuning/provenance tests**

Reject confirmatory bundles when exploratory/confirmatory seed identities overlap, `spec_commit` differs, config hashes differ, sequence-balance rules differ, or any lower gate failed.

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
- Frontier/model execution commands are forbidden.

- [ ] **Step 1: Write failing CLI-surface tests**

Run `python -m aaa_v0.cli --help` under subprocess and assert the four allowed names appear. Invoke `run-frontier` and assert argparse exits nonzero.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_cli.py -v
```

Expected: subprocess/import failure.

- [ ] **Step 3: Implement `audit-history`**

Arguments: config JSON, sequence-balance-rule JSON, history seed. Write canonical JSON with config/history hashes, treatment/control MI, sequence diagnostics, named balance fields, and status `BALANCED` or `BALANCE_FAILURE`. Exit nonzero on failure.

- [ ] **Step 4: Implement `calibrate-controls`**

Arguments: config JSON, candidate sequence-balance-rule JSON, explicit exploratory history/future seed lists, output directory. Run only reference controls. Write raw JSONL plus `exploratory_calibration.json` with `EXPLORATORY_UNSCORED` and `aaa_evidence = NONE`.

- [ ] **Step 5: Implement `freeze-prereg` as a mechanical writer**

Arguments: exploratory report, explicit decision-rules JSON containing sequence/D1/D2/D3 rules, explicit disjoint confirmatory seed lists. Validate disjointness and write `preregistration.json` plus `preregistration.sha256`. The command never invents threshold values.

- [ ] **Step 6: Implement `validate-controls`**

Arguments: frozen preregistration and output directory. Run fresh confirmatory reference controls only. Write `confirmatory_controls.json`; exit nonzero on `BALANCE_FAILURE`, `MATCH_FAILURE`, custody failure, or `CONTROL_MISMATCH`.

- [ ] **Step 7: Add manifest-integrity tests**

Every output directory gets `MANIFEST.json` mapping relative paths to SHA-256. Identical inputs produce identical hashes. Confirmatory output contains the preregistration hash.

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

Expected: no provider integration and no scalar `aaa_score`/`self_improvement` output field. A match inside a negative test assertion is allowed only when that test proves the name is absent from produced output.

- [ ] **Step 10: Commit**

```bash
git add src/aaa_v0/cli.py tests/test_cli.py
git commit -m "feat: add AAA offline assay workflow"
```

---

## Final implementation gate

After Task 9, do not calibrate scientifically and do not run a frontier model. Run fresh verification and create `IMPLEMENTATION_STATE.json` from actual repository/runtime values. `implementation_head` is populated by the implementation script from `git rev-parse HEAD`, never copied manually from this plan.

Required fixed fields:

```json
{
  "concept_commit": "45061c6ea30edf4c16833c3deca0b13c2d98bb8f",
  "assay_design_commit": "04f0f4380d4fad01386c1f61e95a1e65c24f8de0",
  "history_balance": "VERIFIED_ON_TEST_FIXTURES_ONLY",
  "seed_boundary": "VERIFIED_ON_TEST_FIXTURES_ONLY",
  "reference_controls": "SOFTWARE_VALIDATED_ONLY",
  "preregistration": "NOT_FROZEN",
  "confirmatory_controls": "NOT_RUN",
  "frontier_execution": "NOT_RUN",
  "aaa_evidence": "NONE"
}
```

The generated record additionally contains `implementation_head` from Git and exact pytest pass/fail counts under `unit_tests`.

`SOFTWARE_VALIDATED_ONLY` means deterministic tests exercise intended code paths. It is neither exploratory calibration nor the confirmatory scientific control gate.

If any automated test, balance audit, sequence-balance check, seed-boundary check, custody check, or scope-leak check fails, stop and repair implementation before any later calibration action.

## Explicitly outside this plan

The following require later explicit approval:

- frontier-model/provider adapters;
- a frontier-model prompt protocol;
- exploratory tuning using frontier-model behavior;
- scientifically freezing official AAA-v0 thresholds/rules;
- confirmatory scientific control execution;
- frontier-model execution;
- C1/C2/C3/C4 interpretation for a frontier model;
- claims about AGI, recursive self-improvement, ARC-AGI-4, or general corrigibility.
