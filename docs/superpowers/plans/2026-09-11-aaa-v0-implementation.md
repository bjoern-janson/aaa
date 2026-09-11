# AAA-v0 Assay Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, auditable AAA-v0 assay kernel that can generate balanced treatment/control histories, enforce the terminal boundary, produce D1/D2/D3 future tasks, run the three reference controls, record paired learning curves and corrective-recovery trajectories, freeze a preregistration artifact after exploratory calibration, and validate the frozen controls on fresh confirmatory instances—without executing or interpreting any frontier model.

**Architecture:** The implementation is a small Python package split into immutable domain contracts, paired-history generation/audit, future-task/provenance generation, task rendering/environment mechanics, reference agents, measurement, and protocol orchestration. Scientific choices that the approved design leaves open are represented as explicit serialized configuration; exploratory calibration may choose them, but confirmatory execution accepts only a frozen preregistration artifact and fresh post-freeze seeds. The implementation never equates treatment/control internal state and never treats a control-signature failure as evidence about an evaluated model.

**Tech Stack:** Python 3.12+, standard library, NumPy 2.x, pytest 8.x. No network dependency in the assay kernel. JSON/JSONL are the custody formats; SHA-256 is used for content addressing and provenance.

**Spec:** `docs/superpowers/specs/2026-09-11-aaa-v0-assay-design.md`

## Global Constraints

- Concept parent is `45061c6ea30edf4c16833c3deca0b13c2d98bb8f`; assay-design parent is `04f0f4380d4fad01386c1f61e95a1e65c24f8de0`.
- The only manipulated acquisition-history regularity is the `Z`–`Q*` dependency: treatment has `I(Z;Q*) > 0`; control destroys that dependency while preserving the frozen marginals and sequence-balance contract.
- Treatment/control histories use the same ordered `z` sequence, same ordered `theta` sequence, same episode templates, and same presentation/resource schedule. `q*` assignments may differ only as required by the treatment.
- Acquisition history is passive; no exploratory action is selected during the history phase.
- Future targets and future task realizations become available only after the terminal boundary is frozen.
- Matching is bounded observable matching at `b=0`; it never authorizes hidden-state identity.
- Primary measurement is the full paired curve `Delta C(b)` at every frozen budget point. Any area-under-curve quantity is secondary.
- D1 measures surface transfer; D2 measures representation-shift transfer; D3 measures corrective recovery after `phi -> phi'`. D3 success is not defined by raw final competence.
- Reference-control signature is `LOCAL = (-,-,-)`, `FIXED_META = (+,+,-)`, `REVISING_META = (+,+,+)`, with D3 signs referring to the frozen corrective-recovery criterion.
- Exploratory control calibration and confirmatory control validation use disjoint seeds/instances. Numerical thresholds are frozen between them in a preregistration artifact.
- `MATCH_FAILURE` and `CONTROL_MISMATCH` are hard stops. Neither may be reinterpreted as a model result.
- No frontier-model adapter, model execution, or empirical AAA interpretation is in scope for this plan.
- No implementation task may modify `README.md`, `formalization/AAA_CONCEPT_V0.md`, `formalization/CLAIM_HIERARCHY_V0.md`, or the approved assay-design spec.

---

## Planned file map

```text
pyproject.toml
src/aaa_v0/
    __init__.py
    contracts.py          # immutable domain/config/result dataclasses
    serialization.py      # canonical JSON + SHA-256 helpers
    histories.py          # paired H+/H0 construction
    balance.py            # exact marginal/sequence leakage audits
    provenance.py         # protocol-phase state machine and seed custody
    tasks.py              # latent future task generation + D1/D2/D3 families
    environment.py        # probe/observation mechanics
    agents.py             # Agent protocol + LOCAL/FIXED_META/REVISING_META
    measurement.py        # competence, paired curves, recovery trajectories
    runner.py             # paired protocol orchestration and hard stops
    prereg.py             # exploratory report + frozen prereg artifact contract
    cli.py                # offline commands for audit/calibrate/freeze/validate

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

The package deliberately has no `frontier_models.py`, provider SDK, HTTP client, or model-specific prompt adapter.

---

### Task 1: Project skeleton and immutable contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/aaa_v0/__init__.py`
- Create: `src/aaa_v0/contracts.py`
- Create: `src/aaa_v0/serialization.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: `AssayConfig`, `HistoryEpisode`, `HistoryPair`, `LatentTask`, `SurfaceView`, `ProbeObservation`, `TrialRecord`, `ProtocolStatus`, `canonical_json_bytes()`, `sha256_json()`.
- All later tasks consume these exact names; dataclasses are frozen and JSON-serializable through explicit `to_dict()` methods.

- [ ] **Step 1: Add packaging metadata and dependencies**

Create `pyproject.toml` with package name `aaa-v0`, Python floor `>=3.12`, runtime dependency `numpy>=2.0,<3`, test dependency `pytest>=8,<9`, package root `src`, and pytest configured with `testpaths = ["tests"]`.

- [ ] **Step 2: Write failing contract tests**

Create `tests/test_contracts.py` covering:

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


def test_history_episode_separates_latent_and_presented_fields():
    ep = HistoryEpisode(
        episode_id="h-000",
        z=0,
        theta=2,
        q_star=1,
        surface_seed=7,
        resource_units=1,
    )
    assert ep.to_dict()["q_star"] == 1
    assert ep.resource_units == 1
```

- [ ] **Step 3: Run the contract tests and verify RED**

Run:

```bash
pytest tests/test_contracts.py -v
```

Expected: collection/import failure because `aaa_v0.contracts` does not exist.

- [ ] **Step 4: Implement the immutable contracts**

In `contracts.py`, implement frozen dataclasses with explicit types and validation in `__post_init__`:

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
        return cls(
            z_count=4,
            q_count=4,
            theta_count=4,
            history_repeats=8,
            future_tasks_per_family=32,
            max_budget=4,
            d3_change_after=8,
        )

    def to_dict(self) -> dict[str, int]: ...


@dataclass(frozen=True)
class HistoryEpisode:
    episode_id: str
    z: int
    theta: int
    q_star: int
    surface_seed: int
    resource_units: int
    def to_dict(self) -> dict[str, object]: ...


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
    nuisance_values: tuple[int, ...]
    surface_seed: int
    phase_index: int


@dataclass(frozen=True)
class SurfaceView:
    task_id: str
    family: str
    context_tokens: tuple[str, ...]
    probe_tokens: tuple[str, ...]
    observable_structure: tuple[int, ...]


@dataclass(frozen=True)
class ProbeObservation:
    probe_index: int
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
    chosen_probe: int | None
    old_rule_adherence: bool | None


@dataclass(frozen=True)
class ProtocolStatus:
    match_ok: bool
    controls_ok: bool
    stop_code: str | None
```

Validation must reject non-positive counts, `q_count != z_count` in v0, `max_budget < 1`, and `d3_change_after < 1`.

- [ ] **Step 5: Implement deterministic serialization**

In `serialization.py`:

```python
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
```

- [ ] **Step 6: Run Task 1 tests and full suite**

Run:

```bash
pytest tests/test_contracts.py -v
pytest -q
```

Expected: all collected tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add pyproject.toml src/aaa_v0/__init__.py src/aaa_v0/contracts.py src/aaa_v0/serialization.py tests/test_contracts.py
git commit -m "feat: add AAA-v0 immutable assay contracts"
```

---

### Task 2: Paired acquisition-history generator and exact balance audit

**Files:**
- Create: `src/aaa_v0/histories.py`
- Create: `src/aaa_v0/balance.py`
- Test: `tests/test_histories.py`
- Test: `tests/test_balance.py`

**Interfaces:**
- Consumes: `AssayConfig`, `HistoryEpisode`, `HistoryPair`.
- Produces: `make_history_pair(config, seed) -> HistoryPair`, `BalanceReport`, `audit_history_pair(pair) -> BalanceReport`, `empirical_mutual_information(pairs) -> float`.
- `BalanceReport.ok` is the only value the runner may use to authorize progression beyond history construction.

- [ ] **Step 1: Write failing paired-history tests**

Create tests asserting that, for `AssayConfig.exploratory_default()` and seed `17`:

```python
pair = make_history_pair(cfg, seed=17)
assert [e.z for e in pair.treatment] == [e.z for e in pair.control]
assert [e.theta for e in pair.treatment] == [e.theta for e in pair.control]
assert [e.surface_seed for e in pair.treatment] == [e.surface_seed for e in pair.control]
assert [e.resource_units for e in pair.treatment] == [e.resource_units for e in pair.control]
assert sorted(e.q_star for e in pair.treatment) == sorted(e.q_star for e in pair.control)
assert empirical_mutual_information((e.z, e.q_star) for e in pair.treatment) > 0.0
assert empirical_mutual_information((e.z, e.q_star) for e in pair.control) == 0.0
```

Also assert treatment obeys `q_star == treatment_phi[z]` on every episode.

- [ ] **Step 2: Run the history test and verify RED**

```bash
pytest tests/test_histories.py -v
```

Expected: import failure for `aaa_v0.histories`.

- [ ] **Step 3: Implement an exactly balanced finite design**

Implement `make_history_pair()` so that each `z` appears exactly `history_repeats * q_count` times. For each `z`, treatment uses fixed `phi[z]`; control cycles every `q_star` equally often within that same `z`, making the empirical joint distribution factor exactly while preserving the global `Q*` marginal. Use a seeded permutation to order complete balanced blocks, but apply the identical ordered `z`, `theta`, `surface_seed`, template/resource schedule to both arms.

`theta` must be generated independently from `(z, q_star)` using a separate RNG stream derived from `seed` by SHA-256 domain separation (`"history-theta"`, `"history-order"`, `"history-surface"`, `"history-phi"`).

- [ ] **Step 4: Write failing balance-audit tests**

Tests must verify:

```python
report = audit_history_pair(pair)
assert report.ok
assert report.mismatched_fields == ()
assert report.treatment_mi > 0.0
assert report.control_mi == 0.0
```

Then mutate one serialized control episode's `resource_units`, ordering, or `theta` and reconstruct a `HistoryPair`; each mutation must produce `report.ok is False` with the corresponding named mismatch.

- [ ] **Step 5: Implement `BalanceReport` and balance audit**

`BalanceReport` is a frozen dataclass containing:

```python
ok: bool
mismatched_fields: tuple[str, ...]
treatment_mi: float
control_mi: float
treatment_hash: str
control_hash: str
non_treatment_sequence_hash_treatment: str
non_treatment_sequence_hash_control: str
```

The non-treatment sequence fingerprint must include `(episode_id, z, theta, surface_seed, resource_units)` in order and must exclude `q_star` by construction. Exact equality of these fingerprints is required.

Compute empirical MI directly from integer counts using natural logs; normalize values with absolute magnitude `< 1e-15` to exact `0.0` before evaluating the control condition.

- [ ] **Step 6: Add a treatment-isolation regression test**

Serialize treatment and control histories with `q_star` redacted. Assert byte-identical canonical serialization. This test is the executable form of “the sole declared difference is the `Z`–`Q*` association.”

- [ ] **Step 7: Run Task 2 and full tests**

```bash
pytest tests/test_histories.py tests/test_balance.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/aaa_v0/histories.py src/aaa_v0/balance.py tests/test_histories.py tests/test_balance.py
git commit -m "feat: add balanced AAA history intervention"
```

---

### Task 3: Terminal-boundary and seed-custody state machine

**Files:**
- Create: `src/aaa_v0/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces: `ProtocolPhase`, `SeedCustody`, `TerminalSnapshot`, `FutureSeedReveal`.
- `SeedCustody.reveal_future_seed()` is impossible before `freeze_terminal()` returns a snapshot.
- Runner code in later tasks receives future seeds only through `FutureSeedReveal`.

- [ ] **Step 1: Write failing custody tests**

Tests must assert the following sequence:

```python
custody = SeedCustody(history_seed=11, sealed_future_seed=29)
assert custody.phase is ProtocolPhase.HISTORY
with pytest.raises(ProtocolOrderError):
    custody.reveal_future_seed()

snapshot = custody.freeze_terminal(
    treatment_state_hash="a" * 64,
    control_state_hash="b" * 64,
)
assert custody.phase is ProtocolPhase.TERMINAL_FROZEN
reveal = custody.reveal_future_seed(snapshot)
assert reveal.seed == 29
assert custody.phase is ProtocolPhase.FUTURE_REVEALED
```

Also assert a snapshot from another custody object is rejected.

- [ ] **Step 2: Run the provenance test and verify RED**

```bash
pytest tests/test_provenance.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement custody with commitments**

`SeedCustody` must store the future seed privately, expose only `future_seed_commitment = SHA256("aaa-v0-future-seed:" + decimal_seed)`, and maintain a monotonically advancing phase enum:

```python
class ProtocolPhase(Enum):
    HISTORY = "HISTORY"
    TERMINAL_FROZEN = "TERMINAL_FROZEN"
    FUTURE_REVEALED = "FUTURE_REVEALED"
```

`TerminalSnapshot` must contain treatment/control state hashes, history pair hash, config hash, future-seed commitment, and a snapshot hash covering all prior fields.

- [ ] **Step 4: Add tamper tests**

Assert changing a state hash, config hash, or future seed after snapshot construction causes verification failure. Assert the revealed seed hashes to the pre-boundary commitment.

- [ ] **Step 5: Run Task 3 tests and full suite**

```bash
pytest tests/test_provenance.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/aaa_v0/provenance.py tests/test_provenance.py
git commit -m "feat: enforce AAA terminal seed boundary"
```

---

### Task 4: Latent future tasks, D1/D2/D3 rendering, and environment mechanics

**Files:**
- Create: `src/aaa_v0/tasks.py`
- Create: `src/aaa_v0/environment.py`
- Test: `tests/test_tasks.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: `AssayConfig`, `LatentTask`, `SurfaceView`, `ProbeObservation`, `FutureSeedReveal`.
- Produces: `make_future_tasks(config, reveal, phi) -> dict[str, tuple[LatentTask, ...]]`, `render_task(task, shift_seed) -> SurfaceView`, `MetaProbeEnvironment`.
- D2 alters only surface labels/order while preserving `observable_structure`; D3 alters `q_star` after the fixed change point and does not apply the D2 surface transform.

- [ ] **Step 1: Write failing task-family tests**

Generate D1/D2/D3 with a fixed reveal and assert:

```python
families = make_future_tasks(cfg, reveal, phi=(2, 0, 3, 1))
assert set(families) == {"D1", "D2", "D3"}
assert len(families["D1"]) == cfg.future_tasks_per_family
assert all(t.family == "D1" for t in families["D1"])
```

For paired D1/D2 task indexes, assert identical `(z, theta, q_star, nuisance_values)` and different surface seeds. For D3, assert `q_star == phi[z]` before `d3_change_after` and `q_star == phi_prime[z]` afterward, where `phi_prime[z] != phi[z]` for every `z`.

- [ ] **Step 2: Run task tests and verify RED**

```bash
pytest tests/test_tasks.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement domain-separated future RNG streams**

Derive independent NumPy generators from the revealed seed using SHA-256 labels `future-latent`, `future-nuisance`, `future-d1-surface`, `future-d2-surface`, and `future-d3-surface`. Future `theta` values are uniform over `range(theta_count)` and independent of history generation.

- [ ] **Step 4: Implement D2 rendering as surface permutation only**

`SurfaceView` exposes an observable structural tuple and surface tokens. For v0, use a simple public structural signature:

```python
observable_structure = tuple((z + offset) % cfg.z_count for offset in range(cfg.z_count))
```

D1 maps positions to canonical tokens `ctx-0..` and `probe-0..`. D2 applies independent seeded bijections to both token vocabularies and a seeded presentation-order permutation while leaving `observable_structure` unchanged. Reference agents must consume only `SurfaceView`, never latent `z` or `q_star`.

- [ ] **Step 5: Write failing environment tests**

For a latent task with `theta=3` and `q_star=2`:

```python
env = MetaProbeEnvironment(task)
assert env.probe(2) == ProbeObservation(probe_index=2, informative=True, value=3)
assert env.probe(0).informative is False
assert env.probe(0).value is None
```

Assert probing beyond `max_budget` raises `BudgetExceeded` and repeated calls are counted as separate budget units.

- [ ] **Step 6: Implement environment mechanics**

Only the informative probe returns the target value. All other probes return `(informative=False, value=None)`. This deliberately simple mechanics makes the first existence assay about probe-selection efficiency rather than inference under observation noise.

- [ ] **Step 7: Add renderer-leakage tests**

Assert no serialized `SurfaceView` field contains `theta`, `q_star`, latent `z`, or the literal integer mapping used by `phi`. Assert D2 rendered tokens differ from D1 for every configured token while `observable_structure` is identical.

- [ ] **Step 8: Run Task 4 and full tests**

```bash
pytest tests/test_tasks.py tests/test_environment.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/aaa_v0/tasks.py src/aaa_v0/environment.py tests/test_tasks.py tests/test_environment.py
git commit -m "feat: add AAA future task families"
```

---

### Task 5: Agent protocol and the three reference controls

**Files:**
- Create: `src/aaa_v0/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `Agent` protocol, `LocalAgent`, `FixedMetaAgent`, `RevisingMetaAgent`, `EpisodeDecision`.
- All agents receive passive history only through `observe_history(views: tuple[SurfaceView, ...], informative_probe_tokens: tuple[str, ...])` and future tasks only through `start_task(view)`, `choose_probe()`, `observe_probe(obs)`, `guess_theta()`.
- No reference agent receives latent `z`, `q_star`, `theta`, or `phi` directly.

- [ ] **Step 1: Write failing no-omniscience interface tests**

Use `inspect.signature` to assert agent public methods accept `SurfaceView`/`ProbeObservation` only and have no parameter named `z`, `q_star`, `theta`, or `phi`.

- [ ] **Step 2: Run agent tests and verify RED**

```bash
pytest tests/test_agents.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement `LocalAgent`**

`LocalAgent` discards history and probes in visible order. It may solve a local task when it eventually reaches the informative probe but cannot exploit history coherence. Its state resets between future tasks except for deterministic RNG/presentation bookkeeping.

- [ ] **Step 4: Implement `FixedMetaAgent`**

During passive history, infer a mapping from `observable_structure` to the surface-independent structural position of the informative probe. On future tasks, select that predicted structural probe first. The learned mapping is frozen after acquisition history; future mismatch does not alter it.

- [ ] **Step 5: Implement `RevisingMetaAgent`**

Use the same initial mapping as `FixedMetaAgent`, plus per-structural-context counts of post-boundary successes/failures. On an uninformative first probe, search remaining probes in visible order; once an informative probe is found, update the context mapping immediately to that probe's structural position. This is an intentionally transparent v0 change-point learner, not a claim about optimal meta-learning.

- [ ] **Step 6: Add qualitative unit tests independent of the full runner**

Construct minimal hand-authored `SurfaceView`/observation sequences and assert:

```python
# FIXED_META keeps the old first probe after a mismatch.
assert fixed.choose_probe() == old_probe

# REVISING_META changes its preferred first probe after observing the new informative role.
assert revising.choose_probe() == new_probe
```

Also assert `LocalAgent` behavior is unchanged by two different history sequences.

- [ ] **Step 7: Run Task 5 and full tests**

```bash
pytest tests/test_agents.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/aaa_v0/agents.py tests/test_agents.py
git commit -m "feat: add AAA reference controls"
```

---

### Task 6: Paired learning-curve and D3 recovery measurement

**Files:**
- Create: `src/aaa_v0/measurement.py`
- Test: `tests/test_measurement.py`

**Interfaces:**
- Consumes: tuples of `TrialRecord`.
- Produces: `LearningCurve`, `PairedCurve`, `RecoveryTrajectory`, `compute_learning_curve()`, `compute_paired_curve()`, `compute_recovery_trajectory()`.
- No function in this module authorizes C1–C4; it computes observables only.

- [ ] **Step 1: Write failing paired-curve tests**

Hand-author trial records for two tasks and budgets `0..2`; assert exact pointwise treatment/control competence means and exact `delta` values. The test must include two different curve shapes with the same final score so the code cannot reduce the primary endpoint to final accuracy.

- [ ] **Step 2: Run measurement tests and verify RED**

```bash
pytest tests/test_measurement.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement curve dataclasses and computations**

Use frozen dataclasses:

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

Compute AUC only as trapezoidal integration over frozen integer budget points and label it `secondary_auc_delta` in serialized output.

- [ ] **Step 4: Add D3 semantic tests**

Construct one brittle and one revising trajectory with equal final competence but different old-rule adherence after invalidation. Assert the recovery outputs remain distinct. This protects the rule that D3 is not a final-score test.

- [ ] **Step 5: Run Task 6 and full tests**

```bash
pytest tests/test_measurement.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/aaa_v0/measurement.py tests/test_measurement.py
git commit -m "feat: measure paired AAA learning dynamics"
```

---

### Task 7: Protocol runner, matching gate, and hard-stop semantics

**Files:**
- Create: `src/aaa_v0/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: config, history pair, seed custody, agent factories, future tasks.
- Produces: `RunBundle` containing custody hashes, balance report, `b=0` match report, D1/D2 paired curves, D3 recovery trajectories, raw trial records, and `ProtocolStatus`.
- Exposes: `run_reference_pair(...) -> RunBundle`, `assert_match_gate(...)`, `assert_control_gate(...)`.

- [ ] **Step 1: Write failing match-gate tests**

Create a synthetic pair of `b=0` records with equal competence and assert pass. Change one condition so its zero-budget competence differs by more than supplied tolerance and assert:

```python
with pytest.raises(ProtocolStop) as exc:
    assert_match_gate(...)
assert exc.value.code == "MATCH_FAILURE"
```

- [ ] **Step 2: Run runner tests and verify RED**

```bash
pytest tests/test_runner.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement future episode stepping**

For each latent future task and budget point:

1. create paired environment copies;
2. call the agent's current guess at `b=0` before any probe;
3. record correctness;
4. for budgets `1..B`, request exactly one probe, return its observation, request a new guess, and record it;
5. never allow one condition to consume extra probes or retries.

The treatment and control conditions must use the same latent task object and equivalent rendered copies.

- [ ] **Step 4: Implement the matching gate before D1/D2/D3 interpretation**

`assert_match_gate()` accepts an explicit tolerance argument supplied by config/preregistration and evaluates only the declared zero-budget competence diagnostic. It raises `ProtocolStop("MATCH_FAILURE")` before any claim-status computation.

- [ ] **Step 5: Implement raw custody output**

`RunBundle.to_dict()` must contain:

```text
config_hash
history_pair_hash
balance_report
terminal_snapshot_hash
future_seed_commitment
future_seed_reveal_hash
raw_trial_records_hash
D1_paired_curve
D2_paired_curve
D3_recovery_trajectory
protocol_status
```

It must not contain a field named `aaa_score`, `intelligence`, or `self_improvement`.

- [ ] **Step 6: Add end-to-end deterministic runner test**

Run the same config/history/future seeds twice with `FixedMetaAgent` and assert byte-identical canonical `RunBundle` serialization. Change only the future seed and assert history/custody hashes remain identical while future task/raw-record hashes change.

- [ ] **Step 7: Run Task 7 and full tests**

```bash
pytest tests/test_runner.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 7**

```bash
git add src/aaa_v0/runner.py tests/test_runner.py
git commit -m "feat: add AAA protocol runner and stop gates"
```

---

### Task 8: Exploratory calibration, preregistration artifact, and fresh confirmatory control validation

**Files:**
- Create: `src/aaa_v0/prereg.py`
- Test: `tests/test_prereg.py`

**Interfaces:**
- Produces: `ExploratoryCalibrationReport`, `Preregistration`, `ConfirmatoryControlReport`, `freeze_preregistration()`, `validate_confirmatory_controls()`.
- `Preregistration` contains exact config, match tolerance, D1/D2 transfer decision rules, D3 recovery rule, exploratory-seed hashes, and an explicit disjoint confirmatory seed schedule.
- Once serialized, preregistration identity is its SHA-256 hash.

- [ ] **Step 1: Write failing prereg immutability tests**

Assert a frozen prereg artifact round-trips JSON exactly and rejects any confirmatory seed that appears in the exploratory seed set.

- [ ] **Step 2: Run prereg tests and verify RED**

```bash
pytest tests/test_prereg.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement exploratory report without scientific authorization**

`ExploratoryCalibrationReport` records raw reference-control curves/trajectories and candidate numerical parameters, but its `to_dict()` includes:

```json
{"status": "EXPLORATORY_UNSCORED", "frontier_interpretation_authorized": false}
```

No C1–C4 status is emitted from exploratory calibration.

- [ ] **Step 4: Implement frozen preregistration contract**

Use explicit decision-rule structures rather than arbitrary Python callables:

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

`spec_commit` must equal `04f0f4380d4fad01386c1f61e95a1e65c24f8de0` for AAA-v0.

- [ ] **Step 5: Implement confirmatory control-signature evaluation**

Given fresh `RunBundle`s for `LOCAL`, `FIXED_META`, and `REVISING_META`, evaluate the frozen rules and require exactly:

```text
               D1   D2   D3
LOCAL           -    -    -
FIXED_META      +    +    -
REVISING_META   +    +    +
```

D3 signs use only `RecoveryRule`. If any cell differs, return:

```json
{"status": "CONTROL_MISMATCH", "frontier_interpretation_authorized": false}
```

If all cells match and all balance/match/custody gates passed, return:

```json
{"status": "CONTROL_VALIDATED", "frontier_interpretation_authorized": true}
```

The boolean means only that a later frontier execution may begin under a separate approved protocol; it does not itself establish AAA evidence.

- [ ] **Step 6: Add anti-tuning tests**

Assert confirmatory evaluation refuses bundles whose seed hashes intersect the preregistered exploratory seed hashes, whose spec commit differs, or whose config hash differs from the preregistration.

- [ ] **Step 7: Run Task 8 and full tests**

```bash
pytest tests/test_prereg.py -v
pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 8**

```bash
git add src/aaa_v0/prereg.py tests/test_prereg.py
git commit -m "feat: add AAA preregistration and control validation"
```

---

### Task 9: Offline CLI, custody files, and implementation verification

**Files:**
- Create: `src/aaa_v0/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces an offline CLI with subcommands `audit-history`, `calibrate-controls`, `freeze-prereg`, and `validate-controls`.
- No subcommand named `run-frontier`, `score-model`, or equivalent is permitted in AAA-v0 kernel implementation.

- [ ] **Step 1: Write failing CLI surface tests**

Assert `python -m aaa_v0.cli --help` advertises exactly the four allowed subcommands. Assert `run-frontier` is rejected by argparse.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
pytest tests/test_cli.py -v
```

Expected: import or subprocess failure because CLI does not exist.

- [ ] **Step 3: Implement `audit-history`**

Inputs: config JSON path, history seed. Output: canonical JSON containing config hash, history pair hash, treatment/control MI, balance fields, and `status = "BALANCED" | "BALANCE_FAILURE"`. Exit nonzero on balance failure.

- [ ] **Step 4: Implement `calibrate-controls`**

Inputs: config JSON path, explicit exploratory history/future seed lists, output directory. Execute only the three reference controls and write raw JSONL records plus `exploratory_calibration.json` with `EXPLORATORY_UNSCORED` status.

- [ ] **Step 5: Implement `freeze-prereg`**

Inputs: exploratory report, an explicit decision-rules JSON file, explicit disjoint confirmatory seed lists. Validate disjointness and write `preregistration.json` plus `preregistration.sha256`. The command does not invent thresholds; all threshold values must be present in the supplied decision-rules file.

- [ ] **Step 6: Implement `validate-controls`**

Input: frozen preregistration path and output directory. Run fresh confirmatory reference controls only, write `confirmatory_controls.json`, and exit nonzero on `MATCH_FAILURE`, `BALANCE_FAILURE`, custody failure, or `CONTROL_MISMATCH`.

- [ ] **Step 7: Add custody-integrity CLI tests**

Use temporary directories to assert every output file's SHA-256 is listed in a `MANIFEST.json`, re-running with identical inputs gives identical file hashes, and confirmatory output contains the preregistration hash.

- [ ] **Step 8: Run all automated verification**

Run:

```bash
pytest -q
python -m aaa_v0.cli --help
```

Expected: all tests PASS; help shows only the four approved subcommands.

- [ ] **Step 9: Perform design-requirement grep checks**

Run:

```bash
grep -R "frontier\|OpenAI\|Anthropic\|Gemini" -n src/aaa_v0 tests || true
grep -R "aaa_score\|self_improvement" -n src/aaa_v0 tests || true
```

Expected: no model-provider integration and no single “AAA score”/self-improvement field. Any occurrence in a negative test assertion must be inspected and documented as such.

- [ ] **Step 10: Commit Task 9**

```bash
git add src/aaa_v0/cli.py tests/test_cli.py
git commit -m "feat: add AAA offline assay workflow"
```

---

## Final implementation gate

After Task 9, do **not** run a frontier model. Perform a fresh verification pass and produce an implementation-state record containing:

```text
concept_commit       45061c6ea30edf4c16833c3deca0b13c2d98bb8f
assay_design_commit  04f0f4380d4fad01386c1f61e95a1e65c24f8de0
implementation_head  <actual implementation commit>
unit_tests           PASS/FAIL with exact count
history_balance      VERIFIED/FAILED on test fixtures only
seed_boundary        VERIFIED/FAILED on test fixtures only
reference_controls   SOFTWARE-VALIDATED/NOT-VALIDATED
preregistration      NOT_FROZEN unless an explicit later approval froze it
frontier_execution   NOT_RUN
AAA_evidence         NONE
```

The phrase `SOFTWARE-VALIDATED` means the deterministic test fixtures exercise the intended code paths. It is not the confirmatory scientific control gate. Exploratory calibration, preregistration freeze, and fresh confirmatory control validation remain separate protocol actions even though this plan implements their tooling.

If any automated test, balance audit, terminal-boundary check, or deterministic custody check fails, stop and repair the implementation before any calibration action.

## Explicitly outside this plan

The following require a later approval/gate and are not to be implemented or executed under this plan:

- any frontier-model/provider adapter;
- any prompt protocol for a frontier model;
- exploratory tuning using frontier-model behavior;
- an official preregistration freeze with scientific thresholds;
- confirmatory scientific control execution unless separately authorized after implementation review;
- frontier-model execution;
- C1/C2/C3/C4 interpretation for a frontier model;
- claims about AGI, recursive self-improvement, ARC-AGI-4, or general corrigibility.
