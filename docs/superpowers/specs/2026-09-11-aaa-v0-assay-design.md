# AAA-v0 Assay Design

**Status:** design-only specification  
**Branch:** `design/aaa-v0-assay`  
**Concept parent:** `45061c6ea30edf4c16833c3deca0b13c2d98bb8f` (`Freeze AAA concept V0`)  
**Implementation:** not started  
**Execution:** not started  

## 1. Purpose

AAA-v0 is a deliberately narrow existence/identification assay for **Acquired Adaptive Advantage (AAA)**.

It asks one bounded question:

```math
\boxed{\text{Can a controlled history change future skill-acquisition dynamics?}}
```

The intended causal structure is:

```text
history intervention
      -> matched terminal boundary
      -> fresh post-boundary unknown
      -> paired future learning curves
```

AAA-v0 does **not** attempt to instantiate or measure unrestricted recursive self-improvement, AGI, open-ended innovation, general corrigibility, or an official ARC-AGI-4 capability.

The assay must remain smaller than the phenomenon it is intended to detect.

## 2. Frozen conceptual dependencies

AAA-v0 inherits, without modifying, the concept-level commitments in:

- `README.md`
- `formalization/AAA_CONCEPT_V0.md`
- `formalization/CLAIM_HIERARCHY_V0.md`

In particular:

```math
\boxed{\text{what changed} \neq \text{whether it improved} \neq \text{whether it improved future learning}}
```

and:

```math
\boxed{C_1\nRightarrow C_2\nRightarrow C_3\nRightarrow C_4.}
```

This design must not silently strengthen those claims.

## 3. Design principle

The central identification strategy is:

```math
\boxed{\text{Match current observable competence; compare subsequent learnability.}}
```

AAA-v0 does not seek hidden-state equality between treatment and control agents. Instead, it constructs future tasks whose fresh target values are generated after the terminal boundary and are unavailable from either acquisition history.

Thus the obvious alternative explanation—"the treatment agent simply already knew the future answer"—is blocked by construction rather than by assuming internal equality.

## 4. Task family

Each task `tau` contains three conceptually distinct objects:

- a structural context `z \in Z`;
- a set of possible probe roles `Q`;
- a fresh task-specific latent target `theta_tau \in Theta`.

The task-specific target is drawn independently of acquisition history:

```math
\theta_\tau \sim P_\Theta,
\qquad
\theta_\tau \perp H_{\rm past}.
```

An agent cannot solve a future task merely by knowing `z`. It must acquire information about the fresh `theta_tau` during the future episode.

A hidden cross-task meta-relation

```math
\phi:Z\to Q
```

determines which probe role is maximally informative about `theta_tau` in a given structural context.

Therefore the object-level task asks:

> What is the new task-specific `theta_tau`?

while the cross-task regularity offers the learnable meta-level lesson:

> Given structural context `z`, which kind of probe should I use to learn `theta_tau` efficiently?

The assay is designed so that history can teach **how to acquire information** without revealing future task answers.

## 5. The sole manipulated cross-task regularity

This requirement is rigid.

Treatment and control acquisition histories must match every declared marginal and resource quantity, including at minimum:

```math
P(Z),\quad P(Q^\star),\quad P(\theta),\quad
N_{\rm episodes},\quad
\text{per-episode information},\quad
\text{presentation budget},\quad
\text{resource budget}.
```

Where practical, paired treatment/control histories should use the same ordered multiset of episode shells and the same ordered `theta` values.

The histories differ only in the cross-task dependency between structural context and informative probe role:

### Treatment history `H+`

A stable relation holds across episodes:

```math
q_i^\star=\phi(z_i).
```

Hence:

```math
I(Z;Q^\star)_{H^+}>0.
```

### Control history `H0`

The same `z` and `q*` marginals are preserved while their pairing is prospectively balanced so the stable cross-task relation is destroyed:

```math
I(Z;Q^\star)_{H^0}=0
```

up to the exact finite-design tolerance specified in the later preregistration.

The control construction must not change episode count, target distribution, task difficulty, amount of solved information, or available resources.

If any additional cross-task regularity differs between arms, the design is invalid until repaired.

## 6. Passive acquisition history

AAA-v0 uses **passive, standardized solved episodes** during the acquisition-history phase.

The history phase should expose, in a fixed-format episode record:

- the structural context;
- the local task target or resolved outcome;
- the probe roles and their observed informativeness;
- the successful resolution of the episode.

The agent is not asked to choose exploratory actions during history acquisition.

This intentionally removes a major confound surface. AAA-v0 therefore does not initially mix:

```text
exploration + world-model induction + meta-learning.
```

It asks only whether repeated controlled experience can alter later acquisition dynamics.

Interactive acquisition histories may be studied in later assays, but they are outside AAA-v0.

## 7. Terminal boundary and future-target provenance

The causal effect of interest must cross an explicit temporal boundary.

Protocol order:

```text
1. freeze treatment/control histories
2. expose histories to the two conditions
3. freeze terminal agent artifacts/state references
4. derive or reveal the future-task seed
5. generate future D1/D2/D3 tasks
6. begin future learning measurement
```

Future `theta` values must not be instantiated or available to the evaluated agents before step 4.

The later preregistration should use an auditable prospective seed mechanism. The exact randomness source is not fixed by this design document.

For every future task:

```math
P(\theta_\tau\mid H^+)=P(\theta_\tau\mid H^0)=P(\theta_\tau).
```

## 8. Matching gate

AAA-v0 does not interpret matching as hidden-state identity.

The future phase begins with a frozen zero-experience diagnostic at learning budget `b=0`.

Let:

```math
C_A(\tau,b)
```

denote competence on task `tau` after future learning budget `b`.

Because the task-specific target is fresh, the design aims for terminal parity by construction:

```math
C_{A_{H^+}}(\tau,0)
\approx
C_{A_{H^0}}(\tau,0)
```

under a prospectively frozen tolerance and diagnostic rule.

If that gate fails:

```text
MATCH_FAILURE -> STOP -> no AAA interpretation
```

The authorized matching claim is only bounded observable equivalence under the declared diagnostic contract.

## 9. Primary endpoint: the paired learning curve

The **primary endpoint is the entire paired future learning-curve difference**, not an area-under-curve scalar.

For every preregistered budget point

```math
b\in\{0,1,\ldots,B\},
```

define:

```math
\boxed{
\Delta C(b)
=
\mathbb E_{\tau}
\left[
C_{A_{H^+}}(\tau,b)-C_{A_{H^0}}(\tau,b)
\right].
}
```

The geometry of `Delta C(b)` is itself data. Distinct signatures may include:

- faster early acquisition;
- lower sample/action cost to reach a fixed competence;
- delayed but stronger acquisition;
- higher attainable competence under a fixed budget;
- transient advantage;
- late reversal.

The scalar transfer summary is secondary:

```math
G_{\rm transfer}
=
\mathbb E_{\tau}
\left[
\int_0^B
w(b)
\left(
C_{A_{H^+}}(\tau,b)-C_{A_{H^0}}(\tau,b)
\right)db
\right],
```

or its discrete preregistered analogue.

The exact competence metric, budget unit, weighting function, horizon, uncertainty interval, and decision threshold remain to be frozen in the preregistration after control validation.

## 10. Paired evaluation

Treatment and control agents must receive paired future tasks generated from the same latent task draws wherever the evaluated system permits this.

The task generator should therefore produce a canonical latent task object first, then render equivalent treatment/control copies.

No arm may receive easier target instances, more informative observations, more probes, or additional retry budget.

If the evaluated agent is stochastic, repeated paired runs or a prospectively specified stochastic-evaluation protocol will be required. The exact replication scheme is not fixed here.

## 11. D1 — surface transfer

D1 uses:

- fresh post-boundary `theta` values;
- the treatment meta-relation `phi`;
- the same surface/rendering family used during acquisition history.

D1 asks whether treatment history creates a future acquisition advantage under near-surface transfer.

A valid positive D1 result may support bounded `C2 — Adaptive Advantage` if all earlier gates pass.

D1 alone does not establish structural abstraction.

## 12. D2 — representation-shift transfer

D2 preserves the deeper task structure and meta-relation while applying a prospectively defined surface transformation.

The generator must separate:

```text
latent structural roles
        from
surface realization / rendering
```

D2 may transform, for example:

- symbol identities;
- colors or token names;
- spatial placement;
- ordering;
- superficial action labels;
- other representation details not constitutive of the latent role structure.

The transformation must block trivial literal lookup while preserving the structural relation needed to identify informative probe roles.

A positive D2 result after C2 may support `C3 — Transferable Advantage` under the frozen shift family.

A D2 null does not erase a valid bounded D1/C2 result.

## 13. D3 — anti-transfer / invalidation

D3 tests corrective recoverability.

To avoid conflating invalidation with representation shift, D3 should hold its surface family fixed while changing only the previously useful meta-relation at a prospectively defined change point:

```math
\phi\rightarrow\phi',
```

where `phi'` is a preregistered alternative, preferably a derangement of `phi` when the finite sets permit it.

The prior adaptation should become actively misleading, not merely irrelevant.

D3 records at minimum the trajectories of:

- old-rule adherence;
- future-task competence;
- evidence accumulated after invalidation;
- recovery after the change point.

The final `C_recovery` endpoint is not fixed in this design document. It should be frozen only after the environment demonstrably distinguishes brittle from revising reference controls.

A C4 claim requires the lower claim layers required by the frozen hierarchy plus the preregistered D3 correction criterion.

## 14. Reference-control calibration

Before any frontier-model result may be scientifically interpreted, AAA-v0 must reproduce a frozen qualitative control signature.

Three reference agents are required.

### `LOCAL`

A learner with no cross-task persistent/meta state.

Expected qualitative signature:

```text
D1: -
D2: -
D3: -
```

It may learn each local `theta`, but it should not benefit from acquisition-history coherence.

### `FIXED_META`

An explicit learner that estimates the treatment relation from acquisition history and applies it in the future phase, but does not revise that learned relation after invalidation.

Expected signature:

```text
D1: +
D2: +
D3: -
```

### `REVISING_META`

An explicit learner with the same basic cross-task capability as `FIXED_META`, but with maintained uncertainty or change-point logic sufficient to revise the learned relation under D3.

Expected signature:

```text
D1: +
D2: +
D3: +
```

The calibration matrix is therefore:

```text
               D1   D2   D3
LOCAL           -    -    -
FIXED_META      +    +    -
REVISING_META   +    +    +
```

The exact numerical thresholds corresponding to `+` and `-` are a preregistration decision, not a post-hoc interpretation.

If the frozen controls do not reproduce their required qualitative signatures:

```text
CONTROL_MISMATCH -> STOP -> no frontier-model interpretation
```

A failed control signature is a design/implementation failure, not evidence about the evaluated model.

## 15. Claim authorization

AAA-v0 inherits the concept-level hierarchy and maps its assay stages onto it conservatively.

### C1 — Acquisition

Authorized only if treatment history produces a reproducible difference in subsequent acquisition dynamics after the matching gate.

### C2 — Adaptive Advantage

Authorized only if the difference is advantageous on the prospectively frozen novel-task criterion, including the full learning-curve decision rule.

### C3 — Transferable Advantage

Authorized only if C2 is established and the advantage survives the frozen D2 representation shift.

### C4 — Correctable Advantage

Authorized only if the required lower layers are established and the acquired advantage satisfies the frozen D3 corrective-recovery criterion.

No forward implication is automatic.

## 16. Null and negative results

Nulls remain local.

Examples:

- `D1 = null` blocks C2 in this assay but does not refute the AAA concept;
- `D1 > 0, D2 = null` supports bounded C2 while blocking C3;
- `D1 > 0, D2 > 0, D3 = fail` supports the lower layers while blocking C4;
- negative transfer can be informative about the scope or brittleness of the acquired adaptation.

Do not reinterpret failed higher layers as failure of established lower layers.

## 17. Confounds to attack before implementation

The design is not ready for implementation until the following are explicitly checked.

### 17.1 Marginal leakage

Verify exact or prospectively tolerated equality of treatment/control marginals and episode resources.

### 17.2 Sequence leakage

Treatment/control histories must not differ unintentionally in ordering statistics, repetition structure, transition frequencies, or other cues correlated with condition.

### 17.3 Information-volume leakage

The treatment condition must not contain more total object-level information about solved history tasks than control.

### 17.4 Future-target leakage

No future `theta` or future-task realization may be available before the terminal boundary.

### 17.5 Renderer leakage in D2

D2 must not accidentally preserve a literal identifier that trivializes transfer, nor destroy the deeper structure needed by the intended positive controls.

### 17.6 D3 confounding

D3 should isolate invalidation of `phi`; it must not simultaneously introduce an unrelated representation shift or budget change.

### 17.7 Control omniscience

Reference agents must receive only the information available under the assay contract. They must not be handed latent variables unavailable to evaluated agents unless such access is explicitly part of a separate theorem/control layer.

### 17.8 Resource asymmetry

Treatment/control future evaluations must have equal action, observation, token, compute, retry, and wall-clock rules to the extent these are under assay control.

### 17.9 Post-hoc task selection

Future task families, transformation families, and decision rules must be prospectively frozen before frontier-model execution.

## 18. Design gates

The intended development sequence is:

```text
CONCEPT V0 FROZEN
      |
      v
ASSAY DESIGN
      |
      v
FORMAL/GENERATOR AUDIT
      |
      v
REFERENCE-CONTROL CALIBRATION
      |
      v
PREREGISTRATION FREEZE
      |
      v
IMPLEMENTATION VALIDATION
      |
      v
FRONTIER-MODEL EXECUTION
      |
      v
BOUNDED INTERPRETATION
```

No later gate may authorize interpretation when an earlier gate fails.

## 19. Repository and provenance boundary

This document is a design artifact only.

It does not:

- modify AAA Concept V0;
- freeze a benchmark implementation;
- freeze numerical task-family sizes;
- freeze a statistical threshold;
- implement a generator;
- implement reference controls;
- execute a model;
- establish any empirical AAA result.

The concept, assay, implementation, and empirical result remain separate provenance objects:

```math
\boxed{\text{AAA concept} \neq \text{AAA assay} \neq \text{AAA implementation} \neq \text{AAA empirical result}.}
```

## 20. Design acceptance criterion

This assay design is acceptable for transition into implementation planning only if review agrees that:

1. the only manipulated acquisition-history regularity is the declared `Z`–`Q*` dependency;
2. current-answer advantages are blocked by fresh post-boundary targets and the matching gate;
3. the primary endpoint is the full paired learning curve `Delta C(b)`;
4. D1, D2, and D3 make distinct claims and are not silently merged;
5. the reference-control matrix is mandatory and falsifiable;
6. control mismatch stops scientific interpretation;
7. nulls remain local to the claim layer tested;
8. the design preserves the Concept V0 claim ceiling.

Until that review is complete, AAA-v0 remains **DESIGN ONLY / NOT IMPLEMENTED / UNSCORED**.
