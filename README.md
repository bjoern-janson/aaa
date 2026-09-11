# Acquired Adaptive Advantage (AAA)

> **Scientific question:** Can past experience causally improve an agent's future ability to acquire novel skills?

AAA is a proposed, architecture-independent capability class for measuring whether learning changes **subsequent learnability**.

The central distinction is:

```text
what changed != whether it improved != whether it improved future learning
```

AAA concerns the third claim.

## Core phenomenon

Let `A_H` be an agent after an acquisition history `H`, and let `A_C` be an appropriately matched counterfactual. For novel future tasks `tau ~ D_novel`, define the primary conceptual estimand

```math
\Delta_{\mathrm{AAA}}
=
\mathbb E_{\tau\sim\mathcal D_{\mathrm{novel}}}
\left[
\operatorname{LearnEff}(A_H,\tau)
-
\operatorname{LearnEff}(A_C,\tau)
\right].
```

`Delta_AAA > 0` means:

> Past experience caused the system to become better at converting subsequent experience into competence on genuinely novel tasks.

This is stronger than showing that the system changed, retained memory, acquired tools, improved on training environments, or produced better future answers.

## Experimental principle

> **Match current observable competence; compare subsequent learnability.**

A diagnostic surface may establish only bounded observable equivalence. It does **not** establish hidden-state identity or global behavioral identity.

The causal effect of interest must cross a temporal boundary:

```text
PAST: acquisition history
          |
          v
------ terminal boundary ------
          |
          v
FUTURE: novel learning episodes
```

If history produces only better answers at the boundary, AAA has not been established.

## Architecture independence

AAA does not define the changed object as weights, memory, code, tools, retrieval, context, representations, prompts, or any other implementation detail.

Instead, an agent's adaptation dynamics are treated functionally:

```math
\mathcal U_A:(\tau,e,b)\mapsto\Delta\text{competence},
```

where `tau` is a task, `e` is experience, and `b` is a learning/resource budget.

The benchmark-relevant question is whether history changes those future dynamics in a causally advantageous way.

## Learning curves

Let

```math
C_A(\tau,b)
```

be competence reached by agent `A` on task `tau` after learning budget `b`.

AAA is naturally evaluated through differences in future learning curves rather than a binary learned/not-learned outcome. A later assay may operationalize transfer gain using a quantity such as

```math
G_{\mathrm{transfer}}
=
\mathbb E_{\tau}
\left[
\int_0^B
w(b)
\left(
C_{A_H}(\tau,b)-C_{A_C}(\tau,b)
\right)db
\right],
```

but the exact definition of `LearnEff`, budget accounting, weighting, and task construction are **not frozen by this conceptual record**.

## Generalization and correction

A serious AAA assay should distinguish at least:

- **surface transfer** — the advantage persists on superficially similar novel tasks;
- **structural transfer** — the advantage survives substantial representational change while deeper structure remains relevant;
- **anti-transfer** — prior adaptive structure becomes misleading, testing whether later evidence can appropriately weaken or revise it.

Positive transfer and corrective recoverability are distinct scientific properties and should not be silently collapsed into one scalar.

## Conceptual lineage

AAA was motivated by a progression from static task induction toward interactive adaptation, including discussion of a possible successor to ARC-AGI-3. The AAA concept does **not** depend on an official ARC-AGI-4 having this form, and it is intended to stand independently as a measurable capability class.

## Current project status

**Concept V0 remains frozen.** The repository now also contains an **AAA-v0 assay kernel**, but that kernel is only software-validated apparatus. It is not yet a scientifically validated benchmark and it has not produced an empirical AAA result.

Current gate state:

```text
AAA Concept V0             FROZEN
AAA-v0 assay design        FROZEN
AAA-v0 assay kernel        IMPLEMENTED / SOFTWARE-VALIDATED ONLY
scientific preregistration NOT FROZEN
confirmatory controls      NOT RUN scientifically
frontier-model execution   NOT RUN
AAA evidence               NONE
```

The frozen V0 formalization files preserve their historical freeze-time status language. Live implementation state is recorded separately in [`IMPLEMENTATION_STATE.json`](IMPLEMENTATION_STATE.json).

This repository does **not** claim:

- a scientifically validated AAA benchmark;
- a validated matching procedure for frontier models;
- a frozen scientific `LearnEff` operationalization;
- a completed causal assay on a frontier model;
- an empirical demonstration of acquired adaptive advantage;
- an empirical demonstration of transferable or correctable adaptive advantage.

No empirical AAA result has yet been established.

See:

- [`formalization/AAA_CONCEPT_V0.md`](formalization/AAA_CONCEPT_V0.md)
- [`formalization/CLAIM_HIERARCHY_V0.md`](formalization/CLAIM_HIERARCHY_V0.md)
- [`docs/superpowers/specs/2026-09-11-aaa-v0-assay-design.md`](docs/superpowers/specs/2026-09-11-aaa-v0-assay-design.md)
- [`IMPLEMENTATION_STATE.json`](IMPLEMENTATION_STATE.json)
