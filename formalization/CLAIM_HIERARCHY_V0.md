# AAA Claim Hierarchy V0

This record freezes the claim hierarchy for Acquired Adaptive Advantage (AAA).

The purpose is to prevent weaker observations from silently authorizing stronger conclusions.

## C1 — Acquisition

**Claim:** experience changed subsequent learning dynamics under the declared assay.

Evidence must show that the history-exposed condition and matched counterfactual differ in later acquisition behavior.

This may be true even if the change is neutral or harmful.

Therefore:

```math
C_1\nRightarrow C_2.
```

## C2 — Adaptive Advantage

**Claim:** the changed learning dynamics improve acquisition on later novel tasks under the declared resource controls.

Canonical conceptual quantity:

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

A positive bounded result supports acquired adaptive advantage for the tested task distribution and assay.

It does not establish that the advantage survives substantial representation shift.

Therefore:

```math
C_2\nRightarrow C_3.
```

## C3 — Transferable Advantage

**Claim:** the acquired adaptive advantage survives a prospectively defined representation shift that blocks trivial surface reuse while preserving relevant deeper structure.

This supports a stronger abstraction/generalization claim than near-surface transfer alone.

It does not establish that the adaptation remains appropriately revisable when its validity conditions fail.

Therefore:

```math
C_3\nRightarrow C_4.
```

## C4 — Correctable Advantage

**Claim:** when later evidence invalidates or reverses the conditions supporting the acquired adaptation, the agent can appropriately weaken, scope, replace, or revise that adaptation and recover useful learning.

C4 concerns corrective recoverability of the acquired advantage.

It should be evaluated separately from raw transfer magnitude.

## Frozen implication boundary

The hierarchy is:

```math
\boxed{
C_1\nRightarrow C_2\nRightarrow C_3\nRightarrow C_4
}
```

where the chained notation means that **none of the forward implications is licensed without additional evidence**.

Equivalently:

```text
changed acquisition dynamics
    != positive adaptive advantage
    != transferable adaptive advantage
    != correctable adaptive advantage
```

## Negative and null results

Null or negative results are local to the layer tested.

Examples:

- failure to establish `C2` does not imply that acquisition dynamics did not change (`C1` may still hold);
- failure to establish `C3` does not erase a valid bounded `C2` result;
- failure to establish `C4` does not erase a valid `C2` or `C3` result, but it blocks stronger claims about corrective recoverability;
- a negative transfer result may itself be informative about the scope or brittleness of an acquired adaptation.

Do not convert a null at one layer into failure of the entire conceptual architecture.

## Matching claim ceiling

If two agents satisfy

```math
A_H\equiv_{\mathcal T}A_C
```

on a frozen diagnostic surface `T`, the authorized claim is only equivalence on that declared surface and tolerance.

It does not imply:

- hidden-state identity;
- equality of all retained knowledge;
- global behavioral equivalence;
- equality of future learning dynamics.

Indeed, the AAA assay is specifically interested in the possibility that agents matched on current observable competence can differ in subsequent learnability.

## Prohibited claim inflation

The following observations are insufficient by themselves to establish any stronger AAA claim:

```text
self-modification
persistent memory
weight change
tool creation
code generation
representation invention
prompt or policy revision
better terminal performance
better performance on acquisition environments
```

The scientifically relevant question is whether past experience causally changes future acquisition dynamics, whether that change is advantageous, whether it transfers, and whether it remains correctable.

## Current repository status

At V0 this hierarchy is conceptual only.

No implemented assay, benchmark result, empirical `Delta_AAA`, transfer result, representation-shift result, or corrective-recoverability result is claimed.
