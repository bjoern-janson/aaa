# AAA Concept V0

## 1. Scientific object

**Acquired Adaptive Advantage (AAA)** is the phenomenon in which past experience causally improves an agent's future efficiency at acquiring competence on novel tasks.

Canonical question:

```math
\boxed{\text{Can past experience causally improve an agent's future ability to acquire novel skills?}}
```

The concept is intentionally functional and architecture-independent.

## 2. Core distinction

Three claims must remain separate:

```math
\boxed{\text{what changed} \neq \text{whether it improved} \neq \text{whether it improved future learning}}
```

Examples of observations that do **not** by themselves establish AAA include:

- weights changed;
- memory persisted;
- a parser or planner was written;
- a tool was acquired;
- a representation was invented;
- a prompt, policy, retrieval rule, or code path changed;
- performance improved on the acquisition environments;
- future answers improved through retained knowledge.

AAA requires evidence that past experience changed **subsequent acquisition dynamics** in a way that improves learning on later novel tasks.

## 3. Functional adaptation dynamics

For an agent `A`, represent current adaptation dynamics by the functional object

```math
\mathcal U_A:(\tau,e,b)\mapsto\Delta\mathrm{competence},
```

where:

- `tau` is a task or environment;
- `e` is newly obtained experience;
- `b` is a frozen learning/resource budget;
- the output is the resulting change in competence.

`mathcal U_A` is **not** assumed to correspond to a physically separable module. It may be implemented by weights, memory, context, retrieval, programs, tools, representations, scaffolds, or mixtures of these.

Therefore AAA does not require a privileged decomposition such as `A=(K,U)` in the implementation.

## 4. Counterfactual estimand

Let:

- `H` be an acquisition history;
- `A_H` be the history-exposed terminal agent;
- `A_C` be an appropriately matched counterfactual;
- `D_novel` be a prospectively defined family of later novel tasks.

The canonical conceptual estimand is

```math
\boxed{
\Delta_{\mathrm{AAA}}
=
\mathbb E_{\tau\sim\mathcal D_{\mathrm{novel}}}
\left[
\operatorname{LearnEff}(A_H,\tau)
-
\operatorname{LearnEff}(A_C,\tau)
\right]
}
```

A positive value supports the bounded claim that the history-exposed condition has an acquired adaptive advantage under the frozen assay.

It does not by itself establish broad general intelligence, unrestricted recursive self-improvement, transfer to arbitrary domains, or corrigibility.

## 5. Temporal boundary

The causal effect must cross a temporal boundary between acquisition history and later novel learning:

```text
        acquisition history H
                 |
                 v
        terminal agent state
                 |
        -------------------
          temporal boundary
        -------------------
                 |
                 v
        novel learning episodes
                 |
                 v
        future learning curves
```

The target claim has the form

```math
H_{\mathrm{past}}
\longrightarrow
\Delta(\text{future learning dynamics})
\longrightarrow
\text{future acquisition}.
```

By contrast,

```math
H_{\mathrm{past}}\longrightarrow\text{better terminal answers}
```

is insufficient.

## 6. Matching principle

Canonical experimental principle:

```math
\boxed{\text{Match current observable competence; compare subsequent learnability.}}
```

A frozen diagnostic surface `T` may support a statement such as

```math
A_H\equiv_{\mathcal T}A_C.
```

This notation means only that the agents are matched on the declared diagnostic surface under the declared tolerance.

It must **not** be interpreted as:

```math
A_H=A_C,
```

hidden-state identity, global behavioral equivalence, or proof that all retained knowledge is equal.

The scientific role of matching is to reduce obvious alternative explanations such as the history-exposed agent simply beginning the future phase with greater directly usable competence on the measured surface.

The strength of any causal interpretation is bounded by the strength of the matching contract.

## 7. Learning-curve object

Let

```math
C_A(\tau,b)
```

denote competence achieved by agent `A` on future task `tau` after learning budget `b`.

The geometry of

```math
C_{A_H}(\tau,b)-C_{A_C}(\tau,b)
```

may reveal distinct forms of acquired adaptive advantage, including:

- lower evidence or interaction cost to reach a fixed competence;
- faster early acquisition;
- delayed but stronger acquisition;
- a higher attainable competence under the same budget;
- an advantage that appears only after exploration;
- an advantage that later reverses.

A possible transfer summary is

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
\right].
```

This expression is illustrative at V0. The exact `LearnEff` measure, weighting `w`, budget `B`, competence metric, and statistical procedure remain assay-design questions.

## 8. Representation shift

Positive future transfer on near-duplicates can be explained by sophisticated memorization or narrow reuse.

A stronger AAA assay therefore requires prospectively defined representation shift. Conceptually distinguish:

```text
D1: surface transfer
D2: structural transfer across substantial representational change
D3: anti-transfer / invalidation
```

Evidence on `D2` bears on whether the acquired advantage captures structure deeper than the acquisition surface.

A failure on `D2` localizes the result; it does not retroactively erase a valid narrower `D1` result.

## 9. Anti-transfer and corrective recoverability

An acquired adaptation can be useful in one regime and harmful in another.

Therefore positive transfer alone does not establish that the acquired advantage remains appropriately revisable.

An anti-transfer phase should test whether the agent can:

```text
reuse
  -> encounter mismatch
  -> detect invalidity
  -> localize the failure
  -> weaken, scope, replace, or revise the acquired adaptation
  -> recover useful future learning
```

Relevant later measures may include detection latency, revision latency, post-revision recovery, and recurrence of the invalid adaptation.

Corrective recoverability is a distinct property from positive adaptive advantage.

## 10. Non-scalar profile

At the conceptual level, keep at least three quantities distinct:

```math
G_{\mathrm{transfer}}
```

future acquisition gain;

```math
G_{\mathrm{abstraction}}
```

survival of the advantage across representation shift;

```math
C_{\mathrm{recovery}}
```

capacity to revise the acquired advantage when later evidence invalidates it.

V0 does not authorize collapsing these into a single scalar score.

## 11. Relationship to ARC

AAA arose from reasoning about a possible progression beyond ARC-AGI-3:

```text
ARC-1: infer
ARC-2: compose
ARC-3: explore
ARC-4?: improve
```

with the proposed invariant:

```math
\boxed{\text{ARC progressively expands the causal scope of adaptation over future skill acquisition.}}
```

The proposed ARC-4 phase transition is:

```math
\boxed{\text{learning changes not merely competence, but subsequent learnability.}}
```

This is motivation, not dependency. AAA remains well-defined even if no official ARC-AGI-4 adopts this framing.

## 12. Claim ceiling and frozen boundary

V0 establishes a **proposed scientific object**, not an empirical result.

Frozen at the concept level:

- the scientific question;
- the functional, architecture-independent target;
- the temporal boundary;
- the counterfactual nature of the estimand;
- the matching principle and its limited interpretation;
- the distinction between positive transfer, structural transfer, and corrective recoverability;
- the requirement that stronger claims not be inferred from weaker observations.

Not frozen at V0:

- a benchmark implementation;
- task families;
- a diagnostic matching surface;
- a model or architecture;
- a `LearnEff` metric;
- a budget definition;
- a competence metric;
- statistical thresholds;
- a causal identification procedure;
- a correction assay implementation.

**No empirical AAA result has yet been established.**
