# Proposal: measure the gradient convergence test in the max norm

**Status:** proposal, not implemented. Behaviour change — needs a retuned
default and a goldens re-pin, so it is written up rather than slipped into a
bug-fix branch.

**Related:** [`step_quality.md`](step_quality.md), for the trust-region ratio —
a separate quantity that was also being measured in the wrong units.

## Summary

`gradient_tolerance` is currently tested as

```text
‖Jᵀr‖₂ < gradient_tolerance
```

This proposes the **max norm** instead — the largest single gradient component.
It is the more interpretable quantity and it is what Ceres uses.

**But the measurements below did not support the argument this proposal was
started to make**, and they change the priority. Read the evidence section
before acting on this: the honest conclusion is that the norm choice is a
second-order issue, and the *default value* deserves scrutiny first.

## The original argument

If every component of the gradient is around `ε`, then

```text
‖g‖₂ = sqrt(Σⱼ gⱼ²) ≈ ε·√N        ‖g‖∞ = max |gⱼ| ≈ ε
```

so under L2 a 24,000-DOF problem should report a norm ~155× larger than a
1-DOF problem that is equally converged per parameter, and a fixed
`gradient_tolerance` should silently become stricter as the problem grows.

The max norm says what a user setting this tolerance usually means — "no single
parameter still has a slope worth chasing" — and that statement should not
change when you add cameras to the rig.

## Evidence

Measured at the last iteration of each solve, `use_jacobi_scaling` off,
via temporary instrumentation in `compute_step_generic`.

| dataset | DOF | √N | final ‖g‖₂ | final ‖g‖∞ | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ring (SE2) | 1 302 | 36 | 4.47e-1 | 6.52e-2 | **6.86** |
| M3500 (SE2) | 10 500 | 102 | 1.18e-1 | 4.02e-2 | **2.94** |
| parking-garage (SE3) | 9 966 | 100 | 1.99e-3 | 3.78e-4 | **5.26** |
| sphere2500 (SE3) | 15 000 | 122 | 1.01e-1 | 3.32e-2 | **3.03** |
| ladybug-49 (BA, Schur) | 23 769 | 154 | 5.57e2 | 2.04e2 | **2.73** |
| trafalgar-21 (BA, Schur) | 34 134 | 185 | 1.48e3 | 5.26e2 | **2.81** |

Two things fall out, and both cut against the argument above.

**1. The √N effect does not materialise on real problems.** Ratios land between
2.7 and 6.9, against √N values of 36 to 185. Real gradients are *concentrated* —
a handful of components dominate — so the L2 norm behaves far more like a max
norm than like `ε·√N`. Worse for the argument, the ratio does not track problem
size at all: the **smallest** problem (ring, N = 1 302) has the **largest**
ratio, and the largest has nearly the smallest. What drives the ratio is the
shape of the gradient, not the count of parameters.

`ε·√N` is the *uniform-gradient* case, and it is an upper bound rather than a
prediction. A synthetic fixture with every component equal by construction does
follow it exactly (see the next section), which is worth knowing before reading
too much into a benchmark built from one.

So the practical effect of switching norms is a factor of roughly 3–7, not 100.
The remaining defensible complaint is narrower: the ratio spans 2.4× across
these six problems with no usable predictor, so `gradient_tolerance` still
cannot be reasoned about a priori — just far less dramatically than the theory
suggested.

**2. The criterion never fires anyway.** Every final ‖g‖₂ above is between
1e-3 and 1e3, against a default `gradient_tolerance` of **1e-10**. All six
solves terminated on the cost or parameter tolerance. Pushing parking-garage to
absurd settings does not change this:

| parking-garage | final ‖g‖₂ | final ‖g‖∞ |
| --- | ---: | ---: |
| `cost_tol=1e-4, param_tol=1e-4` | 1.99e-3 | 3.78e-4 |
| `cost_tol=1e-6, param_tol=1e-8` (library defaults) | 1.06e-3 | 3.26e-4 |
| `cost_tol=1e-12, param_tol=1e-14` | 1.05e-8 | 2.53e-9 |

Even at 1e-12/1e-14 the gradient norm bottoms out ~100× above the threshold.
On real problems the gradient criterion is effectively **unreachable**, and a
3–7× change of norm does not make it reachable.

That does not make it harmless — it fires readily on small or synthetic
problems, which is where the scaled-units bug fixed alongside this document
actually bit — but it does mean this proposal is a correctness-and-clarity
change, not a performance or convergence one. It should be scheduled as such.

## What actually sets the floor: the Jacobian's scale, not N

The "unreachable" finding above invites the obvious follow-up — unreachable
because of what? A controlled fixture separates the two candidates. Residual
`r_i(x) = K·(x_i² − a_i)` with `a_i` non-square, so the root is irrational and
the solve bottoms out on floating point rather than landing on the answer. `K`
scales the Jacobian (`J_ii = 2·K·x_i`) without moving the solution, so the
Jacobian's scale and the parameter count vary independently. Tolerances all
zero, so the last reported norm is the achievable floor and not a stopping
choice.

| Jacobian scale `K` (at D = 20) | final ‖g‖₂ | ‖g‖₂ / K² |
| ---: | ---: | ---: |
| 1e0 | 8.7403e-15 | 8.740e-15 |
| 1e1 | 8.7403e-13 | 8.740e-15 |
| 1e2 | 8.7403e-11 | 8.740e-15 |
| 1e3 | 8.7403e-9 | 8.740e-15 |
| 1e4 | 8.7403e-7 | 8.740e-15 |

| parameters `D` (at K = 1e3) | final ‖g‖₂ | ‖g‖₂ / √D |
| ---: | ---: | ---: |
| 2 | 1.8567e-9 | 1.313e-9 |
| 20 | 8.7403e-9 | 1.954e-9 |
| 200 | 2.6287e-8 | 1.859e-9 |
| 2 000 | 8.3377e-8 | 1.864e-9 |

**The floor goes as `K²` in the Jacobian's scale and only as `√D` in the size.**
`‖g‖₂/K²` is constant to four significant figures across four decades of `K`;
`‖g‖₂/√D` is flat across three decades of `D`. Four decades of Jacobian scale
move the floor by eight; three decades of parameter count move it by one and a
half.

So size is not irrelevant — the `√D` term is real, and this uniform fixture is
exactly the case where it shows — but it is dominated. An utterly ordinary
`K = 1e3` (a focal length in pixels) raises the floor by `1e6`; matching that
through size alone would take `1e12` more parameters.

That reframes the earlier finding. `gradient_tolerance` is not unreachable
because a problem is *large*; it is unreachable because a problem's Jacobian is
not `O(1)`, which is true of essentially every calibration or
physical-units problem. The default of `1e-10` is attainable only where
`K ≲ 1`.

A downstream project reported the corroborating case, on problems an order of
magnitude *smaller* than anything in the table above:

Reported downstream, not measured here:

| source | DOF | final ‖g‖₂ | ended on gradient tol. |
| --- | ---: | ---: | ---: |
| downstream BA suite, 33 solves | 104–440 | 2.4e-9 – 1.3e2 | 0 of 33 |

Their minimum of 2.36e-9 is what the `K²` law predicts for `K` between 3e2 and
1e3 — the range a focal length in pixels lands in. Small problems, same wall.

## What Ceres does, and where that does not settle it

Ceres tests

```text
maxᵢ |x − Π ⊞(x, −g(x))|  ≤  gradient_tolerance
```

(`nnls_solving.rst`; `TrustRegionMinimizer::GradientToleranceReached`) — max
norm, on the **unscaled** gradient, mapped through the manifold's Plus operator
and projected onto the bounds. Default `1e-10`, same number as apex's.

Three caveats against treating that as the answer:

1. **Ceres has the same unit-dependence.** Its tolerance is absolute too. The
   max norm addresses size-dependence, not units. Nothing in either library
   makes `gradient_tolerance` portable across problems whose parameters mean
   different things.

2. **Its own guidance is dimensionally incoherent.** `solver.h` says
   `gradient_tolerance` "should typically be 1e-4 * function_tolerance". But
   `function_tolerance` is a dimensionless *ratio* (`|Δcost| ≤ ftol·cost`)
   while `gradient_tolerance` carries units of cost per parameter. Multiplying
   one by the other is not a relationship. Do not copy it.

3. **Ceres tried anchoring to the initial gradient and abandoned it.** Through
   1.8 the test was `‖g(x)‖∞ < gradient_tolerance · ‖g(x₀)‖∞`, i.e. scaled by
   the gradient at the *starting point*; 1.9.0 dropped the `‖g(x₀)‖∞` factor
   (`version_history.rst`). The entry does not say why. Worth knowing before
   reaching for option B below as an obvious improvement.

The `Π ⊞(x, −g)` construction is deliberately **not** part of this proposal. It
buys projection onto bounds and lifting through the manifold retraction. apex
does not enforce bounds in the optimizer at all: `Variable::bounds` is written
by `set_bounds` and read only by `Rn::update_variable`, which is not on the path
the optimizers take — `apply_tangent_step` handles `fixed_indices` and nothing
else. So the projection would be a no-op wrapped in machinery, and for an
unconstrained Euclidean problem the whole expression reduces to `|g|∞`. If
bounds are ever wired into the optimizer, revisit it then.

## Two senses of "relative"

The word does double duty in this area and the two meanings behave very
differently, so this document keeps them apart:

- **Relative to a moving reference** — the *current* cost, the *current* `‖x‖`.
  Re-anchored every iteration. This is what `cost_tolerance` and
  `parameter_tolerance` do, in apex and in Ceres alike, and it is
  uncontroversial: "am I still making progress relative to where I am now" is
  always a meaningful question.
- **Relative to a fixed starting reference** — `‖g(x₀)‖`, captured once at the
  seed. This is option B below, and the thing Ceres removed in 1.9.0.

The second is the weaker of the two: anchoring to the seed makes the stopping
point depend on where the solve started, so the same problem from two
initialisations converges to different places, and a solve seeded near the
solution has `g₀ ≈ 0` and can never clear the bar. Those look like the reasons
it did not survive, though Ceres does not say.

**apex's gradient test has never been relative in either sense.**
`check_convergence` has tested `‖g‖ < gradient_tolerance` outright since the
criterion was introduced (`5b6b97a`), the line is byte-identical across every
revision of the file since, and no commit in this repository has ever referenced
an initial or reference gradient. So there is nothing here to undo: option B
would be an addition, and Ceres' experience is evidence against making it.

## Options considered

**A. Max norm** — `‖Jᵀr‖∞ < gradient_tolerance`.
More interpretable, matches Ceres, size-independent in principle. Measured
effect on these problems: 3–7×. Needs a new default and a goldens check.

**B. Anchor to the initial gradient** — `‖g‖ ≤ gtol · ‖g₀‖`.
The only option that makes the tolerance genuinely dimensionless. Two real
costs: meaningless when the solve starts near the solution (`g₀ ≈ 0`), and the
stopping point becomes seed-dependent, so the same problem from two
initialisations converges to different places. Ceres shipped this and moved away
from it.

**C. RMS** — `‖g‖₂ / √N < gradient_tolerance`.
Size-independent like A. But it lets one badly-converged parameter hide behind
24,000 well-converged ones, which is the failure mode the criterion exists to
catch. Strictly worse than A for the same effort — and the evidence above shows
the `√N` it corrects for is not the effect actually present.

**D. Recalibrate the default instead.** Leave the norm alone; set
`gradient_tolerance` to a value that can actually be reached on the problems
this library targets, or document that it is a small-problem safety net.
Addresses the finding that actually showed up in the measurements.

**E. Status quo, documented.** Say in the config docs that
`gradient_tolerance` is an L2 threshold, roughly 3–7× the per-parameter slope it
reads as, and that its reachable floor scales as `K²` in the Jacobian's column
scale — so at the default it is inert on any problem whose parameters are not in
`O(1)` units, regardless of size.

## Recommendation

**A and D together, at low priority.** A is the right shape for the criterion
and cheap; D is what the evidence actually calls for. Doing A alone would make
the number more meaningful while leaving it just as unreachable.

The default must be derived, not copied. Ceres's `1e-10` is a max-norm default
and apex's `1e-10` is an L2 default, so their agreeing is a coincidence rather
than compatibility — and per the tables above, neither is attainable on a real
pose graph.

Deriving it will not rescue it either, and this is the strongest argument in the
document. The threshold's units are the *caller's*, and the reachable floor goes
as `K²`, so whatever number is chosen is wrong by `K²` for somebody — a factor
of `1e6` for an ordinary focal length in pixels. **No single default can work.**
That points at documentation and a *derived* default, or at the criterion being
off unless explicitly configured, rather than at a better constant.

Worth pairing with either: **report which parameter index carries the max**, at
debug level. Free once the max norm is computed, and it turns "the solver did
not converge" into "variable 4172, DOF 2 is still moving", which is the question
anyone actually has at that point.

## Migration

1. Report `‖g‖∞` alongside the L2 value at debug level for one release, so real
   problems can be surveyed before any threshold moves. Report the largest
   column norm of `J` next to it: per the `K²` result above, that is what tells
   a caller whether the criterion is reachable *for their problem* before they
   try to tune it — more useful than either norm on its own.
2. Switch the test; set the new default from that survey, not from Ceres.
3. Re-check `tests/golden_values.rs`. All four goldens terminate on the cost
   tolerance today, so they will likely not move — but confirm rather than
   assume.
4. Keep reporting the L2 value in `ConvergenceInfo` too, or repurposing
   `final_gradient_norm` will silently change what downstream logs mean. That is
   the same class of breakage this line of work started from.

## What this does not fix

`gradient_tolerance` stays unit-dependent under every option except B. The
practical guidance, unchanged by this proposal: **`cost_tolerance` and
`parameter_tolerance` are the portable criteria.** Both are relative to a moving
reference in apex and in Ceres, both are dimensionless, and both mean the same
thing on every problem
— and per the evidence above, both are what actually terminates every real solve
in this repository. `gradient_tolerance` is best set conservatively and treated
as a safety net rather than the criterion a solve is expected to stop on.
