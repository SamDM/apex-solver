# The step-quality ratio ρ, and how to tell when it is lying

Levenberg-Marquardt and Dog Leg steer entirely on

```text
ρ = actual_reduction / predicted_reduction
```

`actual_reduction` is measured by re-evaluating the cost; `predicted_reduction`
is the drop the local quadratic model expected. Everything downstream — whether
a step is accepted, whether λ rises or falls, whether the solver reports
convergence or stalls — reads only ρ.

That makes ρ errors unusually hard to notice. A wrong ρ does not produce a wrong
answer: the solver still descends, because the *step* is computed from the
linear system rather than from ρ. It produces a wrong **damping schedule**. The
two symptoms are

- ρ systematically **too large** → λ collapses to `damping_min` and stays there.
  LM degenerates into undamped Gauss-Newton and converges linearly, for hundreds
  of iterations, never accelerating. "It works but it crawls."
- ρ systematically **too small** → LM concludes every step is poor, ratchets λ
  up on each accepted step, and terminates via `StalledNoProgress` at a bad
  point. "It converges, to the wrong place."

Both regimes were live in this repository. Neither was caught by the benchmark
suite, because both still produce a monotonically decreasing cost.

## Two invariants that make ρ checkable

There is no reference implementation to diff against, but ρ has two properties
that hold for *any* correct implementation, and they are enough to localise a
fault to a single component.

### 1. A linear residual gives ρ = 1 exactly

If `r(x) = A·x − b`, the quadratic model is not an approximation — it *is* the
function. So ρ must equal 1 to machine precision on every iteration, at every λ,
with and without Jacobi scaling. Any deviation is unambiguously an
implementation defect.

This isolates the optimizer's own plumbing: how it pairs the step, gradient and
Hessian, and whether it keeps them in the same coordinate system.

### 2. ρ → 1 as λ → ∞

As λ grows the step length goes to zero, so the quadratic model becomes exact in
the limit and ρ → 1 on *any* problem, linear or not. Concretely: pin λ at 1e8,
switch the tolerances off, and read ρ.

This is the sharper of the two, because it separates "the model is wrong" from
"the problem is hard". A ρ that settles on a constant other than 1 **while the
step norm shrinks by orders of magnitude** cannot be explained by nonlinearity;
it proves the assembled gradient disagrees with the cost it is supposed to
differentiate. And it bisects: vary one thing at a time — factor type, rotation
magnitude, robust loss on or off, gauge fix on or off — until ρ returns to 1.

Do not push λ much past 1e10. By 1e14 the step is ~1e-12, `actual_reduction`
subtracts two nearly equal numbers, and ρ becomes cancellation noise. The same
caveat applies at the other end: once a solve has *reached* its minimum, both
reductions legitimately vanish and ρ is 0/0. Tests that run to convergence
should use a residual that stays bounded away from zero.

Both invariants are pinned in [`tests/lm_step_quality.rs`](../tests/lm_step_quality.rs).

## Faults found this way

Each of these was located by invariant 2 and fixed. They are listed with the
signature that identified them, since the signature is the reusable part.

### The predicted reduction was computed in the wrong coordinate system

*Signature: ρ in the thousands, and negative on rejected steps, but only with
`use_jacobi_scaling(true)`.*

`compute_step_generic` paired the **un-scaled** step with the **scaled**
gradient. The predicted-reduction identity only holds when both live in the
basis the linear system was solved in. Fixed before this document was written
(the predicted reduction is now taken before the inverse scaling is applied);
invariant 1 now guards it.

### SE(3)'s right Jacobian was the left Jacobian

*Signature: ρ settles at ~0.93 instead of 1; the deviation is exactly zero at
zero rotation and grows with the rotation angle.*

`SE3Tangent::right_jacobian` built both diagonal blocks from
`SO3Tangent::new(-theta).right_jacobian()`, which is `Jr(-θ) == Jl(θ)`. The two
agree only at θ = 0. `right_jacobian_inv` inverted that same wrong block, so
`Jr · Jr⁻¹ == I` held throughout and the existing identity tests passed.

The shared Q block was wrong independently of that: its `d` coefficient
multiplied `3` into the wrong factor and dropped a `½`, its last group carried
only `θ̂ρ̂θ̂²` and not the companion `θ̂²ρ̂θ̂`, and the small-angle series had a
sign error on `b`'s θ² term and a factor of 2 on `d`'s.

`log`'s Jacobian is `Jr⁻¹` of its result, so every factor built on
`log`/`right_minus` — `BetweenFactor` and `PriorFactor` on SE(3) — inherited a
gradient that disagreed with the retraction `apply_tangent_step` actually
applies.

**Lesson: `Jr · Jr⁻¹ == I` is not a test of `Jr`.** A consistent inverse of the
wrong matrix satisfies it. Test a Jacobian against its *defining property*
(`exp(τ + δ) ≈ exp(τ) ∘ exp(Jr(τ)·δ)`) by finite differences taken **along the
retraction the optimizer applies** — never against finite differences of the raw
parameter vector, which measures a different derivative.

### Fixed DOFs stayed in the linear system

*Signature: ρ ≈ 0.56 on a gauge-fixed problem, ρ = 1 on the same problem without
the gauge fix.*

`ManifoldVariable::apply_tangent_step` zeroes a fixed DOF's component of the
step, but nothing removed its Jacobian columns from the assembled system. The
solver planned motion along those directions and `compute_predicted_reduction`
charged for it, while the move never happened.

`fix_variable` is how gauge freedom is handled, so this affected **every**
bundle-adjustment problem rather than being a corner case.

### CauchyLoss's ρ(s) did not match its own ρ'(s)

*Signature: ρ = 0.5 exactly, for every Cauchy-weighted residual, at every λ.*

`evaluate` returned `ρ(s) = (δ²/2)·ln(1 + s/δ²)` beside `ρ'(s) = 1/(1 + s/δ²)`,
but the derivative of that ρ is half that ρ'. The optimizer takes a block's cost
from `0.5·ρ(s)` and its gradient from `ρ'(s)·Jᵀr` (through the Triggs
corrector), so the two described different functions.

**Lesson: for a robust kernel, `ρ` and `ρ'` are a contract, not two independent
formulas.** Check `ρ'` against a central difference of `ρ`, and `ρ''` against a
central difference of `ρ'` — a second difference of `ρ` is too ill-conditioned
to be worth asserting on.

## Effect on real problems

Measured on BAL `ladybug-49` (pose + landmark, first camera gauge-fixed, focal
seeded 20% high) and on the `trafalgar-21` self-calibration integration test,
before and after the three fixes above:

| | before | after |
| --- | --- | --- |
| ladybug-49, iterations | 40 | **8** |
| ladybug-49, final RMSE | 1.897 px | **1.265 px** |
| trafalgar-21, final cost | 1.767e4 (stalled at 38 iterations) | **1.370e4** (converged at 67) |

The qualitative change is visible in λ. Before, ρ sat near 0.05–0.1 and λ
climbed from 3.3e-4 past 3e1 over 25 iterations while the cost barely moved.
After, ρ sits at 0.95–1.0 and λ falls monotonically (3.3e-4 → 5.7e-8), which is
what LM is supposed to do on a well-modelled problem.

Pose-graph goldens moved in the same direction: `parking-garage`
6.245107e-1 → 6.245094e-1 and `sphere2500` 2.131994e1 → 2.129065e1, both lower.
The 2D goldens are unchanged to 1e-11, since SE(2) does not go through the
corrected code path.

## A note on Jacobi scaling

`use_jacobi_scaling` **does not change the iterates under the default damping**,
and this surprises people often enough to be worth stating here as well as on
the config field.

With the Marquardt diagonal `D = diag(JᵀJ)`, column scaling cancels out of the
damped system exactly. Writing `J̃ = J·S`, the key identity is
`diag(SᵀHS) = S·diag(H)·S`, so

```text
(J̃ᵀJ̃ + λ·diag(J̃ᵀJ̃))·dx̃ = −J̃ᵀr   ⟺   S·(H + λ·diag(H))·S·dx̃ = −S·Jᵀr
```

and the un-scaled step `S·dx̃` solves the un-scaled damped system. The
trajectories are identical with the flag on and off, down to rounding; only the
conditioning of the matrix handed to the factorisation differs.

The cancellation needs the damping to scale with the columns, so it does *not*
apply to uniform `λ·I` damping (`with_diagonal_bounds(1.0, 1.0)`), where the
flag genuinely changes the iterates.

Three consequences:

- Reaching for this flag to fix slow convergence under the default damping will
  do nothing. The slow convergence has another cause — very likely a ρ that is
  not telling the truth.
- `process_jacobian_generic` caching the column norms from iteration 0 cannot
  matter under the default damping, whatever one thinks of the staleness in
  principle. Recomputing them every iteration was measured on bundle-adjustment
  problems and changed neither the iteration count nor the final cost. Ceres
  fixes its `jacobian_scaling_` at the first iteration the same way.
- **If the flag appears to change the iteration count, suspect the reporting
  before the iterates.** That is exactly how the gradient-units bug was found:
  the convergence test was reading the *scaled* gradient, so turning scaling on
  made `gradient_tolerance` a looser test and solves stopped a step earlier —
  which reads as a conditioning win and is not one. Post-fix the count should
  move only by rounding.

### The `min_diagonal` clamp is not the loophole it looks like

`D_jj = clamp(JᵀJ_jj, min_diagonal, max_diagonal)`, and clamping does not
commute with scaling — `clamp(s²·h) ≠ s²·clamp(h)` — so in principle the clamp
*does* break the cancellation identity above. In practice it does not, and it is
worth writing down why so the exception does not get rediscovered as a bug.

With `s = 1/(1 + ‖c‖)` and `h = ‖c‖²` for a column `c`, the unscaled diagonal
crosses the default `min_diagonal = 1e-6` at `‖c‖ = 1e-3`, and the scaled one at
`‖c‖ ≈ 1.001e-3`. The two saturate at practically the same place, so the window
in which they disagree is 0.1% wide in column norm. Setting
`with_diagonal_bounds(1e-30, 1e30)` — `D = diag(JᵀJ)` exactly, no saturation
anywhere — leaves the on/off difference where it was.

## Outstanding

Found while investigating the above, reproduced, and **not** fixed.

### `Sim3` and `SE23` have the same right-Jacobian bug as SE(3) had

Both build their diagonal blocks from `SO3Tangent::new(-theta).right_jacobian()`
— the left Jacobian. Measured against `Jr`'s defining property at
`|θ| ≈ 0.7 rad`, the maximum absolute error is **7.4e-1** (Sim3) and **4.8e-1**
(SE23) on a finite-difference scale of ~1.

Not fixed here because each has its own `q_matrix` block that would need
verifying alongside the diagonal, and neither manifold is exercised by the
bundle-adjustment or pose-graph paths that motivated this work. The fix is
expected to mirror SE(3)'s.

`SGal3` avoids the whole class by computing its Jacobians as
derivative-by-definition through the crate's own compose/log. `SE2`, `SO2`,
`SO3` and `Rn` are correct.

### Two robust kernels return ρ' inconsistent with their own ρ

Audited every kernel against central differences at
`s ∈ {0.05, 0.25, 0.7, 1.7, 4, 9, 16, 40}`:

- **`AndrewsWaveLoss`** — ρ' disagrees with dρ/ds at 7 of 8 sample points, by a
  *varying* factor (ratios spanned ~0.17 to ~2.99). Not a single missing
  constant like Cauchy's, so it needs its formulas re-derived rather than a
  one-line fix.
- **`TrimmedMeanLoss`** — off by exactly 2× at one sample point.

The same audit flagged ρ'' for `FairLoss` and `TukeyBiweightLoss`, but those
readings came from a second difference of ρ and are not trustworthy; re-check
them by differencing ρ' before acting. ρ'' only shapes the Triggs Hessian
approximation, so it degrades the convergence rate rather than corrupting ρ
outright — ρ' is the one that matters here.

Generalising `cauchy_rho_is_the_antiderivative_of_rho_prime` into a sweep over
every kernel is the natural next step, and would fail today on the two above.

### Robust losses leave ρ above 1 far from the solution

On `trafalgar-21` with Huber, ρ sits at ~1.85 once λ has fallen to its floor.
This is *not* a gradient error: the λ → ∞ invariant holds exactly there
(ρ = 1.000000), so cost and gradient agree. It is the Triggs Hessian
approximation, which drops an indefinite term, under-predicting the reduction
for large Gauss-Newton steps. ρ > 1 means steps do better than predicted, which
is benign — but it does drive λ to `damping_min` and keep it there, so the
robust path effectively runs as Gauss-Newton. Whether that is the right
behaviour for robust bundle adjustment is a design question, not a bug.
