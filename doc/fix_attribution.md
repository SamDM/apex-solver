# Which fix bought which gain

`fix/lm-jacobi-scaling-step-quality` lands four independent correctness fixes.
The aggregate effect is recorded in [`step_quality.md`](step_quality.md); this
is the per-fix breakdown. Eleven cases were run at five checkouts — `main`, then
cumulatively after each `fix:` commit in branch order — so each column reads
against the one to its left.

RMSE is measured **loss-independently**: each case builds an L2 twin of the
problem and evaluates it at the solved parameters, so
`RMSE = sqrt(2·cost_L2 / residual_dim)`. Without that, the Cauchy fix would
appear to change RMSE just by rescaling ρ. Pose-graph RMSE is the per-scalar RMS
of the unweighted SE(2)/SE(3) log residual; BA RMSE is pixels (matches the
repo's `sqrt(cost/num_obs)`).

## Results

Iterations · final RMSE, cumulative:

| case | baseline | +SE3 Jr | +fixedDOF | +Cauchy | +gradnorm |
| --- | --- | --- | --- | --- | --- |
| M3500 LM/L2 | 28 · 7.5570e-2 | 28 · 7.5570e-2 | 17 · 1.3591e-2 | 17 · 1.3591e-2 | 17 · 1.3591e-2 |
| parking-garage LM/L2 | 21 · 1.75869e-2 | 21 · 1.76771e-2 | 9 · 5.76025e-3 | 9 · 5.76025e-3 | 9 · 5.76025e-3 |
| sphere2500 LM/L2 | 29 · 1.32894e-1 | 30 · 1.34268e-1 | 12 · 3.78682e-2 | 12 · 3.78682e-2 | 12 · 3.78682e-2 |
| parking-garage, gauge-free | 7 · 5.759734e-3 | 7 · 5.759728e-3 | 7 · 5.759728e-3 | 7 · 5.759728e-3 | 7 · 5.759728e-3 |
| M3500 LM/Cauchy | 21 · 1.46501e-1 | 21 · 1.46501e-1 | 151 (cap) · 1.36976e-2 | 12 · 1.359127e-2 | 12 · 1.359127e-2 |
| parking-garage LM/Cauchy | 16 · 2.27615e-2 | 16 · 2.28715e-2 | 15 · 5.766842e-3 | 9 · 5.760253e-3 | 9 · 5.760253e-3 |
| parking-garage DogLeg | 17 · 1.87803e-2 | 16 · 1.88011e-2 | 8 · 5.760209e-3 | 8 · 5.760209e-3 | 8 · 5.760209e-3 |
| pg LM + Jacobi, loose gtol | 21 · 1.75869e-2 | 21 · 1.76771e-2 | 4 · 5.767245e-3 | 4 · 5.767245e-3 | 8 · 5.760475e-3 |
| ⤷ same, scaling off (control) | 21 · 1.75869e-2 | 21 · 1.76771e-2 | 8 · 5.760475e-3 | 8 · 5.760475e-3 | 8 · 5.760475e-3 |
| ladybug-49 BA/Huber | 43 · 0.768965 px | 43 · 0.768965 px | 74 · 0.751462 px | 74 · 0.751462 px | 74 · 0.751462 px |
| trafalgar-21 BA/Huber | 38 · 1.087064 px | 38 · 1.087064 px | 67 · 1.010688 px | 67 · 1.010688 px | 67 · 1.010688 px |

In the first two columns every case except the gauge-free control and
`ladybug-49` terminates on `ParameterTolerance` — the stall signature. From
`+fixedDOF` on, the L2 and BA cases terminate on `CostTolerance` instead; the
two `loosegrad` rows terminate on the gradient tolerance by construction, and
the Cauchy rows only join the cost-tolerance group once `+Cauchy` lands.

## Per-fix

**1. SE(3) right Jacobian (`356eed5`)** — small, and only visible once you
remove the gauge fix. 2D and BA are bit-identical (neither path calls
`SE3Tangent::right_jacobian`). On the gauge-free 3D control it's a clean but
tiny win: 7 iterations either way, cost 6.245107e-1 → 6.245094e-1, RMSE
5.759734e-3 → 5.759728e-3 — exactly the golden shift the commit claims. On the
gauge-fixed 3D cases it reads as neutral-to-slightly-worse (sphere2500 29→30
iters, RMSE +1.0%; parking-garage RMSE +0.5%). That isn't a regression from this
fix; the fixed-DOF bug is still present at this commit and dominates, so
correcting the Jacobian just re-steers a solve that is being mis-scored anyway.

**2. Fixed DOFs dropped from the linear system (`d1012cc`)** — by far the
largest effect, and it's the one that makes fix 1 pay off. Every gauge-fixed
case moves; the gauge-free control is bit-identical, which is the clean negative
control. Pose graphs get both faster and better: M3500 28→17 iters with RMSE
−82%, parking-garage 21→9 with −67%, sphere2500 30→12 with −72%. BA goes the
other way on iterations and still improves quality — ladybug 43→74 iters for
−2.3% RMSE, trafalgar 38→67 for −7.0% (cost 1.767e4 → 1.370e4, matching the note
in the integration test). The extra iterations are the solver no longer giving
up early: baseline trafalgar stopped on `ParameterTolerance`, post-fix it runs to
`CostTolerance`.

> The ladybug figures here are **not** comparable to the ones in
> [`step_quality.md`](step_quality.md) ("40 → 8 iterations, 1.897 → 1.265 px").
> That is a different problem — pose + landmark only, focal seeded 20% high.
> This one is the self-calibration setup (pose, landmark *and* intrinsics,
> Huber, sparse Schur), which starts from a better point and has a harder gauge.

**3. `CauchyLoss` ρ (`1f2aedd`)** — perfectly isolated, large on iterations,
negligible on RMSE. Only the two Cauchy cases change; all nine others are
byte-identical. M3500/Cauchy goes from hitting the 151-iteration cap to
converging in 12; parking-garage/Cauchy 15→9. RMSE barely moves (−0.8% and
−0.11%) because both versions were reaching roughly the same minimum — the
ρ = 0.5 signal was costing iterations, not accuracy.

**4. Gradient norm in problem units (`33812fd`)** — no effect unless Jacobi
scaling is on *and* the gradient test binds. Nine of eleven cases are unchanged,
including all reported gradient norms (nothing to unscale). Dog Leg, which
enables Jacobi scaling by default, changes only its reported norm — 9.607e-4 →
5.834e-3, a 6.07× correction — with an identical trajectory, because
`gradient_tolerance=1e-12` is unreachable here anyway. To make it bind at all,
the suite carries a case with scaling on and `gradient_tolerance=1e-2`: there it
goes 4 → 8 iterations, RMSE 5.767245e-3 → 5.760475e-3, and now agrees with the
scaling-off control to 1e-12 in final cost. That is exactly the "reads as a
conditioning win and is not one" claim in the changelog, reproduced.
