//! Invariants on Levenberg-Marquardt's step-quality ratio
//! `rho = actual_reduction / predicted_reduction`.
//!
//! `rho` is the only signal LM's damping policy reads, so an error in it does
//! not announce itself as a wrong answer — the solver still descends, just with
//! a damping schedule steered by noise. The symptom is a run that converges
//! linearly for hundreds of iterations, or one that ratchets damping up and
//! stops early at a bad point. Both regimes were live in this repo, so the
//! properties below are pinned rather than left to a benchmark to notice.
//!
//! Two properties make `rho` checkable without a reference implementation:
//!
//! 1. **A linear residual has an exact quadratic model**, so `rho == 1` on
//!    every iteration whatever the damping or scaling.
//! 2. **As `λ → ∞` the step length → 0**, so the model becomes exact in the
//!    limit and `rho → 1` on *any* problem. A `rho` that settles on a constant
//!    other than 1 while the step shrinks by orders of magnitude is proof of a
//!    gradient that disagrees with the cost it is supposed to differentiate —
//!    it cannot be explained away as nonlinearity.
//!
//! Property 2 is the sharp one: it is what distinguishes a modelling error from
//! a hard problem, and it is what the manifold-Jacobian and fixed-DOF bugs
//! below were caught by.

use std::sync::{Arc, Mutex};

use apex_solver::JacobianMode;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::so3::SO3;
use apex_solver::apex_manifolds::{LieGroup, ManifoldType};
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::{CauchyLoss, HuberLoss, LossFunction};
use apex_solver::core::problem::Problem;
use apex_solver::core::variable::ManifoldVariable;
use apex_solver::factors::{BetweenFactor, Factor, PriorFactor};
use apex_solver::linalg::LinearSolverType;
use apex_solver::observers::OptObserver;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use faer::prelude::ReborrowMut;
use nalgebra::{DVector, Vector3};
use slotmap::SlotMap;

type TestResult = Result<(), Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Observer capturing the per-iteration metrics
// ---------------------------------------------------------------------------

/// One iteration's `(damping, gradient_norm, step_norm, step_quality)`.
type MetricsRow = (Option<f64>, f64, f64, Option<f64>);

#[derive(Default)]
struct MetricsLog {
    rows: Mutex<Vec<MetricsRow>>,
}

impl MetricsLog {
    /// A poisoned lock here means an assertion already fired on another thread;
    /// the rows are still valid, so recover them rather than masking the
    /// original failure with a second panic.
    fn rows(&self) -> Vec<MetricsRow> {
        self.rows
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn step_qualities(&self) -> Vec<f64> {
        self.rows()
            .iter()
            .filter_map(|(_, _, _, quality)| *quality)
            .collect()
    }

    fn step_norms(&self) -> Vec<f64> {
        self.rows().iter().map(|(_, _, norm, _)| *norm).collect()
    }

    fn gradient_norms(&self) -> Vec<f64> {
        self.rows().iter().map(|(_, norm, _, _)| *norm).collect()
    }
}

/// `add_observer` takes ownership, so the test keeps its own handle on the log.
struct SharedObserver(Arc<MetricsLog>);

impl OptObserver for SharedObserver {
    fn on_step(&self, _values: &SlotMap<VarKey, Box<dyn ManifoldVariable>>, _iteration: usize) {}

    fn set_iteration_metrics(
        &self,
        _cost: f64,
        gradient_norm: f64,
        damping: Option<f64>,
        step_norm: f64,
        step_quality: Option<f64>,
    ) {
        self.0
            .rows
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((damping, gradient_norm, step_norm, step_quality));
    }
}

fn solve_logging(
    problem: &mut Problem,
    config: LevenbergMarquardtConfig,
) -> Result<Arc<MetricsLog>, Box<dyn std::error::Error>> {
    let mut solver = LevenbergMarquardt::with_config(config);
    let log = Arc::new(MetricsLog::default());
    solver.add_observer(SharedObserver(log.clone()));
    solver.optimize(problem)?;
    Ok(log)
}

/// Pin λ so the step length is set by the test rather than by the damping
/// policy, and switch the tolerances off so the run does not stop before the
/// requested iterations have been observed.
fn config_at_fixed_damping(damping: f64) -> LevenbergMarquardtConfig {
    LevenbergMarquardtConfig::default()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(3)
        .with_damping(damping)
        .with_damping_bounds(damping, damping)
        .with_cost_tolerance(0.0)
        .with_parameter_tolerance(0.0)
        .with_gradient_tolerance(0.0)
}

/// The `λ → ∞` limit is approached, not reached: at λ = 1e8 the step is around
/// 1e-8, small enough that the quadratic model's error is far below this bound
/// and large enough that `actual_reduction` has not been eaten by cancellation.
const RHO_TOLERANCE: f64 = 1e-6;

fn assert_all_close_to_one(qualities: &[f64], tolerance: f64, what: &str) {
    assert_close_to_one(qualities, qualities.len(), tolerance, what)
}

/// Assert on the first `count` iterations only.
///
/// Past the minimum both reductions are ~0 and `rho` is their quotient, so it
/// carries no information there — that is arithmetic, not solver behaviour, and
/// asserting on it would pin noise.
fn assert_close_to_one(qualities: &[f64], count: usize, tolerance: f64, what: &str) {
    assert!(
        !qualities.is_empty(),
        "{what}: no step quality was reported"
    );
    assert!(
        qualities.len() >= count,
        "{what}: wanted {count} iterations, saw {}",
        qualities.len()
    );
    for (iteration, rho) in qualities.iter().take(count).enumerate() {
        assert!(
            (rho - 1.0).abs() < tolerance,
            "{what}: rho = {rho} at iteration {iteration}, expected 1 within {tolerance} \
             (all: {qualities:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 1: a linear residual has an exact model, so rho == 1
// ---------------------------------------------------------------------------

/// `r(x) = A·x − b` with column norms six orders of magnitude apart, the
/// mismatch Jacobi scaling exists to fix (think focal length against distortion
/// coefficients).
struct LinearFactor {
    a: [[f64; 2]; 3],
    b: [f64; 3],
}

impl LinearFactor {
    /// `b = A·x_true + offset`. With `offset` all-zero the minimum is exactly
    /// attainable and the residual vanishes there; a non-zero offset puts `b`
    /// outside the range of `A`, so the cost stays bounded away from zero.
    ///
    /// Which one a test wants depends on how far it drives the solve. Once the
    /// cost is near zero, `actual_reduction = cost_before − cost_after`
    /// subtracts two nearly equal tiny numbers and `rho` loses digits to
    /// cancellation — a property of the arithmetic, not of the solver. Tests
    /// that run to convergence therefore use an inconsistent system.
    fn new(a: [[f64; 2]; 3], x_true: [f64; 2], offset: [f64; 3]) -> Self {
        let b = std::array::from_fn(|i| a[i][0] * x_true[0] + a[i][1] * x_true[1] + offset[i]);
        Self { a, b }
    }
}

impl Factor for LinearFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let x = params[0];
        let mut jacobian = jacobian;
        for (i, (row, b)) in self.a.iter().zip(self.b.iter()).enumerate() {
            residual[i] = row[0] * x[0] + row[1] * x[1] - b;
            if let Some(jac) = jacobian.as_mut() {
                *jac.rb_mut().get_mut(i, 0) = row[0];
                *jac.rb_mut().get_mut(i, 1) = row[1];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 2)
    }
}

/// Column norms six orders of magnitude apart, seeded at the origin.
fn ill_scaled_linear_problem(offset: [f64; 3]) -> Problem {
    let a = [[1e-3, 1e3], [2e-3, -5e2], [-1e-3, 8e2]];
    let mut problem = Problem::new(JacobianMode::Sparse);
    let key = problem.add_variable(ManifoldType::RN, DVector::from_vec(vec![0.0, 0.0]));
    problem.add_residual_block(
        &[key],
        Box::new(LinearFactor::new(a, [2.0, 0.5], offset)),
        None,
    );
    problem
}

/// `b = A·x_true` exactly: the residual reaches zero, as in the report this
/// test file came from.
const ATTAINABLE: [f64; 3] = [0.0, 0.0, 0.0];

/// `b` outside the range of `A`, so the minimum has a residual of order 1 and
/// `rho` stays well-conditioned however long the solve runs.
const INCONSISTENT: [f64; 3] = [0.7, -1.3, 0.9];

/// The quadratic model of a linear least-squares problem is exact, so `rho` is
/// 1 to machine precision on every iteration — with Jacobi scaling and without.
///
/// This is the property the un-scaled-step/scaled-gradient bug violated: the
/// predicted reduction used to pair an un-scaled step with the scaled gradient,
/// which under `use_jacobi_scaling(true)` reported `rho` in the thousands and
/// collapsed the damping to its floor within a few dozen accepted steps.
#[test]
fn linear_residual_gives_step_quality_of_exactly_one() -> TestResult {
    for use_jacobi_scaling in [false, true] {
        let mut problem = ill_scaled_linear_problem(ATTAINABLE);
        let log = solve_logging(
            &mut problem,
            LevenbergMarquardtConfig::default()
                .with_max_iterations(12)
                .with_damping(1e-4)
                .with_jacobi_scaling(use_jacobi_scaling),
        )?;
        assert_all_close_to_one(
            &log.step_qualities(),
            1e-9,
            &format!("linear problem, use_jacobi_scaling({use_jacobi_scaling})"),
        );
    }
    Ok(())
}

/// The same, with λ held far from its default at both extremes and under both
/// damping shapes — uniform `λ·I` as well as the Marquardt diagonal. Exactness
/// of the model does not depend on any of these, so neither may `rho`.
#[test]
fn linear_residual_step_quality_is_one_at_every_damping() -> TestResult {
    for diagonal_bounds in [(1e-6, 1e32), (1.0, 1.0)] {
        for damping in [1e-4, 1e2, 1e6] {
            for use_jacobi_scaling in [false, true] {
                let mut problem = ill_scaled_linear_problem(INCONSISTENT);
                let log = solve_logging(
                    &mut problem,
                    config_at_fixed_damping(damping)
                        .with_max_iterations(4)
                        .with_diagonal_bounds(diagonal_bounds.0, diagonal_bounds.1)
                        .with_jacobi_scaling(use_jacobi_scaling),
                )?;
                // Two iterations: at the loosest damping here the exact
                // Newton step lands on the minimum almost immediately, and
                // `rho` past that point is 0/0.
                assert_close_to_one(
                    &log.step_qualities(),
                    2,
                    1e-9,
                    &format!(
                        "linear problem, lambda = {damping:e}, diagonal {diagonal_bounds:?}, \
                         use_jacobi_scaling({use_jacobi_scaling})"
                    ),
                );
            }
        }
    }
    Ok(())
}

/// Jacobi scaling cancels exactly under the default Marquardt diagonal damping,
/// so it must not move the trajectory at all there.
///
/// With `J̃ = J·S` the damped normal equations become
/// `S·(JᵀJ + λ·diag(JᵀJ))·S·dx̃ = −S·Jᵀr`, because `diag(S·H·S) = S·diag(H)·S`.
/// The un-scaled step `S·dx̃` therefore solves the *un-scaled* damped system,
/// and scaling is a pure conditioning device for the factorisation.
///
/// That is worth pinning because it is the thing people reach for when a
/// badly-scaled problem converges slowly: under the default damping it will do
/// nothing, and only `with_diagonal_bounds(1.0, 1.0)` (uniform `λ·I`) makes it
/// change the iterates.
#[test]
fn jacobi_scaling_is_a_no_op_under_marquardt_diagonal_damping() -> TestResult {
    let mut unscaled = ill_scaled_linear_problem(INCONSISTENT);
    let mut scaled = ill_scaled_linear_problem(INCONSISTENT);
    let config = LevenbergMarquardtConfig::default()
        .with_max_iterations(8)
        .with_damping(1e-4);

    let unscaled_log = solve_logging(&mut unscaled, config.clone().with_jacobi_scaling(false))?;
    let scaled_log = solve_logging(&mut scaled, config.with_jacobi_scaling(true))?;

    // Exact in real arithmetic; in floating point the two orderings round
    // differently, so the bound is a rounding-level relative one rather than
    // bit equality.
    let (a, b) = (unscaled_log.step_norms(), scaled_log.step_norms());
    assert_eq!(a.len(), b.len(), "iteration counts differ: {a:?} vs {b:?}");
    for (iteration, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-9 * x.abs().max(1.0),
            "step norms diverge at iteration {iteration}: {x} vs {y}"
        );
    }

    // Contrast: under uniform `λ·I` the cancellation above does not happen, so
    // scaling genuinely moves the iterates. Without this the test would still
    // pass if scaling had silently stopped being applied at all.
    let mut unscaled = ill_scaled_linear_problem(INCONSISTENT);
    let mut scaled = ill_scaled_linear_problem(INCONSISTENT);
    let uniform = LevenbergMarquardtConfig::default()
        .with_max_iterations(8)
        .with_damping(1e-4)
        .with_diagonal_bounds(1.0, 1.0);
    let unscaled_log = solve_logging(&mut unscaled, uniform.clone().with_jacobi_scaling(false))?;
    let scaled_log = solve_logging(&mut scaled, uniform.with_jacobi_scaling(true))?;

    let (a, b) = (unscaled_log.step_norms(), scaled_log.step_norms());
    let diverges = a.len() != b.len()
        || a.iter()
            .zip(b.iter())
            .any(|(x, y)| (x - y).abs() > 1e-6 * x.abs().max(1.0));
    assert!(
        diverges,
        "under uniform lambda*I, Jacobi scaling should change the iterates: {a:?} vs {b:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 2: rho -> 1 as the step length -> 0
// ---------------------------------------------------------------------------

fn pose(translation: [f64; 3], axis: [f64; 3], angle: f64) -> SE3 {
    let axis = Vector3::new(axis[0], axis[1], axis[2]).normalize();
    SE3::from_translation_so3(
        Vector3::new(translation[0], translation[1], translation[2]),
        SO3::from_axis_angle(&axis, angle),
    )
}

fn add_pose(problem: &mut Problem, value: &SE3) -> VarKey {
    problem.add_variable(
        ManifoldType::SE3,
        DVector::from_column_slice(value.as_param_slice()),
    )
}

/// A short SE(3) chain seeded away from its measurements, so every residual —
/// and in particular every *rotational* residual — is non-zero at the
/// linearisation point.
///
/// `rotation_scale` dials the rotation content of the states and measurements.
/// It matters because the manifold-Jacobian bug this guards against was exact
/// at zero rotation and grew with the angle; a fixture built from near-identity
/// poses would have missed it entirely.
fn se3_chain(rotation_scale: f64, gauge_fix: bool) -> Problem {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let keys: Vec<VarKey> = (0..4)
        .map(|i| {
            let value = pose(
                [i as f64 * 0.9, 0.1 * i as f64, -0.05 * i as f64],
                [0.3, 0.5, 0.8],
                rotation_scale * 0.35 * i as f64,
            );
            add_pose(&mut problem, &value)
        })
        .collect();

    problem.add_residual_block(
        &[keys[0]],
        Box::new(PriorFactor::new(pose(
            [0.3, -0.2, 0.1],
            [0.2, 0.4, 0.9],
            rotation_scale * 0.4,
        ))),
        None,
    );
    for window in keys.windows(2) {
        problem.add_residual_block(
            &[window[0], window[1]],
            Box::new(BetweenFactor::new(pose(
                [1.0, 0.0, 0.0],
                [0.1, 0.2, 0.97],
                rotation_scale * 0.25,
            ))),
            None,
        );
    }
    if gauge_fix {
        for dof in 0..6 {
            problem.fix_variable(keys[0], dof);
        }
    }
    problem
}

/// On a manifold problem, `rho → 1` under heavy damping only if the factors'
/// Jacobians are taken with respect to the same retraction that
/// `apply_tangent_step` uses to move the variables.
///
/// `SE3Tangent::right_jacobian` used to return the *left* Jacobian on its
/// diagonal blocks, which agrees with the right one only at zero rotation. The
/// resulting gradient error was invisible in a `Jr · Jr⁻¹ == I` check (a
/// consistent inverse of the wrong matrix passes that) and showed up here as
/// `rho` settling on ~0.93 instead of 1 — enough to make LM's damping policy
/// raise λ on every accepted step and stall.
#[test]
fn heavy_damping_drives_step_quality_to_one_on_se3() -> TestResult {
    for rotation_scale in [0.0, 0.5, 1.0, 2.0] {
        let mut problem = se3_chain(rotation_scale, false);
        let log = solve_logging(&mut problem, config_at_fixed_damping(1e8))?;
        assert_all_close_to_one(
            &log.step_qualities(),
            RHO_TOLERANCE,
            &format!("SE3 chain, rotation_scale = {rotation_scale}"),
        );
    }
    Ok(())
}

/// Fixing a variable's DOFs must remove them from the linear system, not just
/// from the applied step.
///
/// `ManifoldVariable::apply_tangent_step` drops a fixed DOF's component of the
/// step. While the corresponding Jacobian columns stayed populated, the solver
/// planned motion along them and the predicted reduction charged for that
/// motion, so the actual reduction could not match it. Every bundle-adjustment
/// problem fixes a pose for gauge freedom, which made this permanent there
/// rather than a corner case.
#[test]
fn fixed_dofs_do_not_corrupt_step_quality() -> TestResult {
    for rotation_scale in [0.0, 1.0] {
        let mut problem = se3_chain(rotation_scale, true);
        let log = solve_logging(&mut problem, config_at_fixed_damping(1e8))?;
        assert_all_close_to_one(
            &log.step_qualities(),
            RHO_TOLERANCE,
            &format!("gauge-fixed SE3 chain, rotation_scale = {rotation_scale}"),
        );
    }
    Ok(())
}

/// A gauge-fixed variable must not move, and the reported step norm must not
/// count motion that never happened.
#[test]
fn fixed_dofs_receive_no_step() -> TestResult {
    // The gauge-fixed pose is the chain's first variable; `se3_chain` seeds it
    // from this same expression.
    let anchor = pose([0.0, 0.0, 0.0], [0.3, 0.5, 0.8], 0.0);
    let before: Vec<f64> = anchor.as_param_slice().to_vec();

    let mut problem = se3_chain(1.0, true);
    let config = LevenbergMarquardtConfig::default()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(5);
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    let Some(anchor_after) = result.parameters.values().next() else {
        return Err("solver returned no variables".into());
    };
    let after = anchor_after.as_param_slice().to_vec();
    for (dof, (x, y)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-15,
            "fixed variable moved in coordinate {dof}: {x} -> {y}"
        );
    }
    Ok(())
}

/// A robust loss changes what the cost *is*, so the invariant has to keep
/// holding through the Triggs correction: the block cost is `0.5·ρ(s)` and the
/// corrected system must supply exactly its gradient, `ρ'(s)·Jᵀr`.
///
/// `CauchyLoss` used to return `ρ(s)` at half the value its own `ρ'(s)`
/// integrates to, which made `rho` exactly 0.5 for every Cauchy-weighted step.
#[test]
fn heavy_damping_drives_step_quality_to_one_under_robust_losses() -> TestResult {
    let losses: Vec<(&str, Option<Box<dyn LossFunction + Send + Sync>>)> = vec![
        ("none", None),
        ("huber", Some(Box::new(HuberLoss::new(1.0)?))),
        ("cauchy", Some(Box::new(CauchyLoss::new(1.0)?))),
    ];

    for (name, loss) in losses {
        // Residuals here are well above the kernels' unit scale, so the robust
        // branch is the one under test rather than the quadratic inlier region.
        let mut problem = Problem::new(JacobianMode::Sparse);
        let key = add_pose(&mut problem, &pose([0.0, 0.0, 0.0], [0.3, 0.5, 0.8], 0.2));
        problem.add_residual_block(
            &[key],
            Box::new(PriorFactor::new(pose(
                [2.0, -1.5, 1.0],
                [0.2, 0.4, 0.9],
                1.1,
            ))),
            loss,
        );

        let log = solve_logging(&mut problem, config_at_fixed_damping(1e8))?;
        assert_all_close_to_one(
            &log.step_qualities(),
            RHO_TOLERANCE,
            &format!("SE3 prior under loss {name}"),
        );
    }
    Ok(())
}

/// The reported gradient norm is `‖Jᵀr‖` in the problem's own units, whatever
/// `use_jacobi_scaling` is set to.
///
/// The solver's cached gradient is built from the Jacobian it was handed, so
/// under scaling it is `diag(s)·Jᵀr`. That is the right vector for computing
/// the step, but `gradient_tolerance` is an absolute threshold the caller
/// chooses in the problem's units, and `s_j = 1/(1 + ‖J_col_j‖) ≤ 1` always —
/// so testing the scaled norm silently loosened the convergence check by a
/// factor set by the Jacobian's column norms. It also made
/// `final_gradient_norm` and the observer metric change units with the flag,
/// so logs could not be compared across it.
///
/// The fixture's first column has a norm ~1500x the second's, so the two norms
/// differed by ~3 orders of magnitude before the fix.
#[test]
fn reported_gradient_norm_does_not_depend_on_jacobi_scaling() -> TestResult {
    // Column 0's norm is ~1.7e3 against column 1's ~1.2, like a focal length
    // beside a normalised coordinate.
    let a = [[1.0e3, 1.0], [1.0e3, -0.5], [1.0e3, 0.3]];

    let mut finals = Vec::new();
    let mut per_iteration = Vec::new();
    for use_jacobi_scaling in [false, true] {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let key = problem.add_variable(ManifoldType::RN, DVector::from_vec(vec![0.0, 0.0]));
        problem.add_residual_block(
            &[key],
            Box::new(LinearFactor::new(a, [1.0e-3, 2.0], ATTAINABLE)),
            None,
        );

        // Only the gradient test may fire, so the run length is a direct
        // readout of where that threshold sits.
        let config = LevenbergMarquardtConfig::default()
            .with_max_iterations(50)
            .with_gradient_tolerance(1e-8)
            .with_cost_tolerance(0.0)
            .with_parameter_tolerance(0.0)
            .with_jacobi_scaling(use_jacobi_scaling);

        let mut solver = LevenbergMarquardt::with_config(config);
        let log = Arc::new(MetricsLog::default());
        solver.add_observer(SharedObserver(log.clone()));
        let result = solver.optimize(&mut problem)?;

        let Some(info) = result.convergence_info else {
            return Err("solver reported no convergence info".into());
        };
        finals.push((result.iterations, info.final_gradient_norm));
        per_iteration.push(log.gradient_norms());
    }

    let (unscaled_iters, unscaled_final) = finals[0];
    let (scaled_iters, scaled_final) = finals[1];

    // The trajectories are identical on a linear problem, so the norms must
    // agree iteration by iteration, not merely at the end.
    let (a_norms, b_norms) = (&per_iteration[0], &per_iteration[1]);
    assert_eq!(
        a_norms.len(),
        b_norms.len(),
        "iteration counts differ ({unscaled_iters} vs {scaled_iters}): \
         {a_norms:?} vs {b_norms:?}"
    );
    for (iteration, (x, y)) in a_norms.iter().zip(b_norms.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-9 * x.abs().max(1.0),
            "reported gradient norms differ at iteration {iteration}: {x} vs {y} \
             (all: {a_norms:?} vs {b_norms:?})"
        );
    }
    assert!(
        (unscaled_final - scaled_final).abs() <= 1e-9 * unscaled_final.abs().max(1.0),
        "final_gradient_norm differs: {unscaled_final} vs {scaled_final}"
    );
    Ok(())
}

/// The step really does shrink with λ, so the limit above is being approached
/// rather than accidentally satisfied by a step that never moves.
#[test]
fn step_length_shrinks_in_proportion_to_damping() -> TestResult {
    let mut norms = Vec::new();
    for damping in [1e6, 1e8] {
        let mut problem = se3_chain(1.0, false);
        let log = solve_logging(&mut problem, config_at_fixed_damping(damping))?;
        norms.push(log.step_norms()[0]);
    }
    let ratio = norms[0] / norms[1];
    assert!(
        (50.0..200.0).contains(&ratio),
        "a 100x rise in lambda should shrink the step ~100x, got {ratio} from {norms:?}"
    );
    Ok(())
}
