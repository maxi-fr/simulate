# Port `adaptive-satellite-control` into `simulate`

## Context

The old repo [`maxi-fr/adaptive-satellite-control`](https://github.com/maxi-fr/adaptive-satellite-control)
is a working-but-sloppy satellite ADCS simulation (≈67% notebooks, 2 test files). We are
re-implementing it cleanly inside this repo's block-based framework (`src/simulate`) + aerospace
package (`src/rigid_body`), using the old repo only as a behavioural outline. MPC and its
AL-iLQR/CasADi solver are explicitly **out of scope**.

Goal: the repo can simulate a satellite holding **nadir pointing** in LEO under real disturbances,
with reaction wheels + magnetorquers, a **full-state nonlinear estimator** (orbit KF + attitude MEKF

+ exposed environment variables), and both a quaternion-feedback and an LQR controller.

The plan is organised into **phases**, each broken into **individually implementable + verifiable
steps**. Each step lists its scope, the file(s) it touches, what to reuse, and a concrete verify
check. Steps within a phase are mostly sequential; phases are strictly ordered by dependency.

---

## Already ported (no work needed)

+ 6-DOF dynamics + quaternion kinematics — [rigid_body.py](src/rigid_body/rigid_body.py), [quaternion.py](src/rigid_body/quaternion.py)

+ Actuators as `Effector`s: `ReactionWheelArray`, `MagnetorquerArray`, `Wrench` — [effector.py](src/rigid_body/effector.py)
+ Disturbances: gravity gradient, third-body, SRP, drag — [effector.py](src/rigid_body/effector.py), [disturbances.py](src/rigid_body/disturbances.py)
+ Environment: IGRF B-field, MSIS density, sun/moon ephemeris (cached), eclipse — [environment.py](src/rigid_body/environment.py)
+ Surfaces, SGP4 orbit propagation — [surface.py](src/rigid_body/surface.py), [orbit_dynamics.py](src/rigid_body/orbit_dynamics.py)
+ Multi-rate sim loop, logging, batch runner, generic Gaussian/random-walk sensors — `src/simulate/*`
+ Existing Outputs: pose, attitude (`q`), rate (`omega`), RW telemetry — [rigid_body.py](src/rigid_body/rigid_body.py)

## Gaps in the old repo (deficiencies we deliberately improve on)

These are things the old repo lacked or did poorly — the port fixes them rather than copying them:

+ **Estimation was attitude-only** (MEKF on `q` + gyro bias). No orbit/position estimation and no
  environment-variable estimation. → This plan adds an **orbit KF** and **environment-variable
  exposure** alongside the MEKF (the main upgrade requested).
+ **Monolithic + notebook-heavy**, JSON config + one big `main.py` loop, almost no tests (only
  quaternion + AL-iLQR). → Replaced by the `simulate` block architecture, YAML dynamic loading, and
  per-component unit tests.
+ **Accelerometer** was only "implied/planned", never implemented. → Out of scope unless asked.
+ **Heavy deps** `control` + `casadi` (for MPC) and manual IGRF coeff download
  (`setup_igrf.py`/`igrf14coeffs.txt`, `pyIGRF`). → Use `scipy` for Riccati and the already-present
  `ppigrf`; add no MPC deps.
+ **Scattered frame conversions** (ORC/SBC/geodetic mixed into kinematics). → Consolidated into one
  `frames.py`.
+ **CSV-only logging, no batch sweeps.** → Already solved by `Logger` + `ExperimentManager`.

## Decisions baked in (from clarification)

+ Controllers: **Quaternion feedback (PD + momentum dumping)** + **LQR / AdaptiveLQR**. No B-dot,
  Avanzini, or MPC.

+ Estimator: **full nonlinear** — orbit Kalman filter (r/v from GPS) + attitude **MEKF**
  (`q` + gyro bias, sun/magnetometer/star-tracker updates) — plus exposed **environment variables**:
  magnetic field, sun direction, atmospheric density, smoothed orbit position/velocity.
+ Sensors to port: **magnetometer, sun sensor, GPS, gyro, RW tachometer**.
+ Target scenario: **nadir pointing** hold (start near-pointed) with reaction wheels under disturbances.
+ All new satellite-specific classes live in `src/rigid_body` (keep `src/simulate` generic).
+ Use `scipy.linalg.solve_discrete_are` for Riccati; **do not** add `control` or `casadi`.
+ `x_hat` layout exposed to the controller = `[r(3), v(3), q(4), omega(3), b_body(3), h_wheel(3)]`
  (length 19). The first 13 are the orbit/attitude state; the trailing **estimated body-frame
  magnetic field** `b_body` and **reaction-wheel angular momentum** `h_wheel` are appended because
  the controllers (magnetorquer allocation + momentum dumping) only receive `x_hat`, never the
  estimator log. Gyro bias and the remaining environment variables still ride in the estimator's
  **log dataclass**. (Implemented divergence from the original "first 13 only" decision — see
  [estimator.py](src/rigid_body/estimator.py).)

---

## Phase 1 — Frames & kinematics helpers

Foundation for the reference, MEKF and controllers.

+ **Step 1.1 — ORC orbital frame.** `src/rigid_body/frames.py`: `orc_from_orbit(r_eci, v_eci) -> Quaternion`
  building the local orbital frame (nadir / along-track / orbit-normal) and inertial→ORC rotation;
  `orbital_rate(r_eci, v_eci) -> omega` for reference feedforward. Reuse [quaternion.py](src/rigid_body/quaternion.py).
  *Verify:* test that nadir axis equals `-r̂` and the frame is orthonormal; rate ≈ mean motion on a circular orbit.
+ **Step 1.2 — Euler conversions.** Add `euler_from_quaternion`/`quaternion_from_euler` (intrinsic Y-X-Z,
  matching old repo) to `frames.py` for references/analysis. *Verify:* round-trip euler↔quaternion to tolerance.
+ **Step 1.3 — Attitude-error helpers.** Add `conjugate`/`inverse` and `error_to(other)` (small-angle error
  quaternion `q_err = q_ref^-1 ⊗ q`) to [quaternion.py](src/rigid_body/quaternion.py) only if missing; reuse
  existing `__mul__`, `apply`, `to_rot_mat`. *Verify:* identity error for equal quaternions; error vector
  sign matches a known small rotation.

## Phase 2 — Environment-coupled measurement Outputs

New satellite sensor Outputs reading orbit state + environment at `epoch + t`. Follow the existing epoch
pattern in `AerodynamicDrag`/`ThirdBody` ([effector.py:198-200, 587-590](src/rigid_body/effector.py)):
`_ensure_utc`, `dt_utc = epoch + timedelta(seconds=t)`, `pymap3d.eci2ecef`. All in `src/rigid_body/measurement.py`.

+ **Step 2.1 — `MagneticFieldOutput(epoch)`.** ECI→ECEF→geodetic, `environment.magnetic_field_vector`,
  rotate into body frame via `Quaternion.apply`. *Verify:* magnitude matches `environment` truth; vector
  rotates correctly when attitude changes.
+ **Step 2.2 — `SunDirectionOutput(epoch)`.** `environment.sun_position` + `environment.is_in_shadow`;
  return body-frame unit sun vector, **zero in eclipse** (old sun-sensor-inactive behaviour). *Verify:*
  unit-norm in sunlight, zero in shadow.
+ **Step 2.3 — `GpsOutput`.** Position (+ velocity) slice of the state, no epoch needed. *Verify:* returns
  `r`/`v` slices matching the state.
+ **Step 2.4 — Gyro & RW tachometer pairing (no new classes).** Document/cover pairing existing
  `RigidBodyRateOutput`→`RandomWalkBiasSensor` (gyro) and `ReactionWheelTelemetryOutput`→`GaussianSensor`
  (tachometer). *Verify:* test that the paired channel yields biased/noisy measurements of the truth.

## Phase 3 — Nadir-pointing reference

Desired attitude is deterministic from the orbit, so derive it from SGP4 (decoupled from feedback state).

+ **Step 3.1 — `NadirPointingReference(Reference)`.** `src/rigid_body/reference.py`: holds an
  [`SGP4`](src/rigid_body/orbit_dynamics.py) propagator + `epoch`; each step propagate to `epoch+t`, build
  desired `q` via `frames.orc_from_orbit` and feedforward `omega` via `frames.orbital_rate`; emit
  `ref = [q_des(4), omega_des(3)]`. `from_config` builds SGP4 from TLE/elements. *Verify:* reference
  attitude tracks nadir over an orbit, `q_des` unit-norm, rate ≈ mean motion.

## Phase 4 — Full nonlinear estimator (orbit KF + MEKF + environment vars)

The estimator receives the concatenated `y_mea` (attitude, rate, magnetometer, sun, GPS, wheel speeds)
and `u`, and returns `x_hat = [r, v, q, omega, b_body, h_wheel]` (length 19,
[simulation.py:154-160](src/simulate/simulation.py#L154-L160)). The trailing `b_body`/`h_wheel` feed
the controllers (see the Phase-5 note); the first 13 are the orbit/attitude state.
Build incrementally; each sub-filter is independently testable. All in `src/rigid_body/estimator.py`,
subclassing [`Estimator`](src/simulate/estimator.py). A frozen log dataclass carries gyro bias, the
exposed environment variables, and the body-frame wheel momentum.

+ **Step 4.1 — Measurement layout helper.** A small parser mapping the concatenated `y_mea` to named
  channels (must mirror the `outputs`/`sensors` ordering in the config). *Verify:* slicing round-trips a
  synthetic concatenated vector.
+ **Step 4.2 — Orbit Kalman filter.** Linear KF over `[r, v]`: two-body predict (reuse
  [orbit_dynamics.py](src/rigid_body/orbit_dynamics.py)), GPS position/velocity update, process/measurement
  covariances from config. *Verify:* on a noisy GPS feed from a known orbit, estimated `r/v` error is below
  the raw measurement noise (smoothing demonstrably helps).
+ **Step 4.3 — Attitude MEKF.** Port the old `AttitudeEKF`: state = error-quaternion + gyro bias + rate;
  gyro propagation; **sun + magnetometer vector updates** (and optional direct star-tracker `q` update);
  Joseph-form covariance; multiplicative reset. Reuse Phase-1 error helpers. *Verify:* torque-free spin with
  noisy gyro/sun/mag converges to true attitude and the gyro-bias estimate tracks an injected bias.
+ **Step 4.4 — Environment-variable exposure.** From the estimated orbit (`r/v` + `epoch+t`) compute and
  log: magnetic field (`environment.magnetic_field_vector`), sun direction + eclipse, atmospheric density
  (`environment.atmosphere_density_msis`), and smoothed geodetic lat/lon/alt. Reuse
  [environment.py](src/rigid_body/environment.py). *Verify:* exposed env vars match the truth-at-estimated-orbit
  within tolerance; logged via the estimator dataclass.
+ **Step 4.5 — Assemble `FullStateEstimator`.** Compose 4.2–4.4 into one `Estimator` producing
  `x_hat = [r, v, q, omega, b_body, h_wheel]` (length 19) and the rich log; `from_config` wires noise
  params, the measurement layout, and the reaction-wheel array (axes + inertia) used to turn the
  tachometer channel into the body-frame `h_wheel`. *Verify:* end-to-end on a short sim, `x_hat`
  tracks truth and the run is deterministic with a fixed seed.

## Phase 5 — Attitude controllers

Consume `x_hat` `[r,v,q,omega,b_body,h_wheel]` + reference `[q_des, omega_des]`; output **actuator
current commands** (not torque). Because the `ReactionWheelArray`/`MagnetorquerArray` effectors
interpret their command slice as currents, each controller computes a desired torque and then
allocates it to currents with `to_current_commands` before returning `u` — porting the legacy
`PI.calc_input_cmds` flow. Output order is `[i_mtq, i_rw]`, so the dynamics config must list the
magnetorquer array before the reaction-wheel array. New classes in `src/rigid_body/controller.py`,
mirroring `PIDController` ([controller.py](src/simulate/controller.py)).

+ **Step 5.1 — `QuaternionFeedbackController`.** `tau_rw = -Kp·q_err_vec - Kd·(omega - omega_des)`
  plus magnetorquer momentum dumping `tau_mtq = -k_m·h_wheel` (disabled by `k_m = 0`), both allocated
  to currents. `b_body` and `h_wheel` come from `x_hat` (the controller never sees the estimator log).
  Port old `ClassicalQuatFeedback`; uses Phase-1 error helpers. *Verify:* closed-loop on real
  `RigidBodyDynamics` drives a small initial error below tolerance within N steps; wheel momentum
  bounded with dumping on.
+ **Step 5.2 — Linearization helpers.** `src/rigid_body/linearization.py`: port the error/attitude
  jacobian + RK2-normalized error dynamics → reduced discrete `(A, B)` from old `controller_models.py`,
  reimplemented in **NumPy** (CasADi is out of scope) via central finite differences. The reduced state
  is the 6-vector `[delta_theta, delta_omega]` with input `[m, tau_rw]` (magnetorquer dipole +
  reaction-wheel torque); `B` enters the input matrix through `m × B`. *Note: a wheel-momentum state is
  deliberately omitted* — with a frozen field the total-momentum component along `B` is uncontrollable
  (`m × B ⊥ B`), which makes the discrete Riccati equation singular; reaction wheels alone keep the
  6-state model fully controllable. *Verify:* finite-difference check of the jacobian against the
  nonlinear error dynamics.
+ **Step 5.3 — `LQRController`.** Solve discrete Riccati via `scipy.linalg.solve_discrete_are` on the
  Phase-5.2 model with field-averaged B; gains from config `Q`/`R` (6×6). Outputs `[m, tau_rw] = -K·x`
  allocated to currents. Magnetic momentum dumping is left to `QuaternionFeedbackController` (the LQR
  model has no momentum state). *Verify:* stabilizes the linearized system (closed-loop eigenvalues
  inside unit circle) and drives the nonlinear plant error to tolerance.
+ **Step 5.4 — `AdaptiveLQRController`.** Re-solve on the updated/averaged model with Newton-Kleinman warm
  start. *Verify:* matches `LQRController` on a static model; adapts (gain changes) when the model is varied.

## Phase 6 — End-to-end nadir-pointing example, config & analysis

+ **Step 6.1 — YAML satellite config.** Full satellite (mass, inertia, surfaces, RW array, magnetorquer
  array, gravity-gradient + drag + SRP disturbances, epoch/TLE), the five sensor channels,
  `FullStateEstimator`, `NadirPointingReference`, `QuaternionFeedbackController`. Drives
  `Simulation.from_yaml`; replaces the old `simulation_config.json`. *Verify:* `from_yaml` builds and runs.

+ **Step 6.2 — Marimo example.** `examples/03_nadir_pointing.py` mirroring
  [02_rigid_body_attitude.py](examples/02_rigid_body_attitude.py), with an LQR variant cell. *Verify:*
  `uv run marimo check` clean; pointing error settles and holds nadir over one+ orbit.
+ **Step 6.3 — Analysis cells.** Pointing error (Euler, Phase-1 helper), body rates, wheel speeds, control
  torque, and estimator vs. truth overlays — replacing old `analysis.ipynb`. *Verify:* plots render from the
  logged `.npz`.

## Phase 7 — Docs & green build

+ **Step 7.1 — Whitepaper.** Update [whitepaper.md](whitepaper.md): new satellite components, SGP4-driven
  nadir reference, the full-state estimator (orbit KF + MEKF + env-var exposure), the controller set; flag
  any divergence from the old repo. *Verify:* reads consistently with the code.

+ **Step 7.2 — Green build.** `uv run ruff check . --fix --unsafe-fixes`, `uv run ruff format .`,
  `uv run ty check`, `uv run pytest` all pass (pre-commit runs these on commit). *Verify:* clean run, no
  unscoped ignores.

---

## Critical files

+ New: `src/rigid_body/frames.py`, `measurement.py`, `reference.py`, `estimator.py`, `controller.py`, `linearization.py`

+ Edit (small helpers only): [src/rigid_body/quaternion.py](src/rigid_body/quaternion.py)
+ New tests: `tests/test_frames.py`, `test_measurement.py`, `test_reference.py`, `test_estimator.py`, `test_attitude_controller.py`
+ New example + config: `examples/03_nadir_pointing.py`, satellite YAML config
+ Docs: [whitepaper.md](whitepaper.md)

## Reuse (do not re-implement)

+ Epoch + `eci2ecef` pattern: [effector.py:198-200, 587-590](src/rigid_body/effector.py)

+ Environment models: `magnetic_field_vector`, `sun_position`, `is_in_shadow`, `atmosphere_density_msis` — [environment.py](src/rigid_body/environment.py)
+ Orbit propagation: [`SGP4`](src/rigid_body/orbit_dynamics.py); two-body accel: [orbit_dynamics.py](src/rigid_body/orbit_dynamics.py)
+ Outputs to pair with generic sensors: `RigidBodyAttitudeOutput`, `RigidBodyRateOutput`, `ReactionWheelTelemetryOutput` — [rigid_body.py](src/rigid_body/rigid_body.py)
+ Measurement/estimator data flow: [simulation.py:151-166](src/simulate/simulation.py#L151-L166)
+ Component/`from_config` conventions: [component.py](src/simulate/component.py), [controller.py](src/simulate/controller.py)

## End-to-end verification

1. `uv run pytest` — all new + existing tests green.
2. Run the nadir-pointing YAML via `Simulation.from_yaml` and the example notebook; confirm pointing error
   converges and holds nadir under disturbances over at least one orbit, and that the estimator's `x_hat`
   and exposed environment variables track truth.
3. `uv run ty check` and `uv run ruff check .` clean (or with justified, scoped ignores only).
