# f1tenth_planning — Bug & Issue Scan

> Adversarial scan (parallel finders per subsystem → skeptic verification → dedup), run 2026-07-20 against branch `dev-sit-lmpc` @ `f751e7e`. 108 raw findings; **67 confirmed** after verification. This complements [CODEBASE.md](CODEBASE.md) §11 — findings already documented there are listed compactly at the end; the detail below is the **new** material.
>
> **Environment note:** these were checked after restoring the pinned `dev-dynamics` gym (see §9.5 of CODEBASE.md). That cleared every §11.1 *import blocker* — `import f1tenth_planning.control`, `f1tenth_params()`, and the `render_closed_lines`/`setData` render path all work on this gym. Those blockers were real for the `dev-humble` install only.

---

## 1. New bugs in active, in-scope code (control / utils / dynamics)

Ranked by severity. Line numbers verified against source unless marked *(scan)*.

### Silent-wrong (produces wrong numbers, no error)

- **MPPI & AP-MPPI clip the control lower bound to `-u_max`, not `u_min`.** [mppi_solver.py:127](f1tenth_planning/control/solvers/mppi_solver.py#L127), [ap_mppi_solver.py:148](f1tenth_planning/control/solvers/ap_mppi_solver.py#L148): `a = jnp.clip(a, -self.config.u_max, self.config.u_max)`. For **asymmetric** actuator limits (`u_min != -u_max`) the saturation floor is wrong. The f1tenth defaults are asymmetric: `MIN_ACCEL = -9.51` vs `MAX_ACCEL = 9.51` happen to match, but `MIN_DSTEER`/`MAX_DSTEER` and any custom limit will not. Masked today only because the truncated-normal sampler already keeps `a` in `[u_min, u_max]`, so the clip rarely binds — but it is the wrong bound.

- **Low-speed yaw-rate uses the state slip angle instead of the geometric one.** [dynamic_model.py:63](f1tenth_planning/control/dynamics_models/dynamic_model.py#L63) (numpy), `:208` (jax), `:288` (casadi) compute `dyaw = v·cos(slip_angle)·tan(delta)/L` using the **state** `slip_angle`. The gym/CommonRoad reference ([single_track.py:89,101]) uses `BETA_HAT = arctan(tan(delta)·lr/L)` — the *geometric* slip implied by steering, not the dynamic state. In the low-speed (kinematic) regime they diverge whenever the vehicle's actual slip differs from the steering-implied slip, so the model the solvers integrate disagrees with the simulator it controls.

- **`f(state, control, params)` permanently rebinds `self.params` on a one-off call.** [dynamic_model.py:46](f1tenth_planning/control/dynamics_models/dynamic_model.py#L46) and [kinematic_model.py:51](f1tenth_planning/control/dynamics_models/kinematic_model.py#L51): `if params is not None: self.params = params`. Passing a temporary parameter vector for a single derivative evaluation (as the LTV linearization and any parameter-sweep does) silently overwrites the model's stored config for every subsequent call. Combined with the shared-mutable-default pattern, cross-instance contamination is reachable.

- **`estimate()` mutates the caller's config object in place.** [nls_estimator.py:131](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L131): `self.params` is the same `DynamicsConfig` instance passed in; the update loop `setattr(self.params, name, …)` overwrites it, so any controller/model holding that reference sees its parameters change under it.

- **Gravity is emitted as an optimizable parameter but dropped on the return trip.** [dynamic_model.py:355](f1tenth_planning/control/dynamics_models/dynamic_model.py#L355): `parameters_vector_from_config` appends `g = 9.81` as entry 8 and `num_params` returns 9, so an estimator treating all `num_params` entries as free variables will "optimize gravity"; but `config_from_parameters_vector` reads only indices 0–7, so the fitted value is silently discarded (and index bookkeeping can desync).

### Latent (wrong under conditions not hit by the shipped examples)

- **Pure Pursuit reacquire path computes steering with the wrong lookahead distance.** [pure_pursuit.py:201](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L201): when `lookahead_distance ≤ nearest_dist < max_reacquire`, `_get_current_waypoint` returns the nearest raceline point (up to 20 m away), but `plan()` still calls `get_actuation` with the *nominal* `lookahead_distance` (e.g. 0.8 m). The curvature `2y/L²` then uses an L that doesn't match the actual target distance → wrong steering during reacquisition.

- **`intersect_point` wrap branch can return a negative segment index.** [utils.py:233](f1tenth_planning/utils/utils.py#L233): in `for i in range(-1, start_i)`, a hit sets `first_i = i` (can be `-1`) rather than `i % n`. Downstream, [pure_pursuit.py:209](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L209) uses it as `target_index`, so `waypoints[-1:9]` slices backwards and yields a near-empty/garbled local plan.

- **Kinematic `f_jax` declares `params=None` but dereferences `params[0]`.** [kinematic_model.py:117](f1tenth_planning/control/dynamics_models/kinematic_model.py#L117) → `params[0]` at `:131`. Calling with the declared default raises `TypeError`; the dynamic sibling guards this. (Also the shape bug from §11.2: `params[0]` is 1-D indexing of a `(1,1)` vector.)

- **`MPPISolver.samples` never initialized in `__init__`.** [mppi_solver.py:87](f1tenth_planning/control/solvers/mppi_solver.py#L87): only `APMPPISolver` sets `self.samples = None`; on the base MPPI, any access before the first successful `solve()` (e.g. a render callback, or after a first-solve exception) raises `AttributeError`.

### Crash (uncaught exception on a reachable path)

- **Classical `plan()` validators read `shape[1]` before the ndim check.** [pure_pursuit.py:174](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L174), [stanley.py:204](f1tenth_planning/control/controllers/stanley/stanley.py#L204), [lqr.py:244](f1tenth_planning/control/controllers/lqr/lqr.py#L244): `if waypoints.shape[1] < K or len(waypoints.shape) != 2:` — `or` short-circuits, so a 1-D `waypoints` argument raises `IndexError('tuple index out of range')` instead of the intended `ValueError`. (Only reachable if a caller passes custom waypoints, which the examples don't.)

### Minor

- **`nearest_point` casts to float32.** [utils.py:31](f1tenth_planning/utils/utils.py#L31): on maps in absolute coordinates (UTM ~1e6), float32's ~7 digits give ~0.05–0.1 m quantization in the projection, which can select the wrong nearest segment. Benign for Spielberg-scale local coordinates.
- **Renderer closes the loop only on first draw.** On `setData(points)` refresh (update path), the first point isn't re-appended, so the closing segment of the waypoint loop is dropped on every redraw. Cosmetic.

*Known-and-documented but worth re-flagging as confirmed:* the nonlinear MPC ignores `config.P` (terminal cost reuses `Q`, [nonlinear_mpc_solver.py:92](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L92)) and enforces no input-rate limits / `Rd` cost ([:101](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L101)) — see CODEBASE.md §4.5.

## 2. Why the 5 MPC/MPPI examples leave the track (task-2 diagnosis)

Not a code bug — expected for constraint-free tracking. All five accel-mode controllers exit the track at the **first sharp corner** (s≈35 m, radius 6.5 m) at ~10–12 % lap progress. The raceline reference there demands **8 m/s through a 6.5 m corner** = ~9.85 m/s² lateral ≈ the friction limit, with **no corner slowdown** and **no track-boundary/obstacle constraint** to keep the car in. The MPC's predicted steering is near-zero (the model believes it can hold the yaw-rate with `delta≈−0.02`), so it under-steers off. The classical controllers survive the same corner because they steer geometrically (Pure Pursuit), add curvature feedforward (LQR), or scale speed to 0.7 (Stanley). Contributing, pre-existing issues that make it worse: unconstrained inputs (kinematic LTV commanded 30.7 m/s² peak accel, >3× the 9.51 limit — CODEBASE.md §11.3) and the reference-velocity interpolation bug (§11.4).

## 3. New bugs in the planning subsystem (real, but scoped out)

Per the stated scope, `planning/` is an intentional gap; these are logged for whoever revives it, not for deletion. `LatticePlanner` and `LaneSwitcher` cannot run today — each fails at import and again at first call:

- **Stale import paths** — both import `from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner` ([lattice_planner.py:34](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L34), [lane_switcher.py:8](f1tenth_planning/planning/lane_switcher/lane_switcher.py#L8)); the real path is `control.controllers.pure_pursuit.pure_pursuit`. Module-level `ImportError`.
- **`LaneSwitcher.plan` uses `nearest_point` without importing it** ([lane_switcher.py:112](f1tenth_planning/planning/lane_switcher/lane_switcher.py#L112)) → `NameError`.
- **`LatticePlanner.eval` called with 1 arg, needs 2** (`cost_weights`) ([:200](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L200)) → `TypeError`.
- **`sample_lookahead_square` bugs** ([:247–259](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L247)): `lh_centers[i] = …` into an empty list (`IndexError`, should be `.append`); loop var `i` shadows the nearest-point index so `intersect_point` restarts from the wrong seed; `np.dot(2×2 rot, (N,3) grid)` shape mismatch; `np.repeat` vs `np.tile` misaligns widths with centers.
- **Cost functions reference undefined `trajectory_generator.NUM_STEPS` / `N_SHIFT` / `N_CULL`** ([:284–317](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L284)) → `NameError`.
- **`len(ego_ittc != 0)`** ([lane_switcher.py:214](f1tenth_planning/planning/lane_switcher/lane_switcher.py#L214)) tests array length, not nonzero count.

## 4. New bugs that live inside already-dead code

Real defects, but the enclosing function is itself unreachable (see [DEADCODE_FINDINGS.md](DEADCODE_FINDINGS.md)); fix only if reviving:

- **`calc_ref_trajectory_indices` index wrap uses one subtraction, not modulo** ([utils.py:82](f1tenth_planning/utils/utils.py#L82)) — indices exceeding `2·ncourse` stay out of bounds. Function is dead.
- **`sample_traj` stores curvature magnitude in the velocity column** ([utils.py:393](f1tenth_planning/utils/utils.py#L393)) — `sqrt(XDD²+YDD²)` for an arc-length clothoid is `|kappa|`, not `v`; the `[x,y,theta,v]` layout is mislabeled. Only consumer is the dead lattice sampler.
- **NLS feeds row-vector state slices to column-oriented CasADi dynamics** ([nls_estimator.py:63](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L63)) — `Xk[i,:]` is `(1,nx)` vs the `(nx,1)` the `ca.Function` expects; trips even after the known arg-order fix. `NLSParameterEstimator` is uninstantiable today.

## 5. Corrections to CODEBASE.md §11 (verification refuted these — for this environment)

- §11.1 blocker "**`SteerActionEnum` import**" and "**`f1tenth_params()` crashes at import**" — **cleared** by restoring the `dev-dynamics` gym. `f1tenth_params()` is defined `def f1tenth_params():` with no default arg (no shared-mutable-default here); the import succeeds. (Still true against a `dev-humble` install.)
- §11.1 "**renderer API stale**" — `render_closed_lines`/`setData` **exist and work** on the pinned gym; the first-render path is valid. Only the minor closing-segment-on-update issue (§1) remains.

## 6. Confirmed-known (already in CODEBASE.md §11 — validated, no new detail)

Verification re-confirmed these; see §11 for detail. Dynamic-model backend disagreements (switch threshold 0.1/1.5/1.5 vs gym; CasADi unsigned switch; `tan(delta)**2` low-speed `dbeta`; CasADi drops slip in dx/dy; no ε-guard on `1/v`); `config_from_parameters_vector` mutates by reference; `num_params` lists `MU` twice; `@jax.jit(static self)` recompiles per instance. Constraints: kinematic MPC unbounded; nonlinear-kmpc bounds-after-solver ordering; AP-MPPI clip nullifies its own constraints; MPPI bounds only `if config is None`. Crashes/silent: LTV `pred_x` `NameError` fallback; `SITLMPCConfig()` `TypeError`; `lax.scan` branch `TypeError` (both MPPI); `LQRConfig.__post_init__` overwrites all fields; PRNGKey reset each `solve`; `u_std` unused + `adaptive_covariance` computed-never-read; `Warning(...)` thrown away; no NMPC warm-start; IPOPT status unchecked; `ca.diagcat` drops off-diagonal `Q`; ref-velocity extrapolation; `dl` uniform-spacing assumption + yaw no-unwrap; `solve_lqr` inverted convergence; `update_matrix` mixed continuous/discrete; case-mismatched Pacejka keys; Stanley `atan2` on `(1,)` ndarray; `solve()` return-order docstrings inverted; `dynamic_mppi` dead raceline load. NLS: `estiamte` typo, missing `model` arg, `rk4` arg order, signed-residual objective, wrong param-vector length/order, infeasible positivity constraints.
