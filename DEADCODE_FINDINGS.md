# f1tenth_planning — Dead / Redundant / Stale Code

> Reachability scan (parallel analyzers → verification re-check), run 2026-07-20 against `dev-sit-lmpc` @ `f751e7e`. 60 candidates, **51 confirmed** after re-checking reference counts / md5 / diff. Complements [CODEBASE.md](CODEBASE.md) and [BUGS_FINDINGS.md](BUGS_FINDINGS.md).
>
> **Scope rule honored:** `planning/` is an intentional gap. Planning code is **not** flagged merely for being unwired — only where it is *truly* dead or redundant on its own terms (byte-identical duplicate files, unused imports). Everything else in `planning/` was deliberately left alone.

---

## 1. Safe to delete — zero reachable callers (verified repo-wide, excluding `.venv`)

### `utils/utils.py` — dead helpers
| Symbol | Line | Note |
|---|---|---|
| `calc_ref_trajectory_indices` | [:54](f1tenth_planning/utils/utils.py#L54) | 0 callers; also buggy (BUGS §4) |
| `quat_2_rpy` | [:349](f1tenth_planning/utils/utils.py#L349) | 0 callers |
| `map_collision` | [:398](f1tenth_planning/utils/utils.py#L398) | body is `pass` |
| `input_acceleration_to_speed` | [:405](f1tenth_planning/utils/utils.py#L405) | 0 callers |
| `input_steering_speed_to_angle` | [:412](f1tenth_planning/utils/utils.py#L412) | 0 callers |

### Solvers / config / estimation — dead symbols
| Symbol | File:line | Note |
|---|---|---|
| `truncated_gaussian_sampler` | [mppi_solver.py:19](f1tenth_planning/control/solvers/mppi_solver.py#L19) | 0 callers; itself wrong (whitens with `diag(R)`, un-whitens with full `R`) |
| `LTVMPCSolver.predict_state` | [LTV_mpc_solver.py:240](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L240) | never called |
| `update_config_from_dict` | [dynamics_config.py:223](f1tenth_planning/control/config/dynamics_config.py#L223) | module-level fn taking `self`, never bound, 0 callers |
| `f1fifth_params` / `fullscale_params` | [dynamics_config.py:199](f1tenth_planning/control/config/dynamics_config.py#L199)/[:211](f1tenth_planning/control/config/dynamics_config.py#L211) | 0 callers (keep if intended as public presets — otherwise dead) |
| `ModelConfig` | [model_config.py:5](f1tenth_planning/control/config/model_config.py#L5) | consumed by nothing (composed into the dead `SITLMPCConfig`) |
| `SITLMPCConfig` | [controller_config.py:262](f1tenth_planning/control/config/controller_config.py#L262) | 0 callers; also crashes if constructed (`default_factory=APMPPIConfig`) |
| `NLSParameterEstimator` / `estimate` | [nls_estimator.py:14](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L14) | orphaned + uninstantiable (`estiamte` typo) |

### Dead branches / unreachable code
| Where | Line | Note |
|---|---|---|
| `DynamicBicycleModel.linearize_around_state` | [dynamic_model.py:405](f1tenth_planning/control/dynamics_models/dynamic_model.py#L405) | ~62 lines after `raise NotImplementedError` |
| trailing `pass` in getters | [dynamic_model.py:443](f1tenth_planning/control/dynamics_models/dynamic_model.py#L443) | unreachable `pass` statements |
| `MPPISolver`/`APMPPISolver` `lax.scan` branch | [mppi_solver.py:272](f1tenth_planning/control/solvers/mppi_solver.py#L272), [ap_mppi_solver.py:382](f1tenth_planning/control/solvers/ap_mppi_solver.py#L382) | never runs (`scan=False` in factories) — and crashes if reached |
| `LTVMPCSolver.solve` OSQP-fail fallback | [LTV_mpc_solver.py:234](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L234) | `NameError` on `pred_x`; dead unless OSQP fails |
| `LTVMPCSolver.update` cost-matrix path | [:172](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L172) | raises then has unreachable assignments |
| `NonlinearMPCSolver.solve` `Warning(...)` | [nonlinear_mpc_solver.py:175](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L175) | constructs+discards an exception, emits nothing |

### Stale attributes / imports
- `NonlinearMPCSolver.U0` [nonlinear_mpc_solver.py:127](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L127) — allocated, never written (advertised warm-start does nothing).
- `ParameterEstimator.estiamte` [paramter_estimator.py:23](f1tenth_planning/estimation/estimators/parameter_estimators/paramter_estimator.py#L23) — misspelled abstract method (also a bug).
- `__all__ = [… "ITLMPCPlanner" …]` [lmpc/__init__.py:47](f1tenth_planning/control/controllers/lmpc/__init__.py#L47) — phantom name; omits the real `SITLMPCPlanner`.
- `adaptive_covariance` / `a_cov` [mppi_solver.py:137](f1tenth_planning/control/solvers/mppi_solver.py#L137) — computed, shifted, stored, never read by the sampler.
- Unused imports in [dynamic_ap_mppi.py](f1tenth_planning/control/controllers/mpc/ap_mppi/dynamic_ap_mppi.py): `os` (:12), `Track` (:19), `List`/`Callable` (:1).
- `DynamicMPPIPlanner.render_sampled_trajectories` [dynamic_mppi.py:96](f1tenth_planning/control/controllers/mpc/mppi/dynamic_mppi.py#L96) — defined, registered by no example (and buggy).

### Commented-out code
- The `lmpc` import block [control/__init__.py:16](f1tenth_planning/control/__init__.py#L16).
- `@njit(cache=True)` on `sample_traj` [utils.py:385](f1tenth_planning/utils/utils.py#L385) (commented out).
- ROS wrapper logging [control_ros_wrapper.py:195](examples/ros_wrappers/control_ros_wrapper.py#L195).

## 2. Consolidate — duplicated / redundant code (drifted copies noted)

- **`render_local_plan` + `render_control_solution` are copy-pasted across all MPC controllers and re-implemented per classical controller** (e.g. [lqr.py:87](f1tenth_planning/control/controllers/lqr/lqr.py#L87)/[:101](f1tenth_planning/control/controllers/lqr/lqr.py#L101)). Lift onto a shared base/mixin. **These are not dead** — the examples register them — so consolidate, don't delete.
- **`APMPPISolver` duplicates ~90 % of `MPPISolver`** ([ap_mppi_solver.py:19](f1tenth_planning/control/solvers/ap_mppi_solver.py#L19)) — the two have already drifted (e.g. `samples` init, rollout clipping). Factor the shared sampling/rollout/weighting core.
- **`dynamic_ap_mppi_config` duplicates `dynamic_mppi_config`'s cost numbers verbatim** ([controller_config.py:242](f1tenth_planning/control/config/controller_config.py#L242)) rather than deriving from it — they will silently drift on any retune.
- **`DynamicAPMPPIPlanner.__init__` duplicates the state/control-bounds block** from `DynamicMPPIPlanner` ([dynamic_ap_mppi.py:75](f1tenth_planning/control/controllers/mpc/ap_mppi/dynamic_ap_mppi.py#L75)).
- **`LTVMPCSolver.__init__` redundant self-assignments** ([LTV_mpc_solver.py:35](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L35)).
- **`LTVMPCSolver` cvxpy `Q/R/Rd/P` Parameters** ([:54](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L54)) — declared, never assigned a value, never used in the objective (which hardcodes the config blocks). This is the structural reason custom cost matrices are "unsupported"; either wire them up or drop them.
- **Namespace pollution:** [config/__init__.py](f1tenth_planning/control/config/__init__.py) star-imports with no `__all__`, and [utils.py](f1tenth_planning/utils/utils.py) is re-exported via `from .utils import *` but every consumer imports the full path — the star-export leaks `np/jax/math/njit`.

## 3. Planning — only the *truly* dead/redundant (rest deliberately kept)

The planning **algorithms** are an intentional gap and were not flagged. What *is* genuinely dead/redundant:

- **`fgm/fgm.py` and `wall_follow/wall_follow.py` contain no FGM / wall-following code** — both are a `pyclothoids` grid benchmark, **byte-identical** to each other (md5 `130a1954…`) and to an old copy of `lattice_planner/test_pyclothoids.py`. These are misplaced duplicates, not stubs of the intended algorithms. Safe to delete or replace with real stubs.
- **`lattice_planner/test_pyclothoids.py`** [test_pyclothoids.py](f1tenth_planning/planning/lattice_planner/test_pyclothoids.py) — a benchmark script that **ships inside the wheel**, matches pytest discovery, has zero assertions, and calls blocking `plt.show()`. Move out of the package or delete.
- Unused imports in the kept planning files: `matplotlib.pyplot` [wall_follow.py:4](f1tenth_planning/planning/wall_follow/wall_follow.py#L4), `cProfile` [test_pyclothoids.py:3](f1tenth_planning/planning/lattice_planner/test_pyclothoids.py#L3); commented debug `ipdb` block [lane_switcher.py:120](f1tenth_planning/planning/lane_switcher/lane_switcher.py#L120).

## 4. Dependencies

**Declared but never imported** (all transitive deps of cvxpy/osqp/numba — pinning is redundant): `osqp`, `scs`, `qdldl`, `kiwisolver`, `llvmlite` ([pyproject.toml:17-23](pyproject.toml#L17)).

**Imported but not declared** (survive only transitively / not at all): `scipy`, `matplotlib`, `pyclothoids` ([pyproject.toml:15](pyproject.toml#L15)). `pyclothoids` in particular is not installed at all → all of `planning/` is import-dead on a clean install.
