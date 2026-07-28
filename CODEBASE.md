# f1tenth_planning — Codebase Reference

> **PARTLY SUPERSEDED — read [DESIGN.md](DESIGN.md) alongside this.**
>
> This describes the codebase as it stood on `dev-sit-lmpc` (`f751e7e`), *before* the
> API redesign. It is still the best reference for the algorithms, the dynamics
> equations, the solver formulations and the gym interface — none of which changed in
> substance. The **API sections are out of date** on `dev-api-redesign`:
>
> | Described here | Actually now |
> |---|---|
> | `controller.plan(state, waypoints=...)` | `controller.compute_control(state)`; the reference is set via `controller.update(reference=...)` |
> | controllers own `render_*` methods | rendering lives with the caller (`examples/control/render_helpers.py`) |
> | `pre_processing_fn` bridges 7-state to 5-state | models declare `STATE_NAMES`; `model.state.idx.v` replaces hardcoded indices |
> | MPPI kernels read config off a static `self` | module-level kernels with `static_argnames`; tuning values are traced and changeable at runtime |
> | `APMPPISolver` duplicates `MPPISolver` | it inherits from it |
> | planning modules unimportable | `LatticePlanner`/`LaneSwitcher` are `Planner`s; FGM/wall-follow are `Controller` stubs |
> | no first-party tests | 77 tests plus CI (`tests/`, `.github/workflows/ci.yml`) |
>
> The §11 bug catalogue is likewise historical — many entries are now fixed; see the
> branch history and [BUGS_FINDINGS.md](BUGS_FINDINGS.md).

> Canonical orientation doc for this repo. Written against branch `dev-sit-lmpc` (HEAD `f751e7e`). Line numbers are accurate as of that commit; treat them as strong hints, not guarantees, after any refactor.

---

## 1. What this repo is

`f1tenth_planning` is a Python library of motion-planning and control algorithms for the [F1TENTH](https://f1tenth.org) autonomous racing platform, designed to be driven against the [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) simulator. Despite the name, it is **overwhelmingly a control library**: [f1tenth_planning/control/](f1tenth_planning/control/) contains eight working controllers spanning geometric tracking (Pure Pursuit, Stanley), linear optimal control (LQR), and four flavors of MPC (LTV/cvxpy, nonlinear/CasADi, MPPI/JAX, AP-MPPI/JAX), while [f1tenth_planning/planning/](f1tenth_planning/planning/) holds two real-but-broken planners and two placeholder files that contain no planning code at all. The library never simulates; it consumes an `f1tenth_gym` `Track` for waypoints, a `VehicleParameters`/param-dict for physics, and a per-step observation dict, and returns a 2-element action. The relationship to the gym is a **hard, tight, and currently broken** coupling — see [§9](#9-the-f1tenth_gym-interface) and [§11](#11-gotchas-drift-and-known-rough-edges).

**Read this first:** `import f1tenth_planning.control` currently raises `ImportError: cannot import name 'SteerActionEnum' from 'f1tenth_gym.envs.action'`. Every controller and every example is unimportable in the current environment. This is a rename drift (`*ActionEnum` → `*ActionType`), not a design problem, but nothing in this repo runs until it is fixed.

---

## 2. Orientation / repo layout

```
f1tenth_planning/                      # FIRST-PARTY package (the library)
├── __init__.py                        # docstring + __version__ = "0.1.1". Imports nothing — this is why
│                                      #   a naive `import f1tenth_planning` smoke test passes while everything is broken.
├── control/                           # The real library. 8 controllers, 4 solvers, 2 models.
│   ├── __init__.py                    # THE public API. Exports 8 controllers; LMPC block commented out (TODO).
│   ├── controller.py                  # Controller ABC — plan(state) -> action. Has the poisoned gym import.
│   ├── dynamics_model.py              # DynamicsModel ABC — multi-backend xdot = f(x,u,p).
│   ├── mpc_solver.py                  # MPCSolver ABC — update()/solve() over (MPCConfig, DynamicsModel).
│   ├── discretizers.py                # Free fns: euler, rk4, system_matrix_discretization (euler|exact ZOH).
│   ├── config/
│   │   ├── controller_config.py       # MPCConfig -> MPPIConfig -> APMPPIConfig; LQRConfig, LMPCConfig, SITLMPCConfig + factories.
│   │   ├── dynamics_config.py         # DynamicsConfig (23 required fields) + f1tenth/f1fifth/fullscale_params().
│   │   └── model_config.py            # ModelConfig (NN value-function hyperparams). DEAD — nothing reads it.
│   ├── dynamics_models/               # __init__.py is EMPTY — import by full path.
│   │   ├── kinematic_model.py         # KinematicBicycleModel (nx=5) + _extract_kinematic_state hook.
│   │   └── dynamic_model.py           # DynamicBicycleModel (nx=7, CommonRoad single-track).
│   ├── solvers/
│   │   ├── LTV_mpc_solver.py          # LTVMPCSolver — cvxpy -> OSQP, sparse-Parameter trick, warm start.
│   │   ├── nonlinear_mpc_solver.py    # NonlinearMPCSolver — CasADi SX multiple shooting -> IPOPT.
│   │   ├── mppi_solver.py             # MPPISolver — JAX, port of google-research/jax_mpc.
│   │   └── ap_mppi_solver.py          # APMPPISolver — MPPI + Adaptive-Penalty lambda sampling. Active dev front.
│   └── controllers/
│       ├── pure_pursuit/ stanley/ lqr/   # Classical. All three __init__.py are EMPTY.
│       ├── mpc/
│       │   ├── mpc.py                 # MPCController — the generic composition root for the whole MPC family.
│       │   ├── LTV_mpc/               # KinematicMPCPlanner
│       │   ├── nonlinear_mpc/         # NonlinearKinematicMPCPlanner, NonlinearDynamicMPCPlanner. NO __init__.py.
│       │   ├── mppi/                  # DynamicMPPIPlanner
│       │   └── ap_mppi/               # DynamicAPMPPIPlanner + constraint factories
│       └── lmpc/__init__.py           # SITLMPCPlanner. BROKEN — base/components/manager were deleted.
├── planning/                          # Abandoned. __init__.py and all 4 sub-__init__.py are 0 bytes.
│   ├── fgm/fgm.py                     # NOT follow-the-gap. A pyclothoids benchmark.
│   ├── wall_follow/wall_follow.py     # NOT wall following. Byte-identical to fgm.py (md5 130a1954…).
│   ├── lattice_planner/               # Real clothoid lattice planner. Broken imports + 6 runtime bugs.
│   └── lane_switcher/                 # Real multi-lane overtaking planner. Broken imports + NameError.
├── estimation/                        # Orphaned. Nothing imports it; its own __init__.py ImportErrors.
│   └── estimators/parameter_estimators/{paramter_estimator.py, NLS/nls_estimator.py}
└── utils/utils.py                     # Leaf module. All njit kernels + reference-traj interpolation + jnp_to_np.

examples/
├── control/                           # 9 scripts, one per controller. THE de-facto documentation. All currently fail at import.
└── ros_wrappers/control_ros_wrapper.py  # ROS 2 node. Targets a dead 2-value plan() API. Broken.

docs/                                  # Sphinx skeleton. All 10 content pages are 2-line stubs. No autodoc.
pyproject.toml / uv.lock               # hatchling + uv. f1tenth_gym is a git dep (branch dev-dynamics).

f1tenth_gym/                           # ⚠️ VENDORED, UNTRACKED (`?? f1tenth_gym/` in git status), NOT in .gitignore.
                                       #    NOT a submodule. NOT what Python imports. Editing it has zero effect.
                                       #    Its tests/ + .github/ are the only test suite and CI in the tree.
```

**Searching:** because the untracked gym clone sits inside the repo root, `grep -r` from the root double-counts every symbol. Use `--exclude-dir=f1tenth_gym` when auditing first-party code.

---

## 3. Core architecture

This is the most important section. The design is **three ABCs plus a two-axis config split**, and the MPC family is built by composition, not inheritance.

### 3.1 The three abstractions

| ABC | File | Contract |
|---|---|---|
| `Controller` | [f1tenth_planning/control/controller.py:9](f1tenth_planning/control/controller.py#L9) | Policy layer. `__init__(track, params, control_mode)` → `plan(state: dict, waypoints=None, **kwargs) -> np.ndarray` |
| `DynamicsModel` | [f1tenth_planning/control/dynamics_model.py:10](f1tenth_planning/control/dynamics_model.py#L10) | Continuous-time `xdot = f(x, u, p)`, exposed in **four backends** + Jacobians + config↔vector round-trip |
| `MPCSolver` | [f1tenth_planning/control/mpc_solver.py:7](f1tenth_planning/control/mpc_solver.py#L7) | Optimization layer. `__init__(config: MPCConfig, model: DynamicsModel)`, `update(...)`, `solve(x0, ref_traj, ...) -> (u_opt, x_opt)` |

`MPCSolver.__init__(self, config, model)` at [mpc_solver.py:13](f1tenth_planning/control/mpc_solver.py#L13) is **the composition seam**: it is what makes any solver usable with any model.

### 3.2 The composition pattern (MPC family)

`MPCController` ([controllers/mpc/mpc.py:15](f1tenth_planning/control/controllers/mpc/mpc.py#L15)) is a single generic controller that never touches an optimizer. Its constructor is:

```python
MPCController(track: Track,
              solver: MPCSolver,
              model: DynamicsModel,
              params: DynamicsConfig = f1tenth_params(),
              pre_processing_fn=None,
              ref_velocity_bounds=None)     # mpc.py:34-42
```

It owns exactly four things: (1) building the 7-column waypoint matrix from `track.raceline` ([mpc.py:48-58](f1tenth_planning/control/controllers/mpc/mpc.py#L48)), (2) assembling `x0` from the obs dict ([mpc.py:169](f1tenth_planning/control/controllers/mpc/mpc.py#L169)), (3) generating the reference trajectory and clipping its velocity ([mpc.py:178-190](f1tenth_planning/control/controllers/mpc/mpc.py#L178)), and (4) the pyqtgraph render callbacks. Then it calls `solver.solve(x0, ref_traj, p=..., Q=..., R=...)` ([mpc.py:199](f1tenth_planning/control/controllers/mpc/mpc.py#L199)) and returns `u_pred[:, 0]`.

Swapping LTV ↔ nonlinear ↔ MPPI is **a constructor argument**. There is a *third* injectable seam: the discretizer, passed as a callable — `LTVMPCSolver(..., discretizer=euler_discretization, dynamics_discretizer=system_matrix_discretization)` ([LTV_mpc_solver.py:19-25](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L19)), `NonlinearMPCSolver(..., discretizer=rk4_discretization)` ([nonlinear_mpc_solver.py:21-27](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L21)).

The **convenience planners** (`KinematicMPCPlanner`, `NonlinearKinematicMPCPlanner`, `NonlinearDynamicMPCPlanner`, `DynamicMPPIPlanner`, `DynamicAPMPPIPlanner`) are thin subclasses of `MPCController` that all follow the same lazy-default ladder — `params → model → config → solver`, each guarded by `if X is None` — so any rung can be overridden by the caller. That ladder is the intended extension point.

**Dimension bridging.** `MPCController` always builds a 7-dim `x0`, but kinematic models want 5. This is solved by a hook, not by subclassing: `pre_processing_fn(x0, xref)` is applied at [mpc.py:196-197](f1tenth_planning/control/controllers/mpc/mpc.py#L196), and the shipped one is `_extract_kinematic_state(x0, xref) -> (x0[:5], xref[:5, :])` ([kinematic_model.py:9-13](f1tenth_planning/control/dynamics_models/kinematic_model.py#L9)).

### 3.2.1 Exercising the seam (direct composition, the ladder, extending)

`MPCController` is **concrete** — you never need a subclass; the convenience planners are sugar. Direct composition (copy-pasteable):

```python
params = f1tenth_params()
model  = KinematicBicycleModel(params)          # nx=5, nu=2
config = kinematic_mpc_config()                 # bounds default to ±inf
# Bounds MUST be set before the solver is built (the solver bakes them at __init__).
config.x_min = np.array([-np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED, -np.inf])
config.x_max = np.array([ np.inf,  np.inf, params.MAX_STEER, params.MAX_SPEED,  np.inf])
config.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
config.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])
solver = LTVMPCSolver(config=config, model=model)
controller = MPCController(track=track, solver=solver, model=model, params=params,
                           pre_processing_fn=_extract_kinematic_state)
```

Swap `LTVMPCSolver` → `NonlinearMPCSolver` and nothing else changes (identical `(config, model)` ctor, identical `solve` contract). Swap to `DynamicBicycleModel` + `dynamic_mpc_config()` and you **drop `pre_processing_fn`** instead — that is the whole seam. **`pre_processing_fn` is load-bearing, not polish:** `MPCController.plan` always builds a 7-state `x0` and 7-column waypoints; any model whose state is not a prefix of `[x, y, delta, v, yaw, yaw_rate, beta]` needs its own hook. **Index 3 must be velocity** — with `ref_velocity_bounds=None` the controller reads `config.x_min[3]`/`x_max[3]` as the reference-velocity clip.

**The lazy-default ladder** in every convenience planner runs `params → model → config (+bounds) → solver`, each guarded by `if X is None`, and the order is a **dependency chain** — each rung consumes the one above:

| Override | Buys you | Costs you |
|---|---|---|
| `params` | Retuned vehicle; bounds re-derive automatically | Nothing — cheapest override |
| `model` | New dynamics | You now own `params`↔`model` consistency |
| `config` | Horizon, costs, `n_samples`, **and all bounds** — passing a config **skips the entire bounds block**; you inherit `±inf` unless you set them | Must set `x_min/x_max/u_min/u_max` yourself |
| `solver` | Different algorithm; `config`/`model` become inert for construction | Everything above is yours |

**To add a controller:** subclass `MPCController` with signature `(track, params=None, model=None, config=None, solver=None, pre_processing_fn=None)`, implement the ladder in order, then `super().__init__(track, solver, model, params, pre_processing_fn)` — note the super-call puts **`solver` before `model`**, the reverse of the ladder's own signature ([mpc.py:34-41](f1tenth_planning/control/controllers/mpc/mpc.py#L34)). Hand-copy `render_local_plan`/`render_control_solution` (they live on `MPCController`, not the ABC — a controller off the ABC breaks all four examples). Export it in [control/\_\_init\_\_.py](f1tenth_planning/control/__init__.py), honoring the alias convention. **Do NOT copy `nonlinear_kmpc.py` as your template** — it sets bounds *after* the solver and inside `if pre_processing_fn is None` (both wrong); use `dynamic_mppi.py`/`dynamic_ap_mppi.py`. **To add a solver:** subclass `MPCSolver`, implement `__init__(config, model)`, `update(...)`, `solve(x0, ref_traj, p=None, Q=None, R=None)` — and **return `(x, u)`**, not the `(u, x)` the ABC docstring wrongly claims (§3.5, §11.4).

### 3.3 How the classical controllers differ

Pure Pursuit, Stanley and LQR subclass `Controller` **directly**. They have no solver, no `DynamicsModel`, no `MPCConfig`. They read `self.params.WHEELBASE` and nothing else out of `DynamicsConfig`, build their own waypoint matrix in `__init__` by `np.vstack([...]).T` from the raceline, and delegate their hot math to `@njit(cache=True)` kernels in [f1tenth_planning/utils/utils.py](f1tenth_planning/utils/utils.py). Config style is inconsistent across the three: LQR takes an `LQRConfig` dataclass; Pure Pursuit and Stanley take bare scalars (`lookahead_distance`, `max_reacquire`, `k_path`). There is no `PurePursuitConfig`/`StanleyConfig`.

Only one render hook is inherited: `Controller.render_waypoints(e)` ([controller.py:58](f1tenth_planning/control/controller.py#L58)), plus the `waypoints_color` property that configures it (`@property` [:42](f1tenth_planning/control/controller.py#L42), setter [:51](f1tenth_planning/control/controller.py#L51), backed by `self._waypoints_color`, default `(0, 128, 0)`). `render_local_plan` and `render_control_solution` are **not on the ABC** — every controller re-implements both by copy-paste convention, with near-identical bodies. The example scripts register all three unconditionally, so a new controller that omits them breaks the examples silently. Note `render_waypoints` reads the color **only on the first call** (later calls take the `setData` fast path), so `waypoints_color` must be set before the first render; it is the intended per-agent trajectory-coloring hook (the whole subject of HEAD `f751e7e`, which fixed a previously-inert setter) but no example currently exercises it. The setter asserts `len(value) == 3` only — no range/type check.

### 3.4 The config system — two axes that never merge

```
DynamicsConfig          ← vehicle PHYSICS. Shared by Controller AND DynamicsModel. 23 required fields, no defaults.
    │
    ├── f1tenth_params() / f1fifth_params() / fullscale_params()   # dynamics_config.py:187 / :199 / :211
    └── _dynamics_config_from_gym_params(gym_params)               # dynamics_config.py:60 — the sole gym→lib translation point

MPCConfig               ← ALGORITHM tuning. Only the solver sees it.
    └── MPPIConfig      (+ n_iterations, n_samples, temperature, damping, u_std, scan, adaptive_covariance)
        └── APMPPIConfig(+ n_lambdas, lambdas_sample_range, constraints, n_constraints)

LQRConfig  /  LMPCConfig  /  ModelConfig        ← standalone
SITLMPCConfig  = LMPCConfig + APMPPIConfig + ModelConfig  (composition root, controller_config.py:261-269 — DEAD)
```

**Preset factories.** [controller_config.py](f1tenth_planning/control/config/controller_config.py) ships four zero/low-arg factories, one per model. **Every one returns a bounds-free config** — `MPCConfig.__post_init__` defaults all six bound arrays to `±inf`; setting bounds is the *planner's* job (see [§11.3](#113-constraints-that-arent-enforced)):

| Factory | Line | Returns | State | Consumed by |
|---|---|---|---|---|
| `kinematic_mpc_config()` | [:103](f1tenth_planning/control/config/controller_config.py#L103) | `MPCConfig` | `nx=5` | `nonlinear_kmpc.py`, `LTV_kinematic_mpc.py` |
| `dynamic_mpc_config()` | [:117](f1tenth_planning/control/config/controller_config.py#L117) | `MPCConfig` | `nx=7` | `nonlinear_dmpc.py` |
| `dynamic_mppi_config()` | [:131](f1tenth_planning/control/config/controller_config.py#L131) | `MPPIConfig` | `nx=7` | `dynamic_mppi.py` |
| `dynamic_ap_mppi_config(constraints=None, lambdas_sample_range=None, n_lambdas=16)` | [:222](f1tenth_planning/control/config/controller_config.py#L222) | `APMPPIConfig` | `nx=7` | `dynamic_ap_mppi.py` |

All four use `nu=2`, `dt=0.1`. The two MPPI factories are **cost-identical** (`N=10`, `n_iterations=2`, `n_samples=1024`, `scan=False`); AP-MPPI adds only lambda/constraint fields. The `SITLMPCConfig`/bare-`APMPPIConfig` constructors take 8 required positional args — always go through `dynamic_ap_mppi_config()`, never `APMPPIConfig()` (that `TypeError`s; see [§11.4](#114-verified-crashes-and-silent-wrongness)).

`MPCConfig` ([controller_config.py:9](f1tenth_planning/control/config/controller_config.py#L9)) requires `N, dt, nx, nu, Q, R, Rd, P` positionally with **no defaults**, and its `__post_init__` ([controller_config.py:46-71](f1tenth_planning/control/config/controller_config.py#L46)) fills the six optional bound fields (`x_min/x_max/u_min/u_max/ud_min/ud_max`) with `±inf` arrays and then asserts shapes: `Q`, `P` are `(nx, nx)`; `R`, `Rd` are `(nu, nu)`; bounds are **1-D** `(nx,)` / `(nu,)`. The `±inf` defaulting is deliberate — solvers can always read `config.x_min` without a `None` check — but it means an unconstrained problem is the *silent* default (see [§11](#11-gotchas-drift-and-known-rough-edges)).

Every subclass in the chain adds only defaulted fields (so the parent's required positional fields stay first) and every `__post_init__` calls `super().__post_init__()` first.

**Parameter round-tripping** (`parameters_vector_from_config` / `config_from_parameters_vector` / `num_params`) exists so a `DynamicsConfig` can be flattened to an `(num_params, 1)` vector and threaded through CasADi/JAX **functionally**, as a solver parameter rather than as `self.params`. That is what makes an MPC problem parameterized over vehicle physics — the NLP is built once and re-solved with new `p` — and it is the hook AP-MPPI's parameter-adaptation path would use.

### 3.5 Shape contracts

| Object | Shape | Notes |
|---|---|---|
| `x0` | `(nx,)` | time-invariant |
| reference trajectory | `(nx, N+1)` | **time along axis 1** |
| `solve()` returns | `(x (nx, N+1), u (nu, N))` | **All four solvers return `x` first** — [LTV_mpc_solver.py:226](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L226), [nonlinear_mpc_solver.py:222](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L222), [mppi_solver.py:300](f1tenth_planning/control/solvers/mppi_solver.py#L300), [ap_mppi_solver.py:411](f1tenth_planning/control/solvers/ap_mppi_solver.py#L411). `MPCController` unpacks `self.x_pred, self.u_pred = solver.solve(...)` ([mpc.py:199](f1tenth_planning/control/controllers/mpc/mpc.py#L199)). **The ABC and 3-of-4 docstrings claim the reverse `(u, x)` and are wrong** — see [§11.4](#114-verified-crashes-and-silent-wrongness). |
| `calc_interpolated_reference_trajectory` returns | `(N+1, ncols)` | **the exception** — hence `.T.copy()` at [mpc.py:190](f1tenth_planning/control/controllers/mpc/mpc.py#L190) |
| MPPI internals | `a/da [n_samples, N, nu]`, `s [n_samples, N, nx]`, `a_cov [N, nu, nu]` | shape comments in the solvers are reliable; docstrings are not |

---

## 4. The controller catalog

| Name | Class (exported name) | File | Backend | Status |
|---|---|---|---|---|
| Pure Pursuit | `PurePursuitPlanner` | [pure_pursuit.py:23](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L23) | numba njit | Working (import-blocked); several latent bugs |
| Stanley | `StanleyController` | [stanley.py:41](f1tenth_planning/control/controllers/stanley/stanley.py#L41) | numba njit | Working (import-blocked) |
| LQR | `LQRController` | [lqr.py:45](f1tenth_planning/control/controllers/lqr/lqr.py#L45) | numba njit (Riccati) | Working; **config args are inert** |
| LTV Kinematic MPC | `KinematicMPCPlanner` | [LTV_kinematic_mpc.py:18](f1tenth_planning/control/controllers/mpc/LTV_mpc/LTV_kinematic_mpc.py#L18) | cvxpy → OSQP | Working; **fully unconstrained** |
| Nonlinear Kinematic MPC | `NonlinearKinematicMPCPlanner` | [nonlinear_kmpc.py:19](f1tenth_planning/control/controllers/mpc/nonlinear_mpc/nonlinear_kmpc.py#L19) | CasADi → IPOPT | Working; **bounds never applied** (ordering bug) |
| Nonlinear Dynamic MPC | `NonlinearDynamicMPCPlanner` | [nonlinear_dmpc.py:16](f1tenth_planning/control/controllers/mpc/nonlinear_mpc/nonlinear_dmpc.py#L16) | CasADi → IPOPT | Working |
| Dynamic MPPI | `DynamicMPPIPlanner` (as `NonlinearDynamicMPPIPlanner`) | [dynamic_mppi.py:17](f1tenth_planning/control/controllers/mpc/mppi/dynamic_mppi.py#L17) | JAX | Working |
| Dynamic AP-MPPI | `DynamicAPMPPIPlanner` (as `NonlinearDynamicAPMPPIPlanner`) | [dynamic_ap_mppi.py:96](f1tenth_planning/control/controllers/mpc/ap_mppi/dynamic_ap_mppi.py#L96) | JAX | Working; **active development front** |
| SIT-LMPC | `SITLMPCPlanner` | [lmpc/\_\_init\_\_.py:15](f1tenth_planning/control/controllers/lmpc/__init__.py#L15) | `APMPPISolver` | **BROKEN** — `base/components/manager` deleted; export commented out |
| Lattice | `LatticePlanner` | [lattice_planner.py:41](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L41) | numba + pyclothoids | **Dead** — stale imports, 6 runtime bugs, not exported |
| Lane Switcher | `LaneSwitcher` | [lane_switcher.py:11](f1tenth_planning/planning/lane_switcher/lane_switcher.py#L11) | scipy cdist | **Dead** — stale import, `NameError` on `nearest_point` |
| FGM / Wall Follow | — | [fgm.py](f1tenth_planning/planning/fgm/fgm.py), [wall_follow.py](f1tenth_planning/planning/wall_follow/wall_follow.py) | — | **Do not exist**. Both files are the same clothoid benchmark. |

Naming note: two classes are exported under **aliases** ([control/\_\_init\_\_.py:9-14](f1tenth_planning/control/__init__.py#L9)) — the public name (`NonlinearDynamic*`) does not appear in the defining module, so grepping the exported name finds nothing. Also note `KinematicMPCPlanner` does **not** say "LTV" even though that is exactly what distinguishes it from `NonlinearKinematicMPCPlanner`. The `Controller` vs `Planner` suffix split follows no rule; all subclass `Controller` and all implement `plan()`.

### 4.1 Pure Pursuit — [pure_pursuit.py](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py)

Waypoints `Nx4 = [x, y, v, yaw]`. `_get_current_waypoint(lookahead_distance, position, theta)` ([:116](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L116)) is a three-way branch: if `nearest_dist < L`, run `intersect_point(position, L, wps[:, :2], t=i+t, wrap=True)` to walk forward to the first circle/polyline intersection; elif `nearest_dist < max_reacquire` (20 m), fall back to the raw nearest row; else return `None` → `plan()` warns and returns `(0.0, 0.0)`. Steering comes from `get_actuation` ([utils.py:245](f1tenth_planning/utils/utils.py#L245)), which projects the lookahead point into the body frame, computes `radius = 1/(2y/L²)` and `steering = arctan(wheelbase/radius)`, guarding `|y| < 1e-6`. **It is the only one of the three that handles lap wraparound in its local-plan slicing.**

### 4.2 Stanley — [stanley.py](f1tenth_planning/control/controllers/stanley/stanley.py)

`calc_theta_and_ef` ([:117](f1tenth_planning/control/controllers/stanley/stanley.py#L117)) projects the pose to the front axle (`fx = x + WHEELBASE*cos(theta)`), computes signed cross-track error as the dot of `(front_axle − nearest_point)` with the body y-axis, and heading error as `pi_2_pi(theta_raceline − theta)`. Control law at [:175](f1tenth_planning/control/controllers/stanley/stanley.py#L175): `delta = atan2(k_path * ef, v) + theta_e`. Speed is the raceline velocity at the nearest index — no feedback.

### 4.3 LQR — [lqr.py](f1tenth_planning/control/controllers/lqr/lqr.py)

Ported from AtsushiSakai/PythonRobotics `lqr_steer_control` (credited at [lqr.py:25](f1tenth_planning/control/controllers/lqr/lqr.py#L25)). Requires `Nx5` waypoints including curvature (`track.raceline.ks`). Builds the 4-state error vector `[e_cg, ė_cg, theta_e, θ̇_e]` by finite-differencing the previous step's cached errors ([:207-210](f1tenth_planning/control/controllers/lqr/lqr.py#L207)), gets `Ad/Bd` from `update_matrix`, solves the discrete Riccati fixed point via `solve_lqr` **every tick** (no caching, up to 50 iterations with two `pinv` calls each), and forms `steer = (K @ x)[0][0] + kappa_ref * WHEELBASE` — the second term is the kinematic curvature feedforward. `state_size` is hardcoded to 4 at [:185](f1tenth_planning/control/controllers/lqr/lqr.py#L185).

### 4.4 LTV MPC — [LTV_mpc_solver.py](f1tenth_planning/control/solvers/LTV_mpc_solver.py)

Non-condensed / multiple-shooting sparse QP with decision vars `xk (nx, N+1)`, `uk (nu, N)`. Objective = `vec(u)' R_blk vec(u) + vec(x−xref)' Q_blk vec(x−xref) + vec(diff(u))' Rd_blk vec(diff(u))` where `Q_blk = blkdiag(Q×N, P)` ([:72-88](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L72)). Constraints are all **hard** — no slacks, no penalty relaxation anywhere.

The problem is compiled **once**. To avoid recompiling as `A`/`B` change each tick, it uses the cvxpy sparse-parameter trick (cvxpy issue #1159, referenced at [:104-127](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L104)): `A_block` is flattened to its nnz data vector as a `Parameter`, reconstituted via a constant csc `Indexer` matrix and `cvxpy.reshape`. **This was verified sound** — `scipy.block_diag` over dense blocks stores all 25 entries per 5×5 block, so nnz stays 375 at both the zero-init linearization and a realistic state; the pattern captured at init stays valid.

Linearization residual: `Cd = x + f(x,u)*dt − Ad@x − Bd@u` ([:273](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L273)) — i.e. `linearize_around_state` returns pure Jacobians, and the affine offset is the solver's job. It linearizes about the **reference** (not the previous prediction, despite the comment at [:206](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L206) saying otherwise). It genuinely warm-starts: shift previous `u` by one, seed `xk.value = xref`, `warm_start=True` to OSQP.

### 4.5 Nonlinear MPC — [nonlinear_mpc_solver.py](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py)

CasADi SX multiple shooting → IPOPT. `g` stacks the initial-state equality `X[:,0] − Params[:nx,0]` with per-stage RK4 defects `X[:,k+1] − RK4(f, X[:,k], U[:,k], params_k, dt)`; `lbg = ubg = 0` over `nx*(N+1)` rows. Boxes go into `lbx/ubx` via `repmat`. Built once; each solve only swaps the parameter matrix.

The `p` parameter is a **single `(nx + num_params) × (N+1)` matrix stuffing both the reference and the per-stage model parameters** ([:52-54](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L52), assembled at [:186-191](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L186)) — rows `[:nx]` are the reference, rows `[nx:]` the model params. This is the mechanism that lets time-varying physics reach the NLP without rebuilding the graph. The LTV solver has no equivalent and ignores `p` entirely.

**It honors a strict subset of `MPCConfig`:** it ignores `config.P` (terminal cost reuses `Q`), `config.Rd` (no input-rate cost at all), and `ud_min/ud_max` (no rate constraints). LTV implements all four. Same config, two different honored subsets.

### 4.6 MPPI — [mppi_solver.py](f1tenth_planning/control/solvers/mppi_solver.py)

Port of `google-research/jax_mpc` (paper arXiv:1707.02342). Per `iteration_step` ([:113](f1tenth_planning/control/solvers/mppi_solver.py#L113)): (1) sample `da ~ TruncNormal(0,1)` truncated to `[u_min − a_opt, u_max − a_opt]` — truncating the *perturbation* by the current nominal guarantees `a = a_opt + da` lands in bounds; (2) `vmap` the rollout over samples; (3) reward-to-go `R = triu(ones(N,N)) @ r`; (4) **per-timestep** softmax weights across samples; (5) `a_opt += weighted_average(da)`.

Weighting ([:180-187](f1tenth_planning/control/solvers/mppi_solver.py#L180)) is `R_stdzd = (R − max R) / ((max R − min R) + damping)`, then `w = exp(R_stdzd / temperature)`, normalized. **Returns are min-max normalized to `[-1, 0]` before the exponential** — so `temperature` acts on a unitless normalized advantage, not on raw cost units. Consequence: temperature does not need retuning when `Q` scales change, but it also cannot express an absolute cost scale. This is *not* textbook MPPI (`exp(-(1/λ)(S − ρ))`).

Receding-horizon shift at the **start** of each `solve` ([:250-252](f1tenth_planning/control/solvers/mppi_solver.py#L250)): `a_opt = concat([a_opt[1:], zeros(nu)])` — appends a **zero** action, not a repeat of the last.

### 4.7 AP-MPPI — [ap_mppi_solver.py](f1tenth_planning/control/solvers/ap_mppi_solver.py)

**AP = Adaptive-Penalty** (stated explicitly at [ap_mppi_solver.py:21](f1tenth_planning/control/solvers/ap_mppi_solver.py#L21), which also carries the citations: paper [ieeexplore 11260933](https://ieeexplore.ieee.org/document/11260933), site `sites.google.com/view/sit-lmpc`, base code `github.com/mlab-upenn/SIT-LMPC` — mlab-upenn is the user's own lab).

`APMPPISolver` is **not** a subclass of `MPPISolver` — it re-derives from `MPCSolver` and duplicates ~90% of it verbatim. The two have already drifted. Any fix to one must be hand-ported.

Algorithm ([:133-242](f1tenth_planning/control/solvers/ap_mppi_solver.py#L133)), after the shared sample+rollout: constraint costs `c = vmap(constraints_costs)(s, a) -> [n_samples, C, N]`; penalize under **every lambda vector simultaneously** via `einsum('scn,cl->sln', c, lambdas)`; `r_modified = r[:,None,:] − c_weighted`; compute returns/weights/weighted-average `da` **per lambda** → `da_candidates [L, N, nu]`; **re-roll all L candidates** (a second `vmap` rollout at [:194](f1tenth_planning/control/solvers/ap_mppi_solver.py#L194)); score each by pure violation and pure return; branchless selection at [:211-227](f1tenth_planning/control/solvers/ap_mppi_solver.py#L211) — if any candidate has `violations == 0`, argmax pure return among feasible; else argmax `−violations` (least-violating).

So "adaptive penalty" means: sample a *population* of penalty weights, solve MPPI once per weight in parallel, and let a feasibility-first rule pick the winner. The penalty is **never gradient-tuned and never updated across steps** — lambdas are drawn once at construction from `uniform(lambdas_sample_range)` with a fixed `PRNGKey(0)` ([:67-92](f1tenth_planning/control/solvers/ap_mppi_solver.py#L67)) and never resampled or annealed.

**Constraint contract:** a callable `(x, u) -> (N,)` where `x` is `[N, nx]`, `u` is `[N, nu]`, and **positive values indicate violation**. Factories `make_state_min_constraint(x_min)` / `make_state_max_constraint(x_max)` ([dynamic_ap_mppi.py:20](f1tenth_planning/control/controllers/mpc/ap_mppi/dynamic_ap_mppi.py#L20) / [:58](f1tenth_planning/control/controllers/mpc/ap_mppi/dynamic_ap_mppi.py#L58)) close over the finite-bound indices at build time and return the L2 norm of the positive part per timestep.

The one behavioral divergence from `MPPISolver`: `APMPPISolver._step` clips rollout states to `config.x_min/x_max` ([:258](f1tenth_planning/control/solvers/ap_mppi_solver.py#L258)); `MPPISolver._step` does not. **This clipping interacts destructively with the constraints** — see [§11](#11-gotchas-drift-and-known-rough-edges).

---

## 5. Dynamics models & state conventions

### 5.1 State and control vectors

```
KINEMATIC (nx = 5):   x = [ x,      y,      delta,      v,      yaw    ]
                      units [m]     [m]     [rad]       [m/s]   [rad]

DYNAMIC   (nx = 7):   x = [ x,  y,  delta,  v,  yaw,  yaw_rate,  beta   ]
                      units                            [rad/s]    [rad]   (slip angle at CoG)

CONTROL   (nu = 2):   u = [ delta_v,   a ]
                      units [rad/s]    [m/s^2]
                          steering RATE,  longitudinal acceleration
```

**The control vector is `(steering_velocity, acceleration)` — NOT `(steering_angle, speed)`.** Every ABC docstring says otherwise ([controller.py:37](f1tenth_planning/control/controller.py#L37), [dynamics_model.py:32](f1tenth_planning/control/dynamics_model.py#L32), [:82](f1tenth_planning/control/dynamics_model.py#L82), [:101](f1tenth_planning/control/dynamics_model.py#L101)). The implementations are the truth ([kinematic_model.py:103-104](f1tenth_planning/control/dynamics_models/kinematic_model.py#L103): `delta_v = control[0]; a = control[1]`), corroborated by the `DynamicsConfig` bounds (`MIN/MAX_DSTEER` [rad/s] and `MIN/MAX_ACCEL` [m/s²] are the *input* bounds; `MIN/MAX_STEER` and `MIN/MAX_SPEED` are *state* bounds), and by `control_mode=(Steering_Speed, Accl)` at [mpc.py:46](f1tenth_planning/control/controllers/mpc/mpc.py#L46). **Trust the code, not the ABC docstrings.** The classical controllers are the exception — they declare `(Steering_Angle, Speed)` and return `(steering_angle, speed)`.

### 5.2 Waypoint column layouts — these differ per controller and are a real trap

| Controller | Shape | Columns | `plan()` validates |
|---|---|---|---|
| Pure Pursuit | `N×4` | `[x, y, v, yaw]` | `m >= 3` ([:174](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L174)) |
| Stanley | `N×4` | `[x, y, v, yaw]` | `m >= 4` ([:204](f1tenth_planning/control/controllers/stanley/stanley.py#L204)) |
| LQR | `N×5` | `[x, y, v, yaw, kappa]` | `m >= 5` ([:244](f1tenth_planning/control/controllers/lqr/lqr.py#L244)) |
| MPC family | `N×7` | `[x, y, delta, v, yaw, yaw_rate, beta]` (delta/yaw_rate/beta zero-filled) | `m >= 3` ([mpc.py:135](f1tenth_planning/control/controllers/mpc/mpc.py#L135)) — **too permissive, see §11** |

Note that "column 3" means **yaw** in the classical layout and **v** in the MPC layout. Both are internally consistent; the `waypoints` argument is simply not one format. Meanwhile the *vehicle state* array used inside the classical controllers is `[x, y, HEADING, VELOCITY]` — indices 2 and 3 **swapped** relative to the classical waypoint layout, and both conventions live in the same functions (e.g. [stanley.py:150-154](f1tenth_planning/control/controllers/stanley/stanley.py#L150) reads `waypoints[i,3]` for heading and `vehicle_state[2]` for heading).

### 5.3 The four-backend pattern

Each model implements the same continuous dynamics **four times**:

| Method | Signature | Consumer |
|---|---|---|
| `f` | `f(state, control, params: DynamicsConfig = None) -> np.ndarray` | LTV linearization residual |
| `f_casadi` | `f_casadi(params=None) -> ca.Function` | `NonlinearMPCSolver` (builds symbols internally; thin wrapper over `f_casadi_opti`) |
| `f_casadi_opti` | `f_casadi_opti(state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX` | Opti stack / NLS estimator |
| `f_jax` | `f_jax(state, control, params=None) -> jnp.ndarray` | `MPPISolver`, `APMPPISolver` |

**Only `__init__` is `@abstractmethod`** on `DynamicsModel` ([dynamics_model.py:11](f1tenth_planning/control/dynamics_model.py#L11)). Everything else is a plain method raising `NotImplementedError`. This is intentional — subclasses implement only the backends their solver needs and can still be instantiated — but it means a missing backend fails at *call* time, not construction time.

**There is no single source of truth for the four backends, and they have drifted.** See [§11.2](#112-the-dynamic-model-backends-disagree).

Models return **derivatives only**; integration is external via [discretizers.py](f1tenth_planning/control/discretizers.py). Solvers compose `discretizer(model.f_jax, x, u, p, dt)`. The discretizers are free functions taking `func(x, u, p)` as the first arg — matching `f`/`f_jax` exactly — so the same integrator works across numpy and JAX by duck typing.

### 5.4 Parameter vectors — positional, per-model, load-bearing

```
KinematicBicycleModel.parameters_vector_from_config -> [WHEELBASE]                       shape (1,1), num_params = 1
DynamicBicycleModel.parameters_vector_from_config  -> [MU, M, I, LR, LF, C_SF, C_SR, H, 9.81]
                                                                                         shape (9,1), num_params = 9
```
Note: **`LR` precedes `LF`**, and **gravity is appended as a pseudo-parameter**. Consumed positionally as `params[0..8]` at [dynamic_model.py:183-191](f1tenth_planning/control/dynamics_models/dynamic_model.py#L183) (JAX, indexed 2-D as `params[0,0]`) and [:265-273](f1tenth_planning/control/dynamics_models/dynamic_model.py#L265) (CasADi, indexed 1-D). Any code building a parameter vector **must** go through the helper, never by iterating `DynamicsConfig` fields (whose declaration order is different).

### 5.5 The equations

Kinematic ([kinematic_model.py:58-62](f1tenth_planning/control/dynamics_models/kinematic_model.py#L58)) — rear-axle reference, no slip angle in the position kinematics:
```
dx = v cos(yaw);  dy = v sin(yaw);  ddelta = delta_v;  dv = a;  dyaw = (v/WHEELBASE) tan(delta)
```

Dynamic, **high-speed (single-track)** branch ([dynamic_model.py:94-121](f1tenth_planning/control/dynamics_models/dynamic_model.py#L94)), with load-transfer shorthand `glr = g*lr − a*h`, `glf = g*lf + a*h`:
```
dx = v cos(yaw + beta);  dy = v sin(yaw + beta);  ddelta = delta_v;  dv = a;  dyaw = yaw_rate
ddyaw = (mu*m / (I*(lf+lr))) * [ lf*C_Sf*glr*delta + (lr*C_Sr*glf − lf*C_Sf*glr)*beta
                                 − (lf²*C_Sf*glr + lr²*C_Sr*glf)*yaw_rate/v ]
dbeta = (mu / (v*(lf+lr))) * [ C_Sf*glr*delta − (C_Sr*glf + C_Sf*glr)*beta
                               + (C_Sr*glf*lr − C_Sf*glr*lf)*yaw_rate/v ] − yaw_rate
```
Verified term-for-term identical to the gym's reference `single_track.py:128-167`, including the trailing `− PSI_DOT` in `BETA_DOT`.

Dynamic, **low-speed (kinematic)** branch ([dynamic_model.py:63-79](f1tenth_planning/control/dynamics_models/dynamic_model.py#L63)):
```
dyaw   = v cos(beta) tan(delta) / WHEELBASE
dbeta  = (lr*delta_v) / (WHEELBASE * cos(delta)² * (1 + (tan(delta)*lr/WHEELBASE)²))
ddyaw  = (1/WHEELBASE) * [ a cos(beta) tan(delta) − v sin(beta) tan(delta) dbeta
                           + v cos(beta) delta_v / cos(delta)² ]
```

The kinematic/dynamic transition is a **hard discontinuous switch**, not a blend — no interpolation, no convex weighting. Each backend uses its language's primitive (Python `if`, `jax.lax.select`, `ca.if_else`). `lax.select`/`if_else` evaluate **both** branches eagerly, which is why the CasADi path needs an epsilon guard at [:304](f1tenth_planning/control/dynamics_models/dynamic_model.py#L304).

`g = 9.81` is hardcoded at three separate sites ([dynamic_model.py:92](f1tenth_planning/control/dynamics_models/dynamic_model.py#L92), [:355](f1tenth_planning/control/dynamics_models/dynamic_model.py#L355), [dynamics_config.py:130](f1tenth_planning/control/config/dynamics_config.py#L130)). There is no `G` field on `DynamicsConfig`.

**Only the kinematic model has a Jacobian.** `DynamicBicycleModel.linearize_around_state` raises `NotImplementedError` at [:402](f1tenth_planning/control/dynamics_models/dynamic_model.py#L402) followed by ~62 lines of unreachable, unfinished code. This is why there is an LTV *kinematic* MPC and no LTV *dynamic* MPC.

**No Pacejka tire model exists.** `DynamicsConfig` carries `BF/BR/DF/DR/CF/CR` and [dynamics_config.py:136-158](f1tenth_planning/control/config/dynamics_config.py#L136) derives them, but grep confirms **nothing reads them** — both models use linear cornering stiffness `C_SF/C_SR` with longitudinal load transfer. The derivation is also dimensionally inconsistent (`BF = C_SF/(CF*DF)` divides a load-*normalized* stiffness `[1/rad]` by a force `[N]`, giving BF ≈ 0.18 where physical Pacejka B is ~5-20).

---

## 6. The planning subsystem

**Be honest: this is abandoned dead code.** [f1tenth_planning/planning/\_\_init\_\_.py](f1tenth_planning/planning/__init__.py) is 0 bytes, as are all four sub-package `__init__.py` files. Nothing in the repo imports `LatticePlanner` or `LaneSwitcher` — no examples, no tests. Git timeline confirms it: `planning/` last touched `12f9f48` (2024-01-05, "minor refactoring, linting with black" — cosmetic only); the Pure Pursuit relocation that broke its imports landed `dc9da67` (2025-03-20); `control/` was refactored through 2026-01-27. **This needs a port, not a patch.**

- **`fgm/fgm.py` and `wall_follow/wall_follow.py` contain no FGM and no wall-following code.** Both are 29-line `pyclothoids` grid benchmarks (`sample_grid` + `test` + `cProfile.run`), byte-identical to each other (md5 `130a19542efbe04eff2498994f051037`) and to an older copy of `lattice_planner/test_pyclothoids.py`. The directory structure implies implementations that do not exist.

- **`LatticePlanner`** ([lattice_planner.py:41](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L41)) has a genuinely good pluggable design: `add_sample_function` / `add_cost_function` (a list) / `add_selection_function` (defaults lazily to `np.argmin` inside `select()`), then `plan()` fits one `Clothoid.G1Hermite(0,0,0, gx, gy, gtheta)` per sampled goal and discretizes it with `utils.sample_traj(clothoid, 100)` → `(100,4)` of `[X, Y, Theta, |curvature|]`. But every path is broken: `eval` is called without its required `cost_weights` arg ([:200](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L200)); `PurePursuitPlanner()` is constructed with no args and called with the pre-2025 positional pose API; `sample_lookahead_square` index-assigns into an empty list; `i` is shadowed between `nearest_point`'s return and the loop counter; `np.dot((2,2), (28,3))` cannot broadcast; three of the four shipped cost functions reference an undefined `trajectory_generator.NUM_STEPS`. There are **three mutually incompatible cost-function contracts** (docstring says `func(pose_x, pose_y, pose_theta, velocity, kwargs)`; `eval` calls `func(traj)`; the shipped functions take `(traj_list, num_traj)`). Frame handling is incoherent end to end — clothoids are fit from the ego origin to world-frame goals and never transformed back.

- **`LaneSwitcher`** ([lane_switcher.py:11](f1tenth_planning/planning/lane_switcher/lane_switcher.py#L11)) is the most complete logic here: yaml/`Namespace` config, N lanes reprojected from raw CSV columns `[1,2,5,3,0]` into `[x, y, v, heading, s]`, opponent-aware occupancy via `scipy.spatial.distance.cdist`, two hysteresis counters (`avoid_buffer` to return to lane 0, `slowdown_counter` reset on each change), velocity scaling, and an outer-loop `cal_objectives` returning `[competitive_progress, safety]` from iTTC statistics. It calls `nearest_point` three times and **never imports it** → guaranteed `NameError`. `self.tracker` is constructed and never used — and is the sole reason for the broken `PurePursuitPlanner` import. It returns an `(N,3)` lane array, not a control command — a third `plan()` return contract alongside `LatticePlanner`'s `(steer, speed, traj)` and `Controller.plan`'s `np.ndarray`.

Neither planner subclasses `Controller`; both use the old positional `plan(pose_x, pose_y, pose_theta, velocity, waypoints)` signature. Migrating them to the ABC is the obvious unstarted work.

---

## 7. utils — [f1tenth_planning/utils/utils.py](f1tenth_planning/utils/utils.py)

One flat 423-line leaf module (imports nothing from `f1tenth_planning`, which is why every subsystem can use it without cycles). Re-exported by `from .utils import *` — but **every consumer imports from `f1tenth_planning.utils.utils` directly**, so the star-export is effectively unused (and, with no `__all__`, leaks `math`, `jax`, `np`, `njit`).

**The two-stage waypoint search is the core pattern of the subsystem:**
1. `nearest_point(point, trajectory) -> (projection, dist, t, i)` ([:15](f1tenth_planning/utils/utils.py#L15), `@njit`) — `t ∈ [0,1]` is the fractional position along segment `i`.
2. That `(i, t)` pair is recombined as a **single float `i + t`** and passed as `intersect_point`'s `t=` seed, which decomposes it back via `start_i = int(t)` / `start_t = t % 1.0` ([:159-160](f1tenth_planning/utils/utils.py#L159)) so the circle/polyline scan resumes exactly where the projection landed and never picks a point behind the car.

`intersect_point(point, radius, trajectory, t=0.0, wrap=False)` ([:150](f1tenth_planning/utils/utils.py#L150)) solves a segment/circle quadratic per segment and takes the first root in `[0,1]`. Wraparound is opt-in and only runs `if wrap and first_p is None`; it iterates `range(-1, start_i)` — deliberately starting at `-1` so the closing segment is tested first. It returns `(None, None, None)` on failure; only Pure Pursuit checks for that.

| Function | Line | Used by | Note |
|---|---|---|---|
| `nearest_point` | [:15](f1tenth_planning/utils/utils.py#L15) | all 4 controllers + lattice | docstring warns duplicate waypoints → div-by-zero, no guard |
| `intersect_point` | [:150](f1tenth_planning/utils/utils.py#L150) | PP, lattice | can return `first_i = -1` (unmodded) |
| `get_actuation` | [:245](f1tenth_planning/utils/utils.py#L245) | PP only | returns `(speed, steering_angle)` — **swapped** vs `plan()`'s return |
| `calc_interpolated_reference_trajectory` | [:87](f1tenth_planning/utils/utils.py#L87) | MPC family only | speed-parameterized; pure Python; **has a live bug, see §11** |
| `solve_lqr` | [:264](f1tenth_planning/utils/utils.py#L264) | LQR only | Riccati fixed point via `pinv` |
| `update_matrix` | [:308](f1tenth_planning/utils/utils.py#L308) | LQR only | builds error-dynamics `A (4×4)`, `B (4×1)` |
| `pi_2_pi` | [:380](f1tenth_planning/utils/utils.py#L380) | Stanley, LQR | wraps to `(-π, π]`; applied to heading **error** only, never raw pose |
| `sample_traj` | [:385](f1tenth_planning/utils/utils.py#L385) | lattice | `@njit` **commented out** — calls into pyclothoids objects |
| `jnp_to_np` | [:419](f1tenth_planning/utils/utils.py#L419) | mpc.py | the only JAX↔numpy boundary marshaller |

**Reference trajectory generation** ([:87-147](f1tenth_planning/utils/utils.py#L87)) is shared by every MPC controller and is **speed-parameterized, not time- or arclength-uniform**: from `nearest_point`'s `(ind_current, t_current)` it integrates `t_i = t_{i-1} + speed*dt/dl`, wraps indices modulo the lap, and blends whole waypoint rows `ref = (1−t)*wp[i] + t*wp[i+1]`. So the horizon *stretches with velocity*. It is generic over state width (returns `(N+1, ncols)`, caller decides what `nx` means).

**Dead:** `map_collision` ([:398](f1tenth_planning/utils/utils.py#L398), body is `pass`), `calc_ref_trajectory_indices` ([:54](f1tenth_planning/utils/utils.py#L54)), `quat_2_rpy` ([:348](f1tenth_planning/utils/utils.py#L348)), `input_acceleration_to_speed` ([:405](f1tenth_planning/utils/utils.py#L405)), `input_steering_speed_to_angle` ([:412](f1tenth_planning/utils/utils.py#L412)). All zero consumers.

**No ZOH lives here** despite being a natural fit — it's `system_matrix_discretization(A, B, dt, method='exact')` in [discretizers.py:43](f1tenth_planning/control/discretizers.py#L43), a matrix-exponential of the block `M = [[A, B], [0, 0]]` via `scipy.linalg.expm`.

`import jax` at [:10](f1tenth_planning/utils/utils.py#L10) is a module-level hard dependency purely for the 5-line `jnp_to_np`. Every consumer — including Pure Pursuit, which has nothing to do with JAX — pays JAX import time.

---

## 8. How to run something

The canonical template, distilled from all nine scripts in [examples/control/](examples/control/). **This is the API as written in the repo today; it does not currently run** (see [§9](#9-the-f1tenth_gym-interface)).

```python
import numpy as np
import gymnasium as gym

from f1tenth_gym.envs import F110Env
from f1tenth_planning.control import PurePursuitPlanner
from f1tenth_planning.control.config.dynamics_config import f1tenth_params


def main():
    # 1. Env. control_input + observation_config MUST manually match the controller's control_mode.
    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": ["speed", "steering_angle"],      # MPC family: "accl"
            "observation_config": {"type": "kinematic_state"},  # MPC family: "dynamic_state"
            "params": F110Env.f1tenth_vehicle_params(),         # optional; keep in sync with f1tenth_params()
        },
        render_mode="human",
    )

    # 2. Track always comes off the env. (Track.from_raceline_file for a custom raceline.)
    track = env.unwrapped.track

    # 3. Controller: track= first, params= optional.
    planner = PurePursuitPlanner(track=track, params=f1tenth_params())

    # 4. Three render callbacks, always these three, always this order.
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

    # 5. Reset to the first raceline pose; poses is (num_agents, 3).
    poses = np.array([[track.raceline.xs[0], track.raceline.ys[0], track.raceline.yaws[0]]])
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # 6. obs -> plan -> step.
    laptime = 0.0
    while not done:
        u0, u1 = planner.plan(obs["agent_0"])  # (steer_angle, speed) | (steer_vel, accl)
        obs, timestep, terminated, truncated, infos = env.step(np.array([[u0, u1]]))
        done = terminated or truncated
        laptime += timestep                    # NB: the "reward" slot IS the timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
```

Invocation: `.venv/bin/python examples/control/pure_pursuit.py` (requires a display for `render_mode="human"`). There is no runner, no Makefile, and the [README.md](README.md) documents none of this.

**Two controller families, distinguished only by `control_mode`** — the env config is not checked against it:

| Family | `control_mode` | env `control_input` | env `observation_config` | `plan()` returns |
|---|---|---|---|---|
| Classical (PP, Stanley, LQR) | `(Steering_Angle, Speed)` | `["speed", "steering_angle"]` | `kinematic_state` | `(steer_angle, speed)` |
| MPC/MPPI (all five, via [mpc.py:46](f1tenth_planning/control/controllers/mpc/mpc.py#L46)) | `(Steering_Speed, Accl)` | `"accl"` | `dynamic_state` | `(steer_vel, accl)` |

`control_mode` is stored by the base ([controller.py:21](f1tenth_planning/control/controller.py#L21)) and **never read by anything**. Correctness depends entirely on the user configuring the env by hand. The tuple is currently decorative — and it is the direct cause of the import blocker.

The **best example to read** is [examples/control/dynamic_ap_mppi.py](examples/control/dynamic_ap_mppi.py) — the only one that teaches a real concept. Its comment block at lines 61-79 is the clearest in-repo statement of the clip-bounds-vs-soft-constraints design: *"Crucial to NOT clip states that we want to constrain!"* / *"For velocity constraints to work, x_clip_max[3] must be > x_max[3]"* — which is why it sets `config.x_max[3] = inf` at [:131](examples/control/dynamic_ap_mppi.py#L131).

### 8.1 The ROS 2 wrapper — the second integration target

[examples/ros_wrappers/control_ros_wrapper.py](examples/ros_wrappers/control_ros_wrapper.py) (226 lines) is the only artifact driving the control library **outside** the gym loop, and the template a real-vehicle port wants. `class ControlRosWrapper(Node)` inverts the control flow — `rclpy.spin()` owns the loop and `pose_callback(self, pose_msg: Odometry)` *is* the controller. The load-bearing patterns worth keeping: it hand-builds a `state_dict` (:161) with exactly the gym's `dynamic_state` keys (`pose_x, pose_y, delta, linear_vel_x, pose_theta, ang_vel_z, beta`) — reconstructing that dict is the actual work of porting, since `obs["agent_0"]` supplies it for free in the gym; it feeds `delta` back from the previous command (:187) because odometry carries no steering angle; it derives `beta = arctan2(vy, vx)` and yaw from the quaternion; and it mirrors the three pyqtgraph render hooks with three RViz `MarkerArray` publishers using a preallocate-and-mutate marker pool (surplus markers flipped to `DELETE`), gated on `get_subscription_count() > 0`. **It does not run as written** — see the ROS-wrapper bullet in [§11.6](#116-structural--hygiene) (stale import name, unassigned `self.params`, a dead `(action, info)` unpack against the flat 2-vector `plan()` returns, and a `trajectory_logs.csv` vs `trajectory_log.csv` filename typo). The structure is sound; the defects are localized to `__init__`/`pose_callback`. Repair, don't rewrite.

---

## 9. The f1tenth_gym interface

`f1tenth_planning` assumes four things from the gym: the env loop, `Track`/`Raceline` as the waypoint source, vehicle parameters, and the `EnvRenderer` object API. **All four have drifted.**

### 9.1 What the library imports

```
from f1tenth_gym.envs.track import Track                          # controller.py:5   — OK
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum
                                                                  # controller.py:7   — ✗ DOES NOT EXIST
from f1tenth_gym.envs.f110_env import F110Env                     # dynamics_config.py:2 — OK, but…
F110Env.f1tenth_vehicle_params()                                  # dynamics_config.py:196 — ✗ DOES NOT EXIST
```
Import sites for the enums (10 references across **5 files** — 5 imports + 4 constructor kwargs + 1 type annotation): [controller.py:7,11](f1tenth_planning/control/controller.py#L7); [mpc.py:11,46](f1tenth_planning/control/controllers/mpc/mpc.py#L11); [stanley.py:31,68](f1tenth_planning/control/controllers/stanley/stanley.py#L31); [lqr.py:32,65](f1tenth_planning/control/controllers/lqr/lqr.py#L32); [pure_pursuit.py:13,53](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L13). **The rename is two-part, not one:** the *types* `SteerActionEnum`/`LongitudinalActionEnum` → `SteerActionType`/`LongitudinalActionType`, **and** the *members* to `SCREAMING_CASE` (`Steering_Angle` → `STEERING_ANGLE`, `Steering_Speed` → `STEERING_SPEED`, `Accl` → `ACCL`, `Speed` → `SPEED`). Fixing only the type name converts the `ImportError` into an `AttributeError` at the four usage lines.

### 9.2 What the gym actually exposes

The gym version reviewed in depth (the vendored clone, branch `dev-humble` @ `bdaec14`) is a **rewritten, typed-dataclass API**:

| The library calls | The gym provides |
|---|---|
| `SteerActionEnum.Steering_Angle` | `SteerActionType.STEERING_ANGLE` (IntEnum, `action.py:18`) |
| `LongitudinalActionEnum.Accl` | `LongitudinalActionType.ACCL` (IntEnum, `action.py:7`) |
| `gym.make(..., config={...})` | `TypeError: config must be an EnvConfig instance` (`f110_env.py:55`) — dicts are dead |
| `F110Env.f1tenth_vehicle_params()` → dict | module-level `get_f1tenth_vehicle_parameters()` / `F1TENTH_VEHICLE_PARAMETERS`, a **frozen `VehicleParameters` dataclass** |
| `e.render_closed_lines(pts, color, size)` then `.setData(pts)` | `e.get_closed_lines_renderer(pts, color, size)` then `.update(pts)` |

The dict→dataclass config migration maps as: `"map"` → `EnvConfig.map_name`; `"control_input": "accl"` → `ControlConfig(longitudinal_mode=LongitudinalActionType.ACCL)`; `"observation_config": {"type": "dynamic_state"}` → `ObservationConfig(type=ObservationType.DYNAMIC_STATE)`; `"params"` → `EnvConfig.params` (a `VehicleParameters`, not a dict).

The renderer fix is mechanical: `render_X(...)` → `get_X_renderer(...)`, `.setData(p)` → `.update(p)`. Affected: [controller.py:65,69](f1tenth_planning/control/controller.py#L65), [mpc.py:90,94,105,109](f1tenth_planning/control/controllers/mpc/mpc.py#L90), [dynamic_mppi.py:114,118](f1tenth_planning/control/controllers/mpc/mppi/dynamic_mppi.py#L114), plus the three classical controllers.

`_dynamics_config_from_gym_params` ([dynamics_config.py:60](f1tenth_planning/control/config/dynamics_config.py#L60)) also needs rewriting: it dict-subscripts `gym_params['s_min']` throughout, but `VehicleParameters` is a frozen dataclass — not subscriptable, and `in` is undefined. It needs attribute access plus `dataclasses.fields()` for the optional-key probes.

### 9.3 Observation contract

The gym's `std_state` is the model-agnostic 7-vector `[X, Y, steering_angle, speed, yaw, yaw_rate, slip_angle]` — **byte-identical ordering to `f1tenth_planning`'s dynamic model state and to the ST model**, so `DYNAMIC_STATE` obs feeds `DynamicBicycleModel` with no permutation. Derived scalar fields are computed from it: `vx = speed*cos(beta)`, `vy = speed*sin(beta)`.

The library's controllers collectively read: `pose_x`, `pose_y`, `pose_theta`, `linear_vel_x`, `delta`, `ang_vel_z`, `beta`. **Under the new gym, `DYNAMIC_STATE` supplies `linear_vel_magnitude`, not `linear_vel_x`** — so even after the config migration, [examples/control/dynamic_ap_mppi.py:192](examples/control/dynamic_ap_mppi.py#L192) would `KeyError`. The only preset covering all seven fields is `FRENET_DYNAMIC_STATE`, or `ObservationType.FEATURES` with an explicit tuple.

### 9.4 Track / Raceline

`track.raceline` supplies `xs`, `ys`, `yaws`, `vxs` (+ `ks` for LQR). **Naming trap:** the `Raceline` constructor kwargs are `psis`/`kappas`/`velxs`/`accxs` but the *attributes* are `.yaws`/`.ks`/`.vxs`/`.axs`. Both schemes are live in the gym. `Track.from_raceline_file(filepath, delimiter=";", skip_rows=3)` is the custom-raceline path — used for real by the ROS wrapper ([control_ros_wrapper.py:40](examples/ros_wrappers/control_ros_wrapper.py#L40)) and vestigially by [dynamic_mppi.py:36](examples/control/dynamic_mppi.py#L36) (which discards it, [§11.4](#114-verified-crashes-and-silent-wrongness)). It is the one spot in `examples/` where the `Raceline` constructor/attribute mismatch (`psis=`/`kappas=` in, `.yaws`/`.ks` out) can actually bite a port. `env.unwrapped.track` is how every example obtains it — note `gym.make` returns a wrapper, so the `env: F110Env` annotation in the examples is a lie and `.unwrapped` is mandatory.

### 9.5 Which gym is installed — RESOLVED, but the target is a real decision

The facts are settled; what to *do* about them is a genuine fork the readers split on (below). Verified:

| Source | Says | Evidence |
|---|---|---|
| `.venv` (installed) | `bdaec14` on **`dev-humble`** — the new typed-dataclass API | `.venv/.../f1tenth_gym-1.0.0.dev0.dist-info/direct_url.json` → `{"commit_id": "bdaec14…", "requested_revision": "dev-humble"}` |
| [pyproject.toml:36](pyproject.toml#L36) + [uv.lock:326](uv.lock#L326) | **`dev-dynamics` @ `67bc6db`** (the pin) | `branch = "dev-dynamics"` / `…#67bc6dbe5826e3160fa97116d4efbb51b7a9670f` |
| `f1tenth_planning/` source | targets **`dev-dynamics`** | imports `SteerActionEnum` — a name that exists **only** at the pin |
| `f1tenth_gym/` (untracked clone) | **`dev-humble`** @ `bdaec14` | byte-identical to the installed copy (`diff -rq` clean) — the clone was installed over the pin |

The install and the pin are **genuinely divergent**, not fast-forwardable: `git merge-base --is-ancestor 67bc6db HEAD(dev-humble)` → false; they forked at `2779a9f` (2025-07-07). `SteerActionEnum`/`LongitudinalActionEnum` (`enum.Enum`) **do exist** at `67bc6db` on `dev-dynamics` — they were removed on `dev-humble` by `c9e4eab` "Deprecate classes for actions and integrators" (2025-09-26) and replaced by `*ActionType` (`enum.IntEnum`, `SCREAMING_CASE` members). (The draft's earlier "exists in no copy of the gym" was wrong; it exists at the pin, absent only on `dev-humble`.) The imported package is always the one in `site-packages`, **never** the local `./f1tenth_gym/` clone (a copy taken at install time, not a symlink — editing the clone has zero runtime effect).

**The decision — the two readers reached opposite recommendations, and this is genuinely yours to make:**

- **Path A — revert the venv to the pin (`uv sync`).** The checked-in source is written against `dev-dynamics`; `uv sync` reinstalls `67bc6db` and the four controllers **import cleanly again** with no source edits. Treats the `dev-humble` install as the thing that is currently wrong with the venv. Cost: discards the 125 commits of `dev-humble` work the clone represents, and leaves the library on the older gym API.
- **Path B — migrate the source to `dev-humble`.** The environment was deliberately staged to `dev-humble` (both the install and the untracked clone sit at `bdaec14`; §9.2's migration table is written against that API), so treat the `ImportError` as the *starting condition of a migration*, not a broken checkout. Do **not** run `uv sync` — it silently undoes the staging by reverting to the pin, and the failure "disappearing" reads as fixed when it isn't. Fix: apply the §9.2 renames across the 5 files and repoint [pyproject.toml:36](pyproject.toml#L36) to `dev-humble`.

Re-verify the install at any time: `cat .venv/lib/python3.13/site-packages/f1tenth_gym-*.dist-info/direct_url.json`.

---

## 10. Current state of active work

**Branch `dev-sit-lmpc`.** SIT = **Safe Information-Theoretic** (triangulated: `ap_mppi_solver.py:21-24` cites the paper/site/base-code; the site titles the paper *"SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks"*; and `SITLMPCConfig`'s own docstring reads *"Combined configuration for Safe-MPPI + LMPC (IT-LMPC) controller"*). The intent is to pair the AP-MPPI sampling inner loop with an LMPC outer loop that accumulates completed laps into a safe set, fits a value function (cost-to-go), and iteratively improves lap time on a repeated task.

**Development is deliberately bottom-up, and the inner loop won.** Reconstructed timeline:

| Date | Commit | What |
|---|---|---|
| 2025-11-25 | `b4dd056`, `d658d68` | LMPC skeleton (`base`/`components`/`manager`) lands against a placeholder `SafeMPPISolver` |
| 2025-12-16 | `b270717` | 46-line safe_mppi placeholder replaced by the real 382-line `APMPPISolver`. **Safe-MPPI renamed AP-MPPI.** |
| 2025-12-16 | `c6c25ca` | camelCase rename sweep (`lmpc_config` → `LMPCConfig`, etc.) — broke the ROS wrapper |
| 2025-12-18 | `01d47ef` | **"Delete old lmpc scripts"** — removes `base.py`, `components.py`, `manager.py` (−343 lines), leaves `__init__.py` importing them |
| 2025-12-22 → 2026-01-27 | `aa5ed9c`, `42205b3`, `d8d5941`, `b1d3818`, `d9bcbe4`, `f751e7e` | **All AP-MPPI**: convenience class, config init fixes, state/control bounds, `ref_velocity_bounds`, the example, rendering |

**The LMPC layer was intentionally demolished, not accidentally broken** — it was written against a placeholder ABC and is awaiting a rewrite against the real `APMPPISolver`. Recover the deleted design with `git show 01d47ef^:f1tenth_planning/control/controllers/lmpc/base.py` (likewise `components.py`, `manager.py`).

**The central unsolved design problem** is a solver-contract mismatch, and it is *two* independent breaks. (1) **Argument binding:** the deleted `LMPCSolver` ABC required `solve(x0, ref_traj, safe_set, value_fn, **kwargs)`; the real `APMPPISolver` implements `solve(x0, ref_traj, vis=True, p=None, Q=None, R=None)` ([ap_mppi_solver.py:339](f1tenth_planning/control/solvers/ap_mppi_solver.py#L339)) — no safe-set, no value-function parameter, so `SITLMPCPlanner` would bind `safe_set_snapshot`→`vis` and the VF→`p`. (2) **Return arity:** `LMPCSolver` returned a 3-tuple `(x_pred, u_pred, meta)`; `APMPPISolver` returns `(xk, uk)` — the deleted `base.py` unpacks three and would `ValueError`.

The draft's earlier hunch — route the value function through `APMPPIConfig.constraints` — was **verified to fail**, twice over, and the second reason is the most dangerous property on the branch:

- **The `constraints` channel is a feasibility gate, not a cost.** [ap_mppi_solver.py:204-227](f1tenth_planning/control/solvers/ap_mppi_solver.py#L204) computes `violations = sum(max(0, c))`; if *nothing* is feasible it selects on `-violations` and **discards `pure_returns` entirely**. A terminal value function is positive almost everywhere → every candidate looks infeasible → reference tracking silently stops. Correct decomposition: **safe-set membership → `constraints`** (`dist(x_N, SS) - ε` is naturally zero when satisfied — exactly the gate's contract); **value function → cost**, via the `reward_function` hook, **not** `constraints`.
- **The stale-JIT trap makes in-place VF retraining permanently ignored.** `iteration_step` is `@partial(jax.jit, static_argnums=(0))` ([:133](f1tenth_planning/control/solvers/ap_mppi_solver.py#L133)) — `self` is static, hashed by identity, and constraints are frozen at **`__init__`** ([:65,102](f1tenth_planning/control/solvers/ap_mppi_solver.py#L65)), not first solve. Mutating `config.constraints` *or* rebuilding a closure over a retrained VF is a **no-op** — the trace keeps lap-1's constants. Verified empirically: retrain in place → still `[2.]`. LMPC's entire premise is iterate-and-improve, so this would run lap 50 on lap 1's value function with no error. The **only** escape that costs zero recompiles is to thread VF params as a **traced argument** (widen `iteration_step`/`constraints_costs`), never closed over.

**Recommended bridge (inference, not established):** an `LMPCSolver` **adapter** wrapping `APMPPISolver` that (a) restores the 3-tuple return at the seam, and (b) passes the VF params as a traced `theta` arg into a widened `iteration_step` — keeping LMPC concepts out of the already-drifting `MPCSolver` ABC. Two **prerequisites block everything**: `SimpleSafeSetStore._compute_cost` scores by *absolute* state norm (`np.dot(states[i], states[i])` over `[x, y, …]`), so cost-to-go depends on world coordinates — replace with progress-to-go; and `SimpleValueFunctionModel.predict` is numpy (`np.argmin`) and cannot be JAX-traced — rewrite in `jnp` if it goes inside the rollout.

**Deleted design worth knowing (from `01d47ef^`):** `LMPCController.plan()` builds the 7-dim `x0`; if `safe_set_store.num_trajectories < lmpc_cfg.ss_size` **and** `base_controller is not None`, it **delegates to the base controller** — that is the bootstrap/exploration-lap mechanism, and exactly why [examples/control/sit_lmpc.py](examples/control/sit_lmpc.py) passes a `NonlinearDynamicMPPIPlanner` as `base_controller`. Otherwise it calls `solver.solve(x0, None, safe_set_snapshot, value_function)`. `complete_iteration()` commits the lap and retrains the value function. `SimpleSafeSetStore` stored ≤5 laps with backward cost-to-go, and `SimpleValueFunctionModel.predict` did nearest-neighbor lookup — explicitly described as mirroring the Rosolia/Borrelli `LinearLMPC addTrajectory/computeCost` pattern. **Caveat before restoring verbatim:** its `_compute_cost` used `np.dot(states[i], states[i])` — a raw quadratic norm of the *absolute* state including global x/y and yaw, so "cost" scaled with distance from the world origin rather than measuring progress. Almost certainly wrong for LMPC.

**Value-function roadmap:** `ModelConfig` ([model_config.py:4](f1tenth_planning/control/config/model_config.py#L4), `hidden_dim=256, hidden_layers=4, lr=5e-4, model_type='nf'` with options `nf`/`bnn`/`mlp`, `max_epoch=200, ensemble_size=1`) is composed into `SITLMPCConfig` and consumed by nothing. Normalizing-flow / Bayesian-NN value functions are the intended endpoint; the nearest-neighbor store was the placeholder.

**Estimation** ([f1tenth_planning/estimation/](f1tenth_planning/estimation/)) is entirely orphaned — nothing imports it, and it is unrelated to the "SIT" acronym. `NLSParameterEstimator` ([nls_estimator.py:14](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L14)) builds a CasADi `Opti` NLP once (decision vars = params, `opti.parameter` placeholders = the last N of `(X_k, U_k, X_{k+1})`), then slides a window and re-solves per step. Architecture is sound; the implementation is not. Six bugs; several are already fixed on the local **`dev-cleanup`** branch (`60d0c95`, `d87b2a1`) — see [§11.5](#115-estimation-nls) and the branch topology in [§10.1](#101-working-on-this-branch-branches--dev-cleanup).

### 10.1 Working on this branch (branches & dev-cleanup)

The branch names are misleading; the real topology (verified by trial merge):

```
origin/main (2025-03-20, Dependabot-only since)
   └── 6794e7e "Add sampled trajectories rendering to MPPI" (2025-12-09)  ← origin/dev-cleanup
          ├── +19 commits → dev-sit-lmpc  (f751e7e, 2026-01-27)  [in sync with origin]
          └── +8  commits → dev-cleanup   (46e5497, 2026-05-15)  [LOCAL ONLY, unpushed]
```

- **`origin/dev-cleanup` IS the merge-base `6794e7e`** — the *published* dev-cleanup is an **ancestor** of dev-sit-lmpc (19 behind, 0 ahead); nothing to pull. The "newer tip" is **8 commits that live only on this machine**, never pushed — small correctness fixes (stanley typos + zero-speed guard, `rk4_discretization` use in the NLS estimator, the `LTV_mpc_solver` `last_pred`→`last_x` fix, a lattice PP-call fix, the Pacejka key-case fix, a Python-3.14 drop). **Push them before anything else** — a disk failure erases them.
- **`dev-sit-lmpc` → `main` is a clean fast-forward** (`origin/main` is a strict ancestor, 173 ahead / 0 behind). There is **no CI, no `.github/`, no PR/CONTRIBUTING** — "open a PR" is a conversation with the maintainer, not a gate.
- **A plain `git merge dev-cleanup` is NOT clean** despite the tiny semantic divergence (10–25 lines/file). Root cause: dev-sit-lmpc commit `c6c25ca` ("Switch to camel case") rewrote 24 `.py` files LF→**CRLF**; the base and dev-cleanup are LF, so every line of every touched file conflicts as one whole-file hunk. `git merge --no-ff -X ignore-cr-at-eol dev-cleanup` auto-resolves the 7 Python files, leaving only [pyproject.toml](pyproject.toml) (hand-resolve) and [uv.lock](uv.lock) (`git checkout --theirs uv.lock && uv lock`). Bare `cherry-pick` also fails on 7/8 commits without the flag. Consider normalizing (`.gitattributes` `*.py text eol=lf` + `dos2unix`) in the same pass — 24 of 51 tracked `.py` files are CRLF today. `815367d` (the 3.14 drop) is the one commit with a genuine LF-vs-LF `pyproject.toml` conflict; take its end state (`requires-python = ">=3.10,<3.14"`) by hand.

**Commit #1 regardless of path: `.gitignore` the `f1tenth_gym/` clone.** It is a nested repo (own `.git`, `dev-humble` @ `bdaec14`); `git add f1tenth_gym/` stages a bare **gitlink** (mode 160000), so a fresh clone gets an *empty* `f1tenth_gym/` and no way to populate it. `f1tenth_planning` itself is installed **editable** (`_editable_impl_*.pth`), so its edits take effect immediately — the opposite of the clone. Two siblings, opposite behavior; this is the single easiest hour to lose.

Only 3 of `origin`'s 16 branches matter (`dev-sit-lmpc`, `dev-cleanup`, `main`); the other 13 are stale (2024 or earlier).

---

## 11. Gotchas, drift, and known rough edges

> **Before fixing anything below, check the local `dev-cleanup` branch** ([§10.1](#101-working-on-this-branch-branches--dev-cleanup)) — it already carries committed fixes for several of these (`git log --oneline dev-sit-lmpc..dev-cleanup`). Findings with a known fix there are marked. Two §11.2 items (the gym's `V < 0.5` threshold; CasADi's unsigned compare) have **no** fix anywhere and remain fully open.

### 11.1 Blockers — nothing runs

1. **`import f1tenth_planning.control` → `ImportError: cannot import name 'SteerActionEnum'`.** Verified. 10 references, 5 files. See [§9.1](#91-what-the-library-imports) — note the fix must rename both the enum *types* and their *members*. A bare `import f1tenth_planning` succeeds because [f1tenth_planning/\_\_init\_\_.py](f1tenth_planning/__init__.py) is 3 lines and imports nothing — which is exactly why a shallow smoke test would go green while the whole library is dead.
2. **`f1tenth_params()` is a default argument** ([pure_pursuit.py:46](f1tenth_planning/control/controllers/pure_pursuit/pure_pursuit.py#L46), [stanley.py:66](f1tenth_planning/control/controllers/stanley/stanley.py#L66), [lqr.py:61](f1tenth_planning/control/controllers/lqr/lqr.py#L61), [mpc.py:39](f1tenth_planning/control/controllers/mpc/mpc.py#L39)), so it's evaluated **at import time**. Its `F110Env.f1tenth_vehicle_params()` call therefore crashes on module import, not instantiation. It's also a **shared mutable default** — every controller omitting `params=` shares one `DynamicsConfig`, and `model.f` writes to `self.params` when passed a config, so cross-instance mutation is reachable.
3. **Renderer API is stale** — `render_*` + `.setData` vs `get_*_renderer` + `.update`. Every registered callback would fail on first frame.
4. **`import f1tenth_planning.estimation` → `ImportError: cannot import name 'NLS'`** ([estimation/\_\_init\_\_.py:1](f1tenth_planning/estimation/__init__.py#L1) — no such symbol; the class is `NLSParameterEstimator`; and `estimators/`, `parameter_estimators/`, `NLS/` have **no `__init__.py`**).
5. **`import ...controllers.lmpc` → `ModuleNotFoundError`** — `base`/`components`/`manager` deleted; export commented out at [control/\_\_init\_\_.py:16-23](f1tenth_planning/control/__init__.py#L16) with `# TODO: Fix lmpc module - base.py is missing`.
6. **`pyclothoids` is imported by 3 shipped modules and is not a declared dependency**, not in `uv.lock`, and not installed. All of `planning/` is dead on a clean install regardless.

### 11.2 The dynamic model backends disagree

This is the most dangerous class of bug in the repo, because it's silent.

| Defect | Detail |
|---|---|
| **Switch threshold is 4-way inconsistent** | numpy `abs(v) <= 0.1` ([dynamic_model.py:61](f1tenth_planning/control/dynamics_models/dynamic_model.py#L61)); JAX `abs(v) <= 1.5` ([:258](f1tenth_planning/control/dynamics_models/dynamic_model.py#L258)); CasADi `v >= 1.5` ([:340](f1tenth_planning/control/dynamics_models/dynamic_model.py#L340)); **gym reference `V < 0.5`**. For `0.1 < |v| < 1.5` the same model object gives materially different derivatives depending on which solver reads it, and **none match the simulator being controlled**. |
| **CasADi switch is unsigned** | `ca.if_else(v >= 1.5, ...)` is not `abs(v)`. In reverse (`v <= -1.5`) CasADi takes the *kinematic* branch while JAX takes the *ST* branch. `MIN_SPEED = -5.0` for f1tenth — this is reachable. |
| **Math bug: `tan(delta)**2` should be `tan(delta)`** | In the JAX ([:214](f1tenth_planning/control/dynamics_models/dynamic_model.py#L214)) and CasADi ([:292](f1tenth_planning/control/dynamics_models/dynamic_model.py#L292)) low-speed `dbeta`. numpy ([:69](f1tenth_planning/control/dynamics_models/dynamic_model.py#L69)) is correct and matches the reference. Verified numerically: `dbeta` error +1.1% at δ=0.2 rad, +3.9% at δ=0.4 rad (near full lock; `s_max=0.4189`). |
| **CasADi low-speed drops the slip angle, and the comments lie** | [:321-322](f1tenth_planning/control/dynamics_models/dynamic_model.py#L321) computes `v*cos(yaw)` / `v*sin(yaw)` while the inline comments on those exact lines say `# dx/dt = v * cos(yaw + slip_angle)`. numpy, JAX and the gym reference all include it. CasADi-based nonlinear dynamic MPC predicts a different low-speed trajectory than MPPI does. |
| **No epsilon guard in numpy/JAX high-speed** | They divide by `v` and `v**2`. CasADi alone adds `epsilon = 1e-4` ([:304](f1tenth_planning/control/dynamics_models/dynamic_model.py#L304)). numpy is protected by its Python `if`, but `jax.lax.select` evaluates **both** branches eagerly — at `v=0` the ST branch yields inf/NaN. Forward values are fine (select discards them), but **any `jax.grad`/`jacobian` through `f_jax` returns NaN**. |

Also: `DynamicBicycleModel.config_from_parameters_vector` ([:370](f1tenth_planning/control/dynamics_models/dynamic_model.py#L370)) binds `current_params = self.params` **by reference**, mutates it, and returns it — the "conversion" silently overwrites the model's live config. The kinematic sibling constructs a new object ([kinematic_model.py:158](f1tenth_planning/control/dynamics_models/kinematic_model.py#L158)) — opposite semantics for the same interface method. And `DynamicBicycleModel.num_params` lists `MU` **twice** and omits gravity; it returns 9, which accidentally matches, so nothing breaks today — but any edit desyncs it from `parameters_vector_from_config`, which [nonlinear_mpc_solver.py:53](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L53) trusts.

`@partial(jax.jit, static_argnums=(0))` on `self` ([:159](f1tenth_planning/control/dynamics_models/dynamic_model.py#L159)) makes the model a static arg hashed by identity: **every new `DynamicBicycleModel` instance triggers a full recompile and retains a cache entry** (leak in loops), and buys nothing since params arrive traced.

**Two verified crashes in the kinematic model** (latent only because no MPPI planner instantiates it today):
- `config_from_parameters_vector` does `return DynamicsConfig(WHEELBASE=p[0])` ([kinematic_model.py:158](f1tenth_planning/control/dynamics_models/kinematic_model.py#L158)) → `TypeError: missing 22 required positional arguments`.
- `f_jax` indexes `params[0]` 1-D style ([:131](f1tenth_planning/control/dynamics_models/kinematic_model.py#L131)) while `parameters_vector_from_config` returns shape `(1,1)` → `TypeError: Cannot concatenate arrays with different numbers of dimensions`. The dynamic model correctly uses `params[0,0]`.

### 11.3 Constraints that aren't enforced

| Where | What |
|---|---|
| **`KinematicMPCPlanner`** ([LTV_kinematic_mpc.py:37-44](f1tenth_planning/control/controllers/mpc/LTV_mpc/LTV_kinematic_mpc.py#L37)) | Sets **no bounds at all**. `kinematic_mpc_config()` passes none, `__post_init__` fills `±inf`, the planner never overrides. The LTV QP has **no steering, speed, steering-rate or acceleration limits**. It also neuters `MPCController`'s reference-velocity clipping, which reads `solver.config.x_min[3]/x_max[3]` → `±inf` → the `np.clip` at [mpc.py:178](f1tenth_planning/control/controllers/mpc/mpc.py#L178) is a no-op. (Verified: `±inf` bounds don't error in cvxpy/OSQP — they're simply vacuous.) |
| **`NonlinearKinematicMPCPlanner`** ([nonlinear_kmpc.py:44-79](f1tenth_planning/control/controllers/mpc/nonlinear_mpc/nonlinear_kmpc.py#L44)) | **Ordering bug.** The solver is built at :45, and `NonlinearMPCSolver.__init__` calls `init_problem()` which bakes bounds into `lbx/ubx` immediately. The bounds are assigned at :49-79 — **after** the NLP froze `±inf`. Also guarded on `if pre_processing_fn is None` (:46), the wrong predicate: a custom hook gets no bounds, and a user-supplied `config` gets clobbered anyway. `nonlinear_dmpc.py` sets bounds at :41-75 **before** the solver at :77 — correct — but guards on `if config is None`, so a user config there silently reverts to `±inf`. **Both siblings are wrong, in opposite directions.** |
| **`DynamicAPMPPIPlanner`** default path | **The adaptive-penalty mechanism is provably a no-op.** Constraints are built from `default_x_min/max` ([dynamic_ap_mppi.py:183-185](f1tenth_planning/control/controllers/mpc/ap_mppi/dynamic_ap_mppi.py#L183)), then the *same arrays* become `config.x_min/x_max` (:190-191); `_step` clips every rollout state to those bounds ([ap_mppi_solver.py:258](f1tenth_planning/control/solvers/ap_mppi_solver.py#L258)), so rollouts can never violate the constraints they're policed by. Verified: a state violating steer+speed by 11.01 is seen as violation 0.0. All lambdas look feasible; selection degenerates to `argmax(pure_returns)`. This directly contradicts the shipped example's own guidance. **Only the example works, because it explicitly sets `config.x_max[3] = inf`.** |
| **`DynamicMPPIPlanner`** | Bounds are set only `if config is None`. [examples/control/dynamic_mppi.py:44-51](examples/control/dynamic_mppi.py#L44) passes its own config → `u_min/u_max/x_min/x_max` stay `±inf`, the truncated normal degenerates to an untruncated standard normal, and reference velocity is never clipped. **The shipped baseline MPPI example runs with no actuator limits.** AP-MPPI was fixed for this (commits `d8d5941`/`b1d3818`, plus the `ref_velocity_bounds` escape hatch); plain MPPI was left behind. |
| **All three classical controllers** | No steering saturation anywhere — none clip to `params.MIN_STEER`/`MAX_STEER`. Stanley at `v≈0` is the worst case: `atan2(k*ef, 0) → ±π/2` — a full-lock command. |

### 11.4 Verified crashes and silent wrongness

| Severity | Item |
|---|---|
| **CRASH** | [LTV_mpc_solver.py:234-238](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L234) — the infeasibility fallback `NameError`s on `pred_x`, which is never assigned. Any OSQP status other than `OPTIMAL`/`OPTIMAL_INACCURATE` turns a recoverable failure into an uncaught crash. Even once fixed, :235 passes `self.p` (an ndarray) as `model.f`'s `params`, and the `params` setter asserts `isinstance(value, DynamicsConfig)` → `AssertionError`. |
| **CRASH** | `SITLMPCConfig()` → `TypeError`. [controller_config.py:268](f1tenth_planning/control/config/controller_config.py#L268) uses `field(default_factory=APMPPIConfig)`, but `APMPPIConfig()` needs 8 required positional args. Should be `dynamic_ap_mppi_config`. Same root cause breaks `SITLMPCPlanner`. |
| **CRASH** | The `lax.scan` multi-iteration branch raises `TypeError` in **both** MPPI solvers ([mppi_solver.py:272-279](f1tenth_planning/control/solvers/mppi_solver.py#L272), [ap_mppi_solver.py:382-390](f1tenth_planning/control/solvers/ap_mppi_solver.py#L382)): carry init is `None` while `iteration_step` unpacks it, **and** the parenthesization calls the returned tuple as a function. Plus `unroll=0` in the AP copy, which is invalid. Dead only because both **factory** configs set `scan=False` — but the `MPPIConfig` **dataclass** defaults are `scan=True, n_iterations=5`, so any bare config hits it. Fix: `jax.lax.scan(lambda c, _: self.iteration_step(c, jax_x0, jax_ref, self.p, self.config.Q, self.config.R), (a_opt, a_cov, rng), None, length=n_iterations)`. |
| **SILENT** | **`LQRConfig.__post_init__` unconditionally overwrites every field** ([controller_config.py:168-173](f1tenth_planning/control/config/controller_config.py#L168)). Verified: `LQRConfig(Q=custom, dt=0.5, max_iterations=999)` returns `Q=diag([0.999,0,0.0066,0])`, `dt=0.01`, `max_iterations=50`. The five constructor params are a lie; LQR is unconfigurable despite `plan(config=...)` plumbing existing. Needs `if self.Q is None:` guards like `MPCConfig` uses. |
| **SILENT** | **PRNG key is reset to the same seed on every `solve()`** — `rng = jax.random.PRNGKey(0)` ([mppi_solver.py:246](f1tenth_planning/control/solvers/mppi_solver.py#L246), [ap_mppi_solver.py:356](f1tenth_planning/control/solvers/ap_mppi_solver.py#L356)). The evolved `rng` is threaded through the loop then **discarded** — only `(a_opt, a_cov)` persist. Verified: holding `a_opt` fixed, two `solve()` calls give byte-identical noise. MPPI loses independent exploration across the receding horizon. |
| **SILENT** | **`u_std` has no effect and `adaptive_covariance` is computed but never used.** `da` comes from `jax.random.truncated_normal`, which is a **standard** normal — no scale argument is passed. `a_cov` is computed, shifted and stored but never read by the sampler. Verified: with `u_std=0.5` and bounds `[-1,1]`, empirical `std(da) ≈ 0.60` (consistent with truncated N(0,1)). Exploration std is ~1.0 in **every** control dim — ~1 rad/s on steering rate and ~1 m/s² on acceleration. **The only way to change sampling scale today is to change `u_min`/`u_max`.** The `# TODO: Find a way to use the covariance matrix` sits at [mppi_solver.py:119](f1tenth_planning/control/solvers/mppi_solver.py#L119). |
| **SILENT** | **`Warning("...")` merely constructs an exception object and throws it away — it emits nothing.** Three sites: [LTV_mpc_solver.py:158](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L158), [nonlinear_mpc_solver.py:175](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L175), [:184](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L184). So `planner.plan(state, Q=...)` on a nonlinear planner is accepted, shape-validated, and **silently ignored**. On LTV it raises `ValueError` instead. Same API, two behaviors. |
| **SILENT** | **No warm start in the nonlinear solver despite advertising one.** `ipopt_opts` sets `warm_start_init_point: 'yes'` and `self.U0` is allocated at [:127](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L127), but `U0` is **never written to** — it stays all-zeros forever. Every solve seeds `repmat(x0)` + zeros. IPOPT cold-starts each tick. `self.U0` is a dead attribute. |
| **SILENT** | **IPOPT return status is never checked** ([:202-222](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L202)). With `max_iter=200` and `acceptable_tol=1e-2`, a diverged solve returns garbage straight to the actuators. |
| **SILENT** | **`ca.diagcat(*np.diag(self.config.Q))`** ([:57](f1tenth_planning/control/solvers/nonlinear_mpc_solver.py#L57)) silently discards off-diagonal cost coupling. `MPCConfig` types `Q` as a full ndarray and `MPCController` validates it as one — so a legitimately non-diagonal `Q` is accepted and quietly truncated. |
| **SILENT** | **Reference-velocity extrapolation bug** in [utils.py:126-128](f1tenth_planning/utils/utils.py#L126): the speed advancing the horizon is *always* interpolated between `cv[ind_current]` and `cv[ind_current+1]` — the **initial** segment — even after `t_list` grows past 1.0 and the horizon has walked many waypoints downstream. For `t >> 1` this linearly extrapolates off the first segment. The `% 1.0` that would fix it is applied only *after* the loop ([:137](f1tenth_planning/utils/utils.py#L137)). Positions are re-derived correctly; the **velocity profile driving the arc-length steps is wrong** for any horizon spanning more than one segment — i.e. always, at N=15/dt=0.1/high speed. |
| **SILENT** | **`dl` is computed once from waypoints 0 and 1** ([utils.py:108](f1tenth_planning/utils/utils.py#L108)) and used as the arclength normalizer for the whole horizon — hard-assumes a uniformly spaced raceline. The yaw column is also blended **linearly** ([:144-146](f1tenth_planning/utils/utils.py#L144)) with no ±π unwrapping. And the `yaw` parameter is accepted and never used. |
| **SILENT** | **`solve_lqr` convergence test is inverted** ([utils.py:300](f1tenth_planning/utils/utils.py#L300)): `np.abs(np.max(P_next − P))` takes max-then-abs. When the Riccati residual is all-negative, `np.max` picks the least-negative element, `abs` makes it tiny, and the loop exits early with an unconverged `P` → wrong gain `K`. Should be `np.max(np.abs(...))`. |
| **SUSPECTED (high confidence)** | **`update_matrix` mixes continuous and discrete forms** ([utils.py:308-340](f1tenth_planning/utils/utils.py#L308)). It sets `A[0][0]=1, A[0][1]=ts, A[2][2]=1, A[2][3]=ts` (forward-Euler rows 0 and 2) but leaves `A[1][2]=v` and `B[3][0]=v/wheelbase` **unscaled by `ts`**, and rows 1 and 3 have no identity term at all (verified: `A[1][1]=0`, `A[3][3]=0`, row 3 all zeros). The PythonRobotics/Apollo source builds a *continuous* A and then discretizes; here discretization was applied to only half the matrix. |
| **SILENT** | **Case-mismatched key lookups** in the Pacejka override path ([dynamics_config.py:117-128](f1tenth_planning/control/config/dynamics_config.py#L117)): guards test lowercase (`if "bf" in gym_params`) but index mixed-case (`gym_params["Bf"]`). A dict supplying `"Bf"` is silently ignored (values recomputed from defaults); a dict supplying `"bf"` would `KeyError`. **User-provided Pacejka coefficients can never take effect.** Moot for now since nothing reads them. |
| **SILENT** | **Stanley passes a shape-`(1,)` ndarray to `math.atan2`** ([stanley.py:175](f1tenth_planning/control/controllers/stanley/stanley.py#L175)): `ef` comes from `np.dot(vec, (2,1) matrix)`. On numpy 2.3.5 this emits a `DeprecationWarning` and **will error in a future numpy**. LQR gets it right by extracting `ef[0]`; Stanley never does. |
| **SILENT (trap)** | **`solve()` return-order docstrings are inverted in the ABC and 3 of 4 solvers.** All solvers return `(x, u)` but [mpc_solver.py:54-55](f1tenth_planning/control/mpc_solver.py#L54) (the authoritative ABC contract a new solver author reads), [LTV_mpc_solver.py:197-198](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L197), [mppi_solver.py:239-240](f1tenth_planning/control/solvers/mppi_solver.py#L239), and [ap_mppi_solver.py:349-350](f1tenth_planning/control/solvers/ap_mppi_solver.py#L349) all document `(u, x)` — each contradicting its own `return` a few lines below. A new solver written to the docstring returns `(u, x)`; `MPCController` binds it as `x_pred, u_pred` ([mpc.py:199](f1tenth_planning/control/controllers/mpc/mpc.py#L199)) with no shape check, then feeds `u_pred[:, 0]` (the first two entries of a *state* vector — x/y position) to the actuators. Car drives wrong, no error. |
| **SILENT** | **`dynamic_mppi.py` loads a raceline and throws it away.** [dynamic_mppi.py:36](examples/control/dynamic_mppi.py#L36) builds `Track.from_raceline_file("trajectory_log.csv", delimiter=";", skip_rows=3)`, then [:41](examples/control/dynamic_mppi.py#L41) reassigns `waypoints_track = env.unwrapped.track`, discarding it before first use. So [examples/control/trajectory_log.csv](examples/control/trajectory_log.csv) is never consumed, the `delimiter`/`skip_rows` args are dead, and `import os` + `from ...track import Track` become dead imports — while still paying the full spline-fit cost at startup. It is also the only example exercising `from_raceline_file`, i.e. the one place the `Raceline` naming trap (`psis=` in, `.yaws` out) can bite. |

### 11.5 Estimation (NLS)

Six bugs; three fixed on `dev-cleanup` only:
1. **Abstract method typo** — ABC declares `@abstractmethod def estiamte` ([paramter_estimator.py:23](f1tenth_planning/estimation/estimators/parameter_estimators/paramter_estimator.py#L23), misspelled) while the impl is `estimate` → the abstract is never overridden → **`NLSParameterEstimator` is uninstantiable**. *(Fixed on `dev-cleanup` `60d0c95`.)*
2. `super().__init__(initial_params)` omits the required `model` arg. *(Fixed, `d87b2a1`.)*
3. `rk4_discretization(x_k, u_k, params, dt, f_casadi)` — wrong arg order; **`func` is FIRST** ([discretizers.py:22](f1tenth_planning/control/discretizers.py#L22)). *(Fixed, `d87b2a1`.)*
4. **It is not least squares, and it is unfixed on both branches.** [nls_estimator.py:68](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L68) does `obj += err` where `err = x_pred − x_k_plus_1` — summing **raw signed residuals** (they cancel), contradicting its own docstring. Worse, `err` is a `1×nx` row vector so `obj` is a vector and `opti.minimize(obj)` requires a scalar. Needs `obj += ca.sumsqr(err)`.
5. **Parameter vector length and ordering are both wrong.** `for field, val in vars(initial_params).items()` ([:48](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L48)) makes **all 23** `DynamicsConfig` fields decision variables in *declaration* order, but `f_casadi_opti` reads a **9-vector** `params[0]=mu, [1]=m, …, [8]=g`. **`MIN_STEER` is read as friction coefficient.**
6. **Infeasible positivity constraints** — `opti.subject_to(v > 0)` on **every** field ([:50](f1tenth_planning/estimation/estimators/parameter_estimators/NLS/nls_estimator.py#L50)), including `MIN_STEER ≈ −0.4`, `MIN_ACCEL < 0`, `MIN_SPEED < 0` → immediately infeasible.

Also: hardcoded to the 7-state model, re-solves the full IPOPT NLP **every control step**, and `new_param_guess` (in the ABC signature) is never implemented — no warm-start path. Filename typo: `paramter_estimator.py`.

### 11.6 Structural / hygiene

- **No tests, no CI, no linter config, no pre-commit, no dev dependency group** anywhere in first-party code. The only test suite (133 tests, all passing) and the only CI (3 workflows) live inside the **untracked** `f1tenth_gym/` clone and belong to the upstream gym. This total absence is precisely why the four independent import-time breakages are sitting unnoticed on the branch.
- **[f1tenth_planning/planning/lattice_planner/test_pyclothoids.py](f1tenth_planning/planning/lattice_planner/test_pyclothoids.py) is a trap.** It ships **inside the wheel**, matches pytest's default discovery, defines `test_cont()`/`test()` with **zero assertions**, and calls blocking `plt.show()`. Adding pytest with default discovery would hang the suite (after ImportError-ing on pyclothoids). Delete it or move it out of the package.
- **`requires-python = ">=3.9,<=3.13"`** ([pyproject.toml:11](pyproject.toml#L11)) is a PEP 440 footgun: `<=3.13` admits **only** 3.13.0. Verified: 3.13.0 → True, 3.13.1 → False. **The project's own venv runs 3.13.12 — outside its declared range**, so `uv sync` would select CPython 3.12.13 and rebuild on a different interpreter. Intent was `<3.14`.
- **Probable packaging bug:** `[tool.hatchling.build.targets.wheel]` ([pyproject.toml:38](pyproject.toml#L38)) — hatchling reads `[tool.hatch.build.targets.wheel]`. The table is ignored; it works only because hatchling auto-detects the matching directory name.
- **README documents `pip install .`**, which **cannot work** — `f1tenth_gym` is declared under `[tool.uv.sources]`, a uv-only mechanism, and is not on PyPI. The real workflow is `uv sync`, which the README never mentions.
- **`jax[cuda13]` is a mandatory, non-optional dependency**, making the package effectively Linux+NVIDIA-only and pulling hundreds of MB of CUDA — even for a user who only wants Pure Pursuit (which uses numba, not JAX). Should be an extra.
- **Dependency list is close to exactly inverted.** Declared-but-never-imported: `osqp`, `scs`, `qdldl`, `kiwisolver`, `llvmlite` (all transitive deps of cvxpy/osqp/numba/matplotlib). Imported-but-undeclared: `pyclothoids`, `scipy`, `matplotlib`.
- **Import-cycle hazard:** [controller_config.py:5](f1tenth_planning/control/config/controller_config.py#L5) imports `ModelConfig` **absolutely**, which triggers the `f1tenth_planning.control` package `__init__` and therefore eagerly imports **every** concrete controller. Importing a single config dataclass pulls in the whole stack (and CasADi + JAX) — and is what makes the `SteerActionEnum` error surface on any config import. Should be `from .model_config import ModelConfig`.
- **`solvers/__init__.py` is an import-time landmine.** There is no way to import one solver: [solvers/\_\_init\_\_.py](f1tenth_planning/control/solvers/__init__.py) eagerly imports all four, so `from f1tenth_planning.control.solvers import LTVMPCSolver` drags in CasADi + cvxpy + OSQP + JAX (~1.4 s measured) even for the LTV-only path — and Python runs the parent `control/__init__.py` *first*, pulling in every controller. Importing by full module path (`...solvers.LTV_mpc_solver import LTVMPCSolver`) avoids only the first layer, not the parent. Both `mppi_solver.py` and `ap_mppi_solver.py` additionally fire module-level side effects on import (`mkdir ~/jax_cache`, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`) — see [§11.7](#117-sharp-edges-worth-knowing-but-not-fixing-today).
- **Namespace pollution:** [config/\_\_init\_\_.py](f1tenth_planning/control/config/__init__.py) star-imports three modules with no `__all__` anywhere, so `from f1tenth_planning.control.config import *` also exports `np`, `dataclass`, `field`, `Callable`, `List`, `F110Env`.
- **Six directories have no `__init__.py`** and resolve only via PEP-420 namespace packages (verified: `PathFinder.find_spec` returns `origin=None`): [dynamics_models/](f1tenth_planning/control/dynamics_models/), [controllers/](f1tenth_planning/control/controllers/) (itself — so *every* `from .controllers.… import` traverses a namespace pkg), [controllers/mpc/nonlinear_mpc/](f1tenth_planning/control/controllers/mpc/nonlinear_mpc/), and all three levels of `estimation/estimators/parameter_estimators/NLS/`. **Current wheel is unaffected** — `hatchling` selects by directory tree, and a `uv build --wheel` confirms all init-less `.py` files ship. The hazard is *conditional*: a migration to setuptools `find_packages()` (which skips init-less dirs) would drop them; `find_namespace_packages()` would be required. If hardening with `__init__.py`, add all six — fixing only `nonlinear_mpc/` leaves the other five exposed.
- **Docs are 100% scaffolding.** All 10 content pages + [docs/contribute.md](docs/contribute.md) are exactly 2 lines (a MyST anchor + an H1). [docs/index.rst](docs/index.rst) advertises three pillars; **Perception does not exist as code at all**, `graph_planner` is vaporware, and `fgm`/`wall_follow` are fake. No autodoc is configured, so the docs build never imports the package and can never fail on drift. [docs/conf.py:56](docs/conf.py#L56) sets `highlight_language = "gdscript"` (Godot!) — copy-paste residue from a Godot docs template — **but it is not a one-line delete**: [docs/extensions/gdscript.py](docs/extensions/gdscript.py) is a 359-line MIT-licensed Godot Pygments lexer that [conf.py:49](docs/conf.py#L49) imports *unconditionally*, so deleting `docs/extensions/` while leaving `conf.py` fails the build (exit 120). Remove as one unit: `conf.py:56`, `:49-52`, `:17`, and the file. The vendored [docs/extensions/sphinx_tabs/](docs/extensions/sphinx_tabs/) (~128K, 2016-era) is **dead code, not a shadow** — `conf.py:17` uses `sys.path.append` (not `insert(0,…)`), so the pip-installed `sphinx-tabs` wins; the vendored copy is only ever reached as a broken fallback (`No module named 'pkg_resources'`). This is the *opposite* of the `f1tenth_gym/` two-copies trap ([§9.5](#95-which-gym-is-installed--resolved-but-the-target-is-a-real-decision)): same smell, opposite outcome, because of `append` vs `insert`. [pyproject.toml:33](pyproject.toml#L33) points `Documentation` at the *simulator's* docs.
- **The ROS wrapper is thoroughly broken:** imports `Nonlinear_Dynamic_MPPI_Planner` (pre-`c6c25ca` name), unpacks `action, info = planner.plan(...)` against a dead 2-value API (`plan` returns a flat 2-vector, so `info` binds to a float and `info["steering_angle"]` `TypeError`s), references `self.params` which is never assigned, and its `raceline` default (`trajectory_logs.csv`) doesn't match the shipped file (`trajectory_log.csv`).
- **Stale docstrings, sampled:** `MPCController`'s says *"MPPI Controller, uses CasADi"* ([mpc.py:17](f1tenth_planning/control/controllers/mpc/mpc.py#L17)) — wrong twice over. Five examples claim to be an *"STMPC example"* (no such controller exists). Seven claim *"fixed waypoints throughout the 2 laps … see the lane switcher example"* (nothing stops at 2 laps; no lane switcher example exists). `Controller.__init__`'s docstring documents `params (dict | str): dictionary or path to yaml` — the signature takes `DynamicsConfig` and no yaml path exists; it also omits `control_mode` entirely. `LQRController`'s class docstring documents args `wheelbase` and `waypoints` that don't exist. `PurePursuitPlanner`'s docstring lists its constructor args **in the wrong order** (`max_reacquire` before `lookahead_distance`) — a positional caller following it swaps a 0.8 m lookahead with a 20 m reacquire radius. `_rollout`'s docstring gets both the `xref` orientation and both return shapes wrong.
- **`ENVIRONMENT` hazard:** `PYTHONPATH` leaks ROS Humble's Python 3.10 site-packages into every invocation, which broke `uv run pytest` outright (`ModuleNotFoundError: No module named lark`). Use `env -u PYTHONPATH`.
- **The untracked `f1tenth_gym/` directory is not in `.gitignore`**, so `git status` is permanently dirty and a contributor could accidentally `git add` a whole second repo. Combined with the API mismatch above (which invites editing it), this is a very easy trap.

### 11.7 Sharp edges worth knowing but not fixing today

- `MPCSolver.update` is `@abstractmethod` **with a working body** meant to be reached via `super().update(...)`. It **mutates `self.config.Q/R/P/Rd` in place** ([mpc_solver.py:34-37](f1tenth_planning/control/mpc_solver.py#L34)) — a config shared between two solvers gets silently changed by whichever calls `update` last. Its body also references `self.p` before `__init__` can create it ([:33](f1tenth_planning/control/mpc_solver.py#L33)) → `AttributeError` on a fresh `NonlinearMPCSolver` (survives only because `solve` always materializes a `p` first; `LTVMPCSolver` sets `self.p` at :40).
- **Signature divergence:** `LTVMPCSolver.solve(self, x0, xref, p=None, Q=None, P=None, R=None, Rd=None)` orders **P before R**, diverging from the ABC and from `NonlinearMPCSolver`. Safe today only because `MPCController` uses keywords.
- `Rd` is declared, validated and shifted through every config but is **never read by either MPPI solver**. Combined with the factories setting `R = Rd = diag([0,0])`, there is **no control-effort or smoothness penalty whatsoever** in the default MPPI/AP-MPPI cost — regularization comes entirely from `u_std`/`temperature` (which, per §11.4, `u_std` doesn't actually do).
- **`P` is set identical to `Q` in all four shipped presets** — no distinct terminal weighting anywhere. `delta` (index 2) and `yaw_rate` (index 5) always carry weight 0.0. `dynamic_ap_mppi_config` duplicates `dynamic_mppi_config`'s numbers verbatim rather than deriving from it — they will drift on any retune.
- **Stale-JIT footgun:** `iteration_step`/`_rollout` are jitted with `self` static, keyed by identity. **Mutating config after the first solve will not trigger a retrace** — the kernel keeps the old `N`, `n_samples`, `constraints`. Both examples mutate config *before* construction, which is safe, but nothing enforces it. `APMPPIConfig.n_constraints` is likewise computed once in `__post_init__` and goes stale if `constraints` is mutated after.
- **AP-MPPI feasibility uses exact float equality** `violations == 0` ([:211-212](f1tenth_planning/control/solvers/ap_mppi_solver.py#L211)). Sound only because the shipped factories return `norm(maximum(0, ...))`. Any user constraint returning a small negative margin breaks it. The `-1e10` infeasible sentinel would also collide with genuinely large negative returns.
- **Both MPPI solvers have module-level import side effects** ([mppi_solver.py:12-16](f1tenth_planning/control/solvers/mppi_solver.py#L12)): they `mkdir ~/jax_cache` and set `XLA_PYTHON_CLIENT_PREALLOCATE=false` **at import time** — and the env var is set *after* `import jax`, so it may not take effect depending on backend init order.
- `Controller.__init__` is decorated `@abstractmethod` yet is the real initializer every subclass calls via `super()`. It enforces nothing beyond what `plan` already provides.
- **Both Stanley and LQR use `WHEELBASE` as the CoG-to-front-axle distance** when the correct quantity is `LF` — over-projecting the front axle by `LR`. LQR's error is named `e_cog` ("lateral error of CoG") but computed at the **front axle**.
- **`DynamicsConfig` has 23 required fields, no `__post_init__`, and no validation.** Nothing enforces `MIN_STEER < MAX_STEER`, `WHEELBASE == LF + LR` (even though `_dynamics_config_from_gym_params` derives it that way), or `M, I > 0`. Its class docstring omits `BF/BR/DF/DR/CF/CR` entirely and gives no units for `MU/C_SF/C_SR/M/I`.
- **Dead:** `update_config_from_dict` ([dynamics_config.py:223](f1tenth_planning/control/config/dynamics_config.py#L223)) is a module-level function taking `self`, never bound to the class, zero callers. `truncated_gaussian_sampler` ([mppi_solver.py:19](f1tenth_planning/control/solvers/mppi_solver.py#L19)) — zero callers, and itself wrong (whitens bounds with `diag(R)` but un-whitens with full `R @ samples`); looks like an abandoned first attempt at the `u_std` TODO. `LTVMPCSolver.predict_state` ([:240](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L240)) — never called. `LTVMPCSolver`'s cvxpy `Q/R/Rd/P` Parameters ([:54-57](f1tenth_planning/control/solvers/LTV_mpc_solver.py#L54)) — declared, never in the objective, never assigned a `.value`; this is the structural reason custom cost matrices are "not supported yet".
- `nonlinear_mpc_solver.py:186-189` **discards `xref[:,0]`** — the k=0 tracking term is identically zero (the constraint already pins `X[:,0]`), so the effective tracking horizon is N, not N+1. Harmless, confusing.
- `MPCController.plan` validates `waypoints.shape[1] < 3` but reads `self.waypoints[:, 3]` — a 4-column array is the true minimum. All three classical controllers index `shape[1]` **before** the ndim guard, so a 1-D array raises `IndexError` instead of the intended `ValueError`.
- **Runtime kwarg overrides mutate the instance.** `plan(lookahead_distance=..., k_path=..., config=..., waypoints=...)` **permanently** replaces the attribute — they are sticky, not per-call.
- `DynamicMPPIPlanner.render_sampled_trajectories` raises `AttributeError` if called before the first `solve()` (`MPPISolver.__init__` never sets `self.samples`; `APMPPISolver` does). Its `len(sampled_trajectories // 100)` ([:109](f1tenth_planning/control/controllers/mpc/mppi/dynamic_mppi.py#L109)) floor-divides the **array** elementwise then takes `len()` → 1024, not 10. Accidentally harmless (over-allocates). It also passes JAX device arrays straight to the renderer with no `jnp_to_np`. **`DynamicAPMPPIPlanner` has no sampled-trajectory renderer at all** despite `APMPPISolver` populating `samples` identically — the port is mechanical.
- `local_plan` and `control_solution` are computed **twice per step** in all three classical controllers — once at the end of `plan()`, again in the render callback. Pure waste when rendering is off; immediately overwritten when it is on.
- Typo `goal_veloctiy` is baked in as a real identifier across both [stanley.py](f1tenth_planning/control/controllers/stanley/stanley.py) and [lqr.py](f1tenth_planning/control/controllers/lqr/lqr.py) — don't fix it in one place.
- `lmpc/__init__.py`'s `__all__` exports **`ITLMPCPlanner`** (missing the leading S) — a name that exists nowhere — and omits the real `SITLMPCPlanner`.
- `version` is duplicated in [pyproject.toml:4](pyproject.toml#L4) and [f1tenth_planning/\_\_init\_\_.py:3](f1tenth_planning/__init__.py#L3). Currently in sync; nothing enforces it.

---

## 12. Dependencies & environment

**Verified installed** in the root `.venv` (Python **3.13.12**, Clang 22.1.1, uv-managed): `jax 0.10.0`, `casadi 3.7.2`, `cvxpy 1.7.5`, `numba 0.62.1`, `numpy 2.3.5`, `scipy 1.16.3`, `f1tenth_gym` (no `__version__`), `f1tenth_planning 0.1.1`.

**Verified missing** from the root `.venv`: `pyclothoids`, `matplotlib`, **`pytest`**. (`uv.lock` has 0 matches for pyclothoids, matplotlib, or pytest.) Note the installed `jax 0.10.0` is **newer than any entry in the lock** (which encodes a python-conditional matrix: 0.4.30 for <3.10, 0.6.2 for 3.10, `jax[cuda13]` 0.8.1 for ≥3.11), indicating the venv was populated outside the lock.

| Dependency | What it's actually for |
|---|---|
| **numpy** | Everywhere. |
| **numba** | CPU JIT. **Exactly 2 import sites**: [utils.py:12](f1tenth_planning/utils/utils.py#L12) (`@njit(cache=True)` on `nearest_point`, `intersect_point`, `get_actuation`, `solve_lqr`, `update_matrix`, `pi_2_pi`, `quat_2_rpy`, `get_rotation_matrix` — the hot loops for PP/Stanley/LQR) and [lattice_planner.py:38](f1tenth_planning/planning/lattice_planner/lattice_planner.py#L38) (cost kernels). Zero `@njit` in any controller file. |
| **cvxpy** | **Only** `LTVMPCSolver`. Builds the QP with `Variable`/`Parameter`/`quad_form`, solved via `.solve(solver=cvxpy.OSQP, warm_start=True)`. |
| **osqp** | The QP backend cvxpy dispatches to. **Never imported directly.** Explicitly pinned but already a hard cvxpy dep — redundant. |
| **casadi** (≥3.7.2) | Symbolic dynamics (`f_casadi`, `f_casadi_opti`) + `ca.nlpsol('solver','ipopt', ...)` in `NonlinearMPCSolver`, and `ca.Opti` in the NLS estimator. **Ships its own IPOPT** — there is no separate ipopt dep. |
| **jax** (`jax[cuda13]>=0.4.30`) | `f_jax`, `MPPISolver`, `APMPPISolver` (jit/vmap/lax.scan/PRNG), plus `jnp_to_np` in utils. Writes a compilation cache to `~/jax_cache`. |
| **scipy** | `linalg.expm` (exact ZOH, [discretizers.py:2](f1tenth_planning/control/discretizers.py#L2)), `sparse.block_diag`/`csc_matrix` (LTV), `spatial.distance.cdist` (lane_switcher). **Undeclared** — survives only transitively via cvxpy. |
| **pyclothoids** | `Clothoid.G1Hermite` for the lattice planner + the two fake planners. **Undeclared and uninstalled.** |
| **pyyaml** | Declared and imported — but **only** by the broken, unexported `LaneSwitcher`. |
| **f1tenth_gym** | The core coupling. Git dep, `branch = "dev-dynamics"` ([pyproject.toml:36](pyproject.toml#L36)), locked to `67bc6db`. **Not on PyPI** — so `pip install .` cannot resolve it. |
| `llvmlite`, `scs`, `qdldl`, `kiwisolver` | **Never imported.** Transitive deps of numba/cvxpy/osqp/matplotlib, pinned redundantly. |

**Docs toolchain** is isolated in [docs/requirements.txt](docs/requirements.txt) (sphinx, sphinx-rtd-theme, sphinx-tabs, sphinx-copybutton, myst-parser — all unpinned) and is not referenced from `pyproject` (no `[project.optional-dependencies]`, no `[dependency-groups]`).

**Two virtualenvs coexist and disagree:** root `.venv` is Python **3.13.12**; `f1tenth_gym/.venv` is Python **3.14.3**. The gym's 133 tests pass under 3.14 in the gym's own venv — that result says **nothing** about the root project's environment.

**Tooling on the machine:** `uv 0.11.2` at `~/.local/bin/uv`. ROS Humble is installed and leaks into `PYTHONPATH`.