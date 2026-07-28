# f1tenth_planning — API Design

> **Status:** IMPLEMENTED on `dev-api-redesign`. Written as a proposal 2026-07-28 and
> built out over the commits listed in §11; the build order there records what landed.
> Where the implementation deviated from the proposal, the section says so.
> Developed on branch `dev-api-redesign` (off `fix/correctness-pass` @ `9dd7bf1`, so the correctness
> fixes come along). Companion to [CODEBASE.md](CODEBASE.md) (what exists today),
> [BUGS_FINDINGS.md](BUGS_FINDINGS.md), and [DEADCODE_FINDINGS.md](DEADCODE_FINDINGS.md).
>
> **This is a clean-break branch — no backwards compatibility.** The library is in heavy development
> and research code lives on other branches, so the API is free to change shape. No deprecation
> aliases, no dual-API support, no staged renames: each change lands in its final form and the nine
> examples are updated in the same commit. That simplifies several decisions below, and the phases in
> §11 are ordered by *build dependency*, not by blast radius.

## 1. Purpose

`f1tenth_planning` should be a **common API for implementing controllers and planners** for the
F1TENTH platform — good enough to host research code (SIT-LMPC and successors) without fighting it.
Three properties drive every decision below:

1. **Scalable** — adding a controller, model, or solver is additive, not a refactor.
2. **User-friendly** — small required surface; the common case is short; failures are loud and early.
3. **Fully configurable** — every parameter (dynamics, cost matrices, bounds, reference) settable
   **before init and at runtime**.

### Non-goals

Explicitly out of scope, so they don't creep in:

- No `Policy`/`Agent` abstraction — this is a control and planning library, not an RL stack.
- No new `Reference`/`Trajectory` type — the gym's `Raceline` is the reference (§8).
- No forced auto-derived Jacobians — hand-written Jacobians stay first-class (§5.4).
- No `xp`-parameterized single-source dynamics — rejected on measurement (§12).
- No explicit hot/cold parameter API — the controller decides internally (§7).
- FGM / wall-follow / lattice / lane-switcher implementations are **later**; the design must
  accommodate them, not deliver them now.

---

## 2. The two entities

**`Controller` and `Planner` are separate ABCs with no shared parent.** They are different jobs:

| | `Controller` | `Planner` |
|---|---|---|
| Job | produce an actuation command | produce a reference / path |
| Output | control vector (2,) | a reference (§8) |
| Runs the car alone | yes | no |

**There is exactly one `Controller` class — no subtypes.** A controller that consumes a raceline
(Pure Pursuit, Stanley, LQR, MPC), one that reads a laser scan and holds no reference (FGM,
wall-follow), and one that learns a safe set and value function across laps (LMPC) are all just
`Controller`s. Reference-holding, sensor-reading, and learning are *internal implementation details*,
not type distinctions. Any learned or accumulated state lives inside the controller and is serviced
by the lifecycle hooks in §3.

**Controllers do not take a `Planner`.** The user owns the loop and wires the two together:

```python
reference = planner.plan(obs)            # user code
controller.update(reference=reference)   # explicit hand-off
u = controller.compute_control(obs)
```

This inversion is deliberate: it needs no agreement on a `Planner` output type (the hard,
unresolved question), it lets the `Planner` ABC be designed later without blocking controller work,
and it reuses the `update()` mechanism that has to exist anyway for runtime reconfiguration.

---

## 3. The `Controller` ABC

Small required core, plus optional lifecycle hooks that default to no-ops.

```python
class Controller(ABC):
    # ---- required ------------------------------------------------------
    @abstractmethod
    def __init__(self, track=None, params: DynamicsConfig = None, config=None): ...

    @abstractmethod
    def compute_control(self, state: dict) -> np.ndarray:
        """Return the control vector for the current state."""

    # ---- configuration (concrete in the base; rarely overridden) --------
    def update(self, *, config=None, params=None, reference=None) -> None: ...   # §7
    def get_config(self): ...

    # ---- lifecycle (optional; no-ops by default) ------------------------
    def reset(self) -> None: ...
    def complete_iteration(self) -> None: ...   # lap/episode boundary; LMPC commits its safe set
```

**Naming.** `plan()` is the *planner's* verb; giving it to `Controller` too is what blurred the two
entities. `Controller.compute_control()` / `Planner.plan()` read distinctly at call sites. Clean
break: `plan()` is removed from `Controller` outright, and the examples are updated in the same
commit — no alias.

**Why these hooks.** `reset()` clears per-episode state (LQR's cached errors, MPPI's warm-started
control sequence, the PRNG). `complete_iteration()` is the lap boundary — it is where LMPC commits a
completed trajectory to the safe set and retrains its value function. The deleted LMPC skeleton
needed exactly this and had nowhere to put it; `examples/control/sit_lmpc.py` already calls
`planner.complete_iteration()`.

**What is deliberately *not* on the ABC:** rendering (§9), and any assumption that a controller holds
a reference at all (FGM does not).

**`control_mode` must become real.** Today it is stored and never read, so a controller and its env
can silently disagree about whether the action is `(steer_angle, speed)` or `(steer_vel, accl)`.
It should be readable by the user (and ideally validated against the env's `control_input`).

---

## 4. The `Planner` ABC

Deliberately minimal until we implement a real planner — the §2 inversion means nothing depends on
its output type yet, so firming it up early would be guesswork.

```python
class Planner(ABC):
    @abstractmethod
    def plan(self, state: dict, **context): ...
    def update(self, *, config=None, **kwargs) -> None: ...
    def reset(self) -> None: ...
```

Planners need richer input than ego state (scan, opponent poses, map/track), hence `**context`.
Firming up the return type is deferred to whenever the lattice/lane-switcher port happens — see
§13 open questions.

---

## 5. Dynamics models

### 5.1 Capability-based backends

Today every model is expected to implement the same dynamics **four times** (`f`, `f_casadi`,
`f_casadi_opti`, `f_jax`). That is a 4× tax on every new model and it is what let the three
implementations drift apart (the `tan²` and switch-threshold bugs).

**A model implements only the backends it needs.** A jax-only research model (e.g. a learned
dynamics network) implements `f_jax` and nothing else.

```python
class DynamicsModel(ABC):
    def f(self, x, u, p=None):              raise NotImplementedError   # numpy
    def f_jax(self, x, u, p):               raise NotImplementedError   # jax
    def f_casadi(self):                     raise NotImplementedError   # casadi Function
    def f_casadi_opti(self, x, u, p):       raise NotImplementedError   # casadi SX
    def jacobian(self, x, u, p=None):       raise NotImplementedError   # optional, hand-written

    @classmethod
    def backends(cls) -> set[str]: ...      # derived from which methods are overridden
```

**Capability check at composition time.** Each solver declares what it needs
(`LTVMPCSolver` → numpy + jacobian; `NonlinearMPCSolver` → casadi; `MPPISolver` → jax) and
`MPCSolver.__init__` validates it, raising a clear error at construction. Today a missing backend
fails deep inside a solve with `NotImplementedError`.

### 5.2 State and control layout — named, per model

Generic code must stop hardcoding indices (`config.x_min[3]` meaning velocity) and stop bridging
layouts by slicing (`_extract_kinematic_state`, which only works because the kinematic state happens
to be a prefix of the dynamic one).

```python
model.state.idx.v          # -> 3   int index, autocompletes, no string keys
model.control.idx.a        # -> 1
model.state.names          # ("x", "y", "delta", "v", "yaw", "yaw_rate", "beta")
model.state.size           # 7
```

`state`/`control` are small spec objects holding `names`, `size`, and an `idx` NamedTuple of ints.
Symmetric, string-free, per-model correct.

> **Measured (this machine):** an extra attribute hop costs ~25 ns; at ~5 lookups per tick that is
> **0.38 µs — 0.004 % of a 10 ms budget.** Naming is therefore a readability choice, not a
> performance one. Standard rule applies: hoist to a local inside any hot loop.

**The runtime container stays a raw array.** Measured: attribute access on a dataclass is faster
*per element* (7.3 ns vs 36 ns), but arrays are **12× faster batched** (7.9 µs vs 97.7 µs over 1024
samples) and `jax.vmap`/`jit` require array-like values. Names are static metadata resolved once per
tick; numerics stay in `np.ndarray` / `jnp.ndarray`. An optional `view(x)` helper returning a
NamedTuple is fine for debugging at the boundary — never in the hot path.

**What this buys:** `MPCController` can build `x0` from the observation generically (each model pulls
the named fields it needs), `pre_processing_fn` disappears, and a model whose state is *not* a prefix
of the 7-state layout (Frenet, point-mass, neural) works without special-casing.

### 5.3 Reference ↔ state mapping

With named layouts, mapping a `Raceline` into a model's state vector is by name, not by column
position — which retires the worst current trap: waypoint column 3 means **yaw** to the classical
controllers and **velocity** to the MPC family (CODEBASE.md §5.2).

### 5.4 Jacobians

Optional model capability, **hand-written by default**. Hand-derived Jacobians can be faster, more
accurate, and more numerically stable, so they stay first-class; auto-derivation (`jax.jacfwd`,
`ca.jacobian`) is available to a model that wants it, never forced. Completing
`DynamicBicycleModel.linearize_around_state` — currently `raise NotImplementedError` followed by
~60 lines of unreachable half-written code — is a separate correctness task (and is what blocks an
LTV *dynamic* MPC).

### 5.5 Anti-drift: a test, not an abstraction

Where a model provides multiple backends, a **cross-backend agreement unit test** is the guarantee:
sample random `(x, u, p)` across low-speed, high-speed, and reverse regimes and assert all provided
backends agree within tolerance. This is exactly the check that validated the correctness pass
(numpy/jax/casadi now agree to ≤6e-4). It costs zero library complexity, keeps every backend
hand-tuned, and covers CasADi — which no source-sharing scheme can.

---

## 6. Solvers

The MPC family's `controller = model + solver + config` composition is the part of the current design
that already works. Keep it and extend it.

- **`APMPPISolver` inherits from `MPPISolver`**, overriding only the constraint/penalty logic.
  Today they are ~90 % duplicated siblings that have already drifted (the correctness pass had to
  hand-port four fixes between them).
- **A shared reference-tracking base** for the machinery every tracking controller repeats:
  raceline handling, nearest-point/lookahead search, reference generation. Pure Pursuit, Stanley,
  and LQR each re-implement this today.
- **Solvers expose parameter slots** (§7) so cost matrices, bounds, and model parameters can be
  swapped without rebuilding the problem.

---

## 7. Configuration and runtime reconfiguration

The requirement — every parameter settable before init *and* at runtime — is the biggest structural
lever in this design.

### 7.1 One entry point, three slots

No 20-argument signature, no explicit hot/cold API. The **config dataclasses are the parameter
namespace** (they already are). Mutate a copy and hand it back:

```python
controller.update(config=..., params=..., reference=...)
#                 algorithm    vehicle    reference
```

### 7.2 The rule: shape change → rebuild, value change → hot-swap

The base class diffs the incoming config against the current one and decides internally:

- **Structural** (`N`, `nx`, `nu`, `n_samples`, or any array whose *shape* changed) → the problem
  structure changed → rebuild the solver. Unavoidable and acceptable.
- **Tuning** (`Q`, `R`, `P`, `Rd`, bounds, dynamics params — *values* only) → route into the
  solver's parameter slot. No rebuild, no recompile.

A workable default heuristic: **int-typed fields are structural, float/array fields are tuning**,
with a per-solver override for exceptions. The implementer declares nothing; the base class derives
it. That satisfies "easy to implement" and "simple to use" simultaneously.

### 7.3 Parameter slots per backend

Each solver routes hot values into its backend's native mechanism:

| backend | slot | status today |
|---|---|---|
| cvxpy (LTV) | `cvxpy.Parameter` | `Q`/`R`/`Rd`/`P` Parameters **exist but are never wired** — the abandoned seed of this design |
| casadi (NMPC) | the `p` parameter matrix | **already works** for model params; the pattern to copy |
| jax (MPPI/AP-MPPI) | traced arguments | partially — costs are traced, but config is read off `self` |

### 7.4 The blocker: `static_argnums=(0)`

`iteration_step` and `_rollout` are jitted with `self` static, hashed **by identity**. Any config
read through `self.*` inside the kernel is baked into the trace at first call: mutating
`config.constraints` afterwards is a no-op, and even rebuilding a closure does not force a retrace.
This is simultaneously

- why MPPI/AP-MPPI runtime reconfiguration silently does not work, and
- the LMPC stale-value-function trap (a retrained value function would be ignored forever —
  the controller would run lap 50 on lap 1's model with no error).

**Measured, in isolation:** mutate `temperature` on a solver whose kernel reads it off static `self`
→ **the output does not change.** That is the bug, reproduced.

#### The rule

> **Static = things that determine the *shape* of the computation.
> Traced = numbers that flow *through* it.**

| Value | static / traced | why |
|---|---|---|
| `N`, `n_samples`, `nu`, `nx` | **static** | they *are* array shapes; a traced `N` raises `TypeError: Shapes must be 1D sequences of concrete values` |
| `scan`, `adaptive_covariance` | **static** | drive a Python `if` |
| `step_fn`, `reward_fn`, `constraints` | **static** | Python callables, hashed by identity |
| `Q`, `R`, `p`, `ref_traj`, `x0` | **traced** | already correct in today's code |
| `u_min`, `u_max` | **traced** | *cannot* be static — JAX rejects unhashable arrays |
| `temperature`, `damping` | **traced** | currently baked off `self`; pure arithmetic, no shape role |

Only the last row plus `u_min`/`u_max` are misplaced today — the cost matrices and model params were
already threaded correctly. **Killing static `self` is a surgical change, not a solver rewrite.**

Three regimes, worth separating because they fail differently:

| | correct? | cost |
|---|---|---|
| traced | ✅ | compiles once |
| static (mis-marked) | ✅ still correct | recompiles on **every** change — a 20-point tuning sweep = 20 compiles (measured: 5 temperature values → 5 traces vs 1) |
| read off identity-hashed `self` | ❌ **silently stale** | never updates at all |

Note the built-in safety net: array-valued params *cannot* be mis-marked static (JAX raises
`ValueError: Non-hashable static arguments are not supported`). Only hashable scalars like
`temperature` can slip through silently.

#### Decision: explicit arguments, with `static_argnames`

Kernels become module-level functions taking everything explicitly — no `self`. Use
**`static_argnames`, not `static_argnums`**: positional index lists get silently out of sync with a
~19-parameter signature (this exact mistake was made while drafting this section), whereas names are
reorder-safe and self-documenting. Keyword-only placement makes the split visible at the call site.

```python
@partial(jax.jit, static_argnames=("N", "n_samples", "nu", "scan",
                                   "adaptive_cov", "step_fn", "reward_fn"))
def _iteration_step(a_opt, a_cov, rng,                      # carry      (traced)
                    x0, ref_traj, p, Q, R,                  # problem    (traced)
                    u_min, u_max, temperature, damping,      # tuning     (traced)
                    *, N, n_samples, nu, scan,
                    adaptive_cov, step_fn, reward_fn):       # structural (static)
    ...
```

Verified: 5 tuning values → **1 trace**; a structural change → exactly **1 retrace**.

The public API is unaffected — `update(config=...)` is identical under any of these schemes; only the
internal kernel boundary changes (~4–6 call sites across MPPI and AP-MPPI).

**Fallback if the signature ever becomes unwieldy:** group *only the traced tuning values* into a
pytree-registered container and pass it as a single argument. This keeps the static set explicit while
shortening the signature. Don't reach for it pre-emptively.

Fixing this is a prerequisite for both the configurability requirement and SIT-LMPC — for LMPC the
value function's **weights** become a traced argument (no staleness, no retrace) while its *code*
stays static.

---

## 8. References

**The gym's `Raceline` is the reference type.** It already has fixed named attributes — `xs`, `ys`,
`yaws`, `vxs`, `ks`, `ss`, `length` — so controllers access `reference.vxs` directly. No string
registry, no declared field list, no new type: **the type is the contract.** A controller needing
curvature simply requires `reference.ks`.

Today's controllers repack the raceline into private `Nx4`/`Nx5`/`Nx7` arrays with incompatible
column meanings; consuming the named fields directly is what removes that class of bug.

The reference is a **hot input** (§7). This replaces the current `plan(state, waypoints=...)`
sticky-mutation wart, where a per-call kwarg silently and *permanently* overwrites the instance
attribute.

---

## 9. Rendering

**Rendering moves out of the library.** Controllers already expose everything worth drawing as
attributes (`waypoints`, `ref_traj`, `x_pred`, `u_pred`); users extract what they want and examples
demonstrate a render callback. No `render_*` methods on controllers, no renderer abstraction.

Benefits beyond simplicity: it deletes ~3 near-identical copy-pasted methods per controller, and it
**removes the library's dependency on the gym renderer API entirely** — which is one of the real
drift points (`render_closed_lines`/`setData` vs `get_*_renderer`/`update`, CODEBASE.md §9.2). The
ABC's `waypoints_color` property and `waypoint_render` handle become moot and can go with it.

---

## 10. Testing

There are no first-party tests today; that is the root cause of the bug count, and this design should
not land without them.

1. **Cross-backend agreement** (§5.5) — per model providing >1 backend.
2. **Lap regression harness** — the existing headless harness: all 8 reference controllers
   constructed and driven on Spielberg; assert the 3 classical controllers complete a lap and no
   controller raises or emits non-finite actions. This is what verified the correctness pass with
   zero regressions.
3. **Config round-trip** — `update()` with a tuning change takes effect *without* a rebuild and
   *does* change behavior (this is the test that would have caught the static-`self` staleness);
   a structural change triggers exactly one rebuild.
4. **Trace counting** (§7.4) — the sharper version of (3), and it belongs in Phase 0 because it
   catches *both* failure modes that (3) alone can miss. Increment a counter inside the jitted kernel
   (it runs only at trace time), then assert:
   - sweeping a **tuning** value (`temperature`, `Q`, `u_min`) → output changes **and** trace count
     stays at 1 → catches silent staleness *and* accidental recompile churn;
   - changing a **structural** value (`N`, `n_samples`) → exactly **one** additional trace;
   - constructing a second structurally-identical solver → **no** new trace (cache is shared).

   These three assertions fail against today's code, which is exactly why they go in first.
5. **Capability errors** — composing a solver with a model lacking the required backend raises at
   construction, not mid-solve.

---

## 11. Build order

No backwards-compatibility constraint, so phases are ordered by **build dependency** — each lands in
final form, updates the examples in the same commit, and is verified against the lap harness (§10)
before the next starts.

| Phase | Content | Status |
|---|---|---|
| **0** | Test scaffolding: lap-regression harness, cross-backend agreement, trace-counting | ✅ `7aea58f` |
| **A** | State/control specs (§5.2), capability queries (§5.1); `pre_processing_fn` retired | ✅ `b9eb579` |
| **B** | Rendering removed from the library; examples own their render callbacks | ✅ `2fe9306` |
| **C** | `Controller`/`Planner` ABCs final: `compute_control()`, `update()`, lifecycle hooks | ✅ `05be620` |
| **D** | Explicit jit args with `static_argnames`; static-`self` eliminated | ✅ `32b9b57` |
| **E** | `APMPPISolver` inherits `MPPISolver` | ✅ `32b9b57` |
| **F** | AP-MPPI: constraint no-op fixed (clip vs constrain), `u_std` made effective. **SIT-LMPC safe set + value function still outstanding** | ◐ `32b9b57`, `cc89174` |
| **G** | lattice/lane-switcher ported to `Planner`; FGM/wall-follow are honest `Controller` stubs (**algorithms still to write**) | ◐ |

**Phase 0 first is the point.** Zero first-party tests is the root cause of the bug count; landing
the harness before the refactor is what makes every later phase verifiable instead of hopeful.

**Phase B is now cheap.** Without compatibility staging, rendering removal is a straight deletion of
~3 copy-pasted methods per controller plus the ABC's `waypoints_color`/`waypoint_render` — no alias
period, no dual path.

SIT-LMPC (**F**) depends on **D** because a retrained value function is silently ignored until the
static-`self` trace is fixed (§7.4) — that dependency is the real reason to do **D** before resuming
research work.

---

## 12. Decisions log — considered and rejected

| Rejected | Why |
|---|---|
| `Policy`/`Agent` top-level abstraction | RL/AV-stack vocabulary; wrong for a control + planning library |
| Splitting `Controller` into reactive / tracking / learning subtypes | A false distinction — PP and Stanley *do* take a reference; only FGM/wall-follow don't. LMPC's learning is internal state, not a type |
| A new `Reference`/`Trajectory` type | Duplicates the gym's `Raceline`, which already has named fields |
| Forced auto-derived Jacobians | Hand-written can be faster/more stable; make it optional instead |
| `xp`-parameterized single-source dynamics | **Measured:** zero runtime cost (3611 vs 3626 ns numpy; 43.2 vs 41.8 µs jax), *but* it forces `np.where` semantics on numpy — **18× slower branching** (1818 vs 101 ns) — and still can't cover CasADi. Net complexity is a wash for 2-of-3 backends. The agreement test is the simpler, more complete remedy |
| Globally switching the numpy path to `jax.numpy` | **Measured:** cvxpy hard-rejects jax arrays (`TypeError: ArrayImpl is not a valid type for a Constant value`) → breaks the LTV solver outright; and 9.3× slower per single call (35 µs vs 3.8 µs), turning LTV's linearization into 528 µs/tick. jax wins only batched (92× on 1024 samples) — so jax for MPPI, numpy elsewhere |
| dataclass/dict as the runtime state container | 12× slower batched; incompatible with `jax.vmap`/`jit` without pytree registration |
| Explicit hot/cold parameter API | Pushes an implementation detail onto users; the shape-vs-value rule derives it automatically |
| Pytree-registering the solver/config to fix static-`self` (§7.4) | **Measured:** behaviourally correct and ~equal speed (59.7 vs 62.5 µs; the ±6% spread is noise), and it reads ergonomically (`solver.u_min` stays traced). Rejected anyway: it adds a concept to learn, and a tuning field misfiled into `aux_data` is *silently* stale. Explicit arguments make the static/traced split readable in the signature, and a forgotten one is a loud `TypeError`. Kept as a documented fallback if signatures grow unwieldy |
| Custom `__hash__`/`__eq__` on structural fields to fix static-`self` | Also measured correct (66.2 µs), but requires permanent discipline: the day anyone reads a tuning value off `self` inside a kernel, staleness silently returns. No structural guard against it |
| `static_argnums` (positional) for the chosen approach | Index lists silently desync from long signatures — the mistake was made while drafting §7.4. `static_argnames` is reorder-safe and self-documenting |
| A `ControllerRenderer` / `get_plan_state()` interface | Unnecessary — controllers already expose their solutions; users render what they want |

---

## 13. Open questions

1. **`update()` kwarg sugar** — should `update(Q=...)` be allowed alongside `update(config=...)`?
   Convenient, but reintroduces string names and silent typos unless validated against
   `dataclasses.fields()`.
2. **`Planner` return type** — a full `Raceline` is heavy for a short local path (it spline-fits and
   builds an occupancy grid); a lane-switcher emits "which lane". Resolve when porting a real planner.
3. **Structural-field heuristic** — is "ints are structural, floats/arrays are tuning" right in every
   case, or does it need a per-config override list from the start? (`dt` is a float that changes
   discretization but not shape — hot by this rule, which seems correct.)
4. **`control_mode` enforcement** — validate against the env's `control_input` at construction, or
   leave it advisory?

---

## 14. Consequences for the pending cleanup

Two items previously queued for deletion in [DEADCODE_FINDINGS.md](DEADCODE_FINDINGS.md) are
**load-bearing for this design and must be kept**:

- **`LTVMPCSolver`'s unused cvxpy `Q`/`R`/`Rd`/`P` Parameters** — these are the parameter slots §7.3
  needs; wire them up rather than delete them.
- **`NonlinearMPCSolver.U0`** — an unfinished warm-start hook, not dead code (it *is* read to seed
  IPOPT).

Everything else on the safe-to-delete list is unaffected by this design.
