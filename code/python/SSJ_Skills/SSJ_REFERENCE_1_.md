# Sequence-Space Jacobian (SSJ) — Expert Reference

Reference for the `sequence-jacobian` package (v1.0.0, PyPI: `sequence-jacobian`, import
`sequence_jacobian as sj`). Derived from reading the source, not the docs. Written so an
assistant can reason about SSJ code correctly without re-deriving the mechanics each time.

Companion paper: Auclert, Bardóczy, Rognlie, Straub (2021), "Using the Sequence-Space Jacobian
to Solve and Estimate Heterogeneous-Agent Models," *Econometrica* 89(5). The code implements the
**fake-news algorithm** of that paper. Section/Proposition references below point to it.

Hard dependency: `numba` is imported at package top level (`estimation.py`) but is **not** pulled in
by the wheel — install it explicitly or the import fails.

---

## 1. Mental model

A model is a **DAG of blocks**. Each block is a map `inputs (dict of time paths) -> outputs
(dict of time paths)`. Three primitive block types plus two composites:

| Type | Decorator / ctor | Role |
|---|---|---|
| `SimpleBlock` | `@simple` | Aggregate equations (firms, market clearing, Taylor rule). Scalars or time paths. |
| `HetBlock` | `@het(...)` | Heterogeneous-agent problem: backward policy iteration + forward distribution. |
| `SolvedBlock` | `@solved(...)` or `.solved()` | A mini general-equilibrium model embedded as one block (solves internal unknowns/targets). |
| `CombinedBlock` | `combine([...])` / `create_model([...])` | Topologically sorts children, chains them. `create_model` just sets a "Model" repr alias. |
| `JacobianDictBlock` | (auto) | Wraps a raw `JacobianDict` as a block when one is passed into `combine`. |

Everything solves in **sequence space**: variables are length-`T` paths (`T≈300` default),
deviations from steady state. The central objects are `T×T` **Jacobians** `dO/dI` mapping input
paths to output paths. General equilibrium = find the path of unknowns `U` making target paths
`H(U,Z)=0`; linearised, `dU = -H_U^{-1} H_Z dZ`.

Two-layer method dispatch on every `Block`: the public method (`steady_state`, `jacobian`, …)
handles **variable remapping** via a `Bijection` `self.M`, calling the private `_method`:
`self.M @ self._method(self.M.inv @ args)`. `remap({old: new})` is how one block instance is
reused under renamed variables (see §12, heterogeneity).

---

## 2. `@simple` blocks

```python
@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y
```

- Inputs = function signature (via `inspect.signature`). Outputs = **parsed from source by regex**
  on the *last* `return` line: `re.findall('return (.*?)\n', source)[-1].replace(' ','').split(',')`.
- **`X(-1)` / `X(+1)` = time displacement** (lag/lead). Implemented by `Displace` objects at
  impulse time and by shifted diagonals in the Jacobian.
- **`X.ss` = the steady-state level of `X`**, usable inside `@simple` / `@solved` bodies. It is a
  *constant*: it adds no entry to `block.inputs` and carries no Jacobian (verified). Two standard
  uses: deviation-form rules (`r_prem = -psi_nfa * (nfa_u(-1) - nfa_u.ss)`, `D_B - D_B.ss`) and
  hand-linearisation, where replacing a live aggregate `X_t` by `X.ss` in a product with a
  first-order term is exact to first order and removes a DAG edge (Sec. 13).
  Caveat: `.ss` tracks the steady state actually solved. Patching a calibration *without*
  re-solving leaves `.ss` at the old value -- a silent source of "my counterfactual did nothing".
- SS: just calls `f` on scalars (`ignore` wrapper strips displacement, `numeric_primitive`
  postprocess).
- Jacobian: **exact/analytic**, not numerical. Each input is seeded with an `AccumulatedDerivative`
  (a dual-number-like object); running `f` accumulates the derivative structure, which becomes a
  `SimpleSparse` operator. `X(-1)` -> basis element on sub/super-diagonal.
- `SimpleSparse` stores a linear combination of shift operators `(i, m)`: `i` = diagonal (# above
  main), `m` = # of leading entries missing. `(0,0)=I`, `(1,0)`=left-shift (a lead), `(-1,0)`=right
  -shift (a lag). Closed under `@` via `multiply_basis` (Prop. 2 of the paper, sign-flipped).
  `IdentityMatrix` is an even cheaper `I` used to seed `dX/dX`.

### The `return`-line rules (both are real source-parsing constraints)
1. **Single flat line.** The regex captures up to the first `\n`, so a multi-line parenthesised
   tuple `return (\n a,\n b)` truncates the output list. Always `return a, b, c` on one line.
2. **Names are read verbatim from text.** Output names come from the source string, not runtime
   values. `return c, a` names outputs `c, a`; the values must be assigned to those exact names.

---

## 3. `@het` blocks (heterogeneous agents)

```python
def hh_init(a_grid, y, r, eis):                       # backward_init: seed for Va
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    return (1 + r) * (0.1 * coh) ** (-1 / eis)

@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):                # ONE backward step (EGM here)
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])                          # borrowing constraint
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)                     # envelope: next Va
    return Va, a, c
```

`@het` arguments:
- `exogenous` — name(s) of Markov transition matrix input(s) (e.g. `'Pi'`). Drives the exogenous
  state(s). Up to several allowed; each is a `Markov` acting on its own state axis.
- `policy` — endogenous continuous state(s) chosen by the agent, e.g. `'a'` (max **two**: 1D or 2D
  lottery). Must be a function output.
- `backward` — the backward variable(s) carried across the Euler step, e.g. `'Va'`. For each `X` in
  `backward`, the function must take `X_p` (next period) as input and return `X` as output.
- `backward_init` — function returning a seed for the backward variable(s) at SS.
- `hetinputs`, `hetoutputs` — usually attached later via `.add_hetinputs`/`.add_hetoutputs`.

### Naming conventions enforced by `static_checks`
- Policies and non-backward outputs must be **lowercase** at the individual level.
- SSJ **aggregates every non-backward output over the distribution and returns it CAPITALIZED**:
  `a -> A`, `c -> C`. Aggregate = `np.vdot(D, x)`. This capitalization is automatic (`M_outputs`
  bijection). So a hetoutput `t_E` becomes aggregate `T_E`, available to other blocks with no extra
  wiring — **check the hetblock's outputs before assuming an aggregate is "missing."**
- `D`, `Dbeg` are reserved (the distribution). A backward var cannot be named `D`/`Dbeg`.

### hetinputs / hetoutputs
- **hetinput**: a plain function run *before* the backward step to build individual-level inputs on
  the grid (e.g. `income(w, e_grid) -> y`, `make_grids(...) -> e_grid, Pi, a_grid`). Its outputs
  become internal grid arrays; its inputs become block inputs; its outputs are removed from the
  block's input list. Attach with `hh.add_hetinputs([income, make_grids])`.
- **hetoutput**: a function run *after* the backward step on individual arrays (e.g. a tax paid,
  MPC). Its outputs are aggregated+capitalized like ordinary outputs and exposed as internals.
- Both are `CombinedExtendedFunction`s (their own mini-DAG); order is auto-sorted.

### Steady-state internals of a HetBlock (`_steady_state`)
1. `backward_steady_state`: iterate the Euler/backward step to convergence (`backward_tol=1e-8`,
   `maxit=5000`; convergence checked on `policy` every 10 iters). Each iter takes expectations of
   the backward var over the exogenous Markov (`X_p = exog.expectation(X)`).
2. `forward_steady_state`: iterate the distribution to its stationary point. **Two-step law of
   motion per period**: first exogenous (`Markov.forward`), then endogenous (policy lottery). Seeds
   from outer product of exogenous stationary dists × uniform on endogenous grids unless a
   `Dbeg_seed` / `<state>_seed` is provided (`forward_tol=1e-10`, `maxit=100000`).
   Returns `Dbeg` (start of period, before exog shock) and `D` (after exog, the one used for
   aggregation).
3. Aggregate outputs, capitalize, stash grids/policies/`D`/`Dbeg` under `ss.internals[block.name]`.

`Dbeg` vs `D`: `Dbeg --exog(Markov)--> D --endog(lottery)--> Dbeg'`. Aggregation of outcomes uses
`D` (post-exogenous, the distribution over which this period's policies are defined).

### Policy lottery (endogenous forward step)
`lottery_1d(a, a_grid)` interpolates each agent's chosen `a` onto the grid, splitting mass between
the two bracketing nodes (`i`, `i+1`) with weights (`pi`, `1-pi`). `forward` pushes `D` forward;
`expectation` pulls values back (used for `curlyE`). A shock to policy `da` maps to a shock in
lottery weights `pi_shock = -da/space` where `space = grid[i+1]-grid[i]` — this is how a
differential change in savings perturbs next period's distribution (Prop. in paper, "forward
shock").

---

## 4. The fake-news algorithm (HetBlock `_jacobian`)

Computes `dO/dI` for a HetBlock analytically in 4 steps (paper §3–4). For each input `i` and
output `o`:

**Step 1 — backward** (`backward_fakenews`): a *fake-news* shock is a shock to input `i` that hits
only at horizon `s` and is anticipated. Propagate one unit shock backward `T` periods to get
`curlyY[i][o][s]` (impact on aggregate outcome today) and `curlyD[i][s]` (impact on the
distribution one step ahead). Uses `differentiable(...)` finite-difference versions of the backward
function / hetinputs / hetoutputs (`h=1e-4`, one- or two-sided).

**Step 2 — expectation vectors** (`expectation_vectors`): `curlyE[o][t]` = expectation of
steady-state outcome `o` propagated `t` periods under the law of motion, demeaned (demeaning is
numerically stabilising, theoretically neutral). Captures how a distributional perturbation today
maps into aggregate `o` in the future.

**Step 3 — fake-news matrix** `F` (`build_F`): `F[0,:] = curlyY`; `F[1:,:] = curlyE @ curlyD.T`.
Entry `F[t,s]` = response of `o` at `t` to a fake-news shock to `i` learned-about at `s`.

**Step 4 — Jacobian** `J` (`J_from_F`): recursively cumulate `F` along diagonals:
`J[1:,t] += J[:-1,t-1]`. This converts the fake-news matrix into the true Jacobian (the object
`G` if it were GE; here it is the partial `dO/dI`).

Cost is `O(T)` backward iterations total (not `O(T^2)`), the key efficiency result. The resulting
`J[O.upper()][i]` is a dense `T×T` array wrapped in a `JacobianDict`.

---

## 5. DAG composition (`CombinedBlock`)

- On construction, children are **topologically sorted** (Kahn's algorithm, `utilities/graph.py`).
  A **cycle raises** `Topological sort failed: cyclic dependency ...` naming the blocks.
  `outmap` also enforces **each variable is output by at most one block** (`'{o}' is output twice`).
- `inputs` = names consumed but never produced inside the DAG; `outputs` = all produced names.
- `_steady_state`: run children in sorted order, threading a growing `ss` dict.
- `_jacobian` (partial equilibrium composition): forward-accumulate along the sorted DAG via the
  chain rule:
  ```
  total_J = Identity(inputs)
  for block in sorted_blocks:
      J_block = block.jacobian(...)          # curlyJ of this block
      total_J.update(J_block @ total_J)      # compose: dO/dInputs
  ```
  `JacobianDict.__matmul__` composes (`compose`) or applies (`apply`) or remaps (Bijection).
- `partial_jacobians` caches each leaf block's `JacobianDict` (the "curlyJ"s) so they're computed
  once and reused across `H_U`, `H_Z`, and final `G`. Pass `Js=...` to reuse across calls.

---

## 6. Steady state (`solve_steady_state`)

```python
ss = model.solve_steady_state(calibration, unknowns, targets, solver='hybr')
```
- `calibration`: dict of fixed params + initial values.
- `unknowns`: `{name: guess}` or `{name: (lo, hi)}` (bounds for bracketing solvers).
- `targets`: `{residual_name: value}` (usually `0.`) or `{lhs: rhs}`. `compute_target_values`
  forms `lhs - rhs`.
- Internally: a residual closure updates `ss` with candidate unknowns, re-runs `self.steady_state`,
  returns target residuals; a root-finder drives them to zero.

Solvers (`solver=`):
- Multivariate scipy: `'hybr'` (default-ish, robust), `'lm'`, `'broyden1/2'`, `'anderson'`, …
- Univariate scipy: `'brentq'`, `'brenth'`, `'ridder'`, `'toms748'`, `'newton'`, … (need bounds).
- Custom: `'broyden_custom'`, `'newton_custom'` (with backtracking line search, in
  `utilities/solvers.py`), `'broyden'`. `'solved'` = assume already solved (used by `dissolve`).
- `provide_solver_default`: picks `'brentq'` for a **single** bounded unknown. For
  `len(unknowns) > 1` it **raises** `ValueError: Unable to find a compatible multi-dimensional
  solver` unless every value is a plain scalar -- bounds plus several unknowns therefore requires
  passing `solver="broyden_custom"` explicitly (verified). No need to patch `site-packages`.
- Constrained solving via `constrained_method='linear_continuation'` keeps iterates within bounds
  by **adding a penalty** to the residual outside them (`residual_with_linear_continuation`).

### Unknown value formats
| form | meaning |
|---|---|
| `x: 1.0` | scalar initial value |
| `x: (lo, hi)` | discouraged: warns, then averages to `(lo+hi)/2` as an initial value |
| `x: (lo, init, hi)` | lower bound, initial value, upper bound; asserts `lo < init < hi` |

### `solve_steady_state` does NOT validate its own result (verified)
`Block.solve_steady_state` calls `solve_for_unknowns(...)` then `return ss`. There is **no** check
on the targets. `run_consistency_check()` is defined in `blocks/support/steady_state.py` but is
**called from nowhere in the package**, and the `ctol=1e-9` entry of
`solve_steady_state_options` is dead code.

Combined with the penalty continuation above this gives a genuine silent-failure mode: the solver
parks an unknown on a bracket boundary, converges on the *penalised* objective, and returns a
normal-looking `SteadyStateDict` whose true residuals are O(1). Reproduced in a Krusell-Smith
test -- `solve_steady_state` returned with no error and no warning while `asset_mkt = -5.0` (both
unknowns pinned at their lower bounds). The *same* model with a tighter bracket instead raised
`ValueError: No convergence after 100 iterations`. The two failure modes are inconsistent, so an
exception cannot be relied on.

**Always assert the residuals yourself, after every ss solve and every impulse solve**, including
the residuals *not* used as targets (Walras / balance-of-payments identities) -- those are the
ones that silently absorb specification errors.
```python
def test_targets(d, names, tol=1e-8):
    for k in names:                       # works on SteadyStateDict and ImpulseDict alike
        assert np.max(np.abs(d[k])) < tol, f"{k}: {np.max(np.abs(d[k])):.3e}"
```

**SS-vs-dynamic block trick**: a common idiom (see KS example) is two DAGs sharing the HA block —
one with a `firm_ss` block that *inverts* the calibration targets (solve `Z, K` from `Y, r`), one
with the structural `firm` block for dynamics. Cleaner than solving everything numerically.

---

## 7. General-equilibrium Jacobians (`solve_jacobian`)

```python
G = model.solve_jacobian(ss, unknowns=['K'], targets=['asset_mkt'],
                         inputs=['Z'], T=300)
dY = G['Y']['Z'] @ dZ         # apply GE Jacobian to a shock path
```
Mechanics:
1. `partial_jacobians` → curlyJ of every block.
2. `H_U = jacobian(unknowns → targets)` and `H_Z = jacobian(inputs → targets)`.
3. `U_Z = -H_U^{-1} H_Z` (packed to dense `(len·T)²`, `np.linalg.solve`; `.pack(T)`/`.unpack`).
4. Compose `U_Z` back through the DAG to get `G = dOutputs/dInputs` in GE: builds a combined block
   `[U_Z, self]` and takes its Jacobian.

`G[out][shock]` is a `T×T` matrix. IRFs = `G[out][shock] @ dshock_path`.
`FactoredJacobianDict` pre-factorises `H_U` (LU) so repeated solves reuse the factorisation
(`H_U_factored` kwarg; also what SolvedBlocks cache).

---

## 8. Impulses

- **Linear** (`solve_impulse_linear`): `dU = -H_U^{-1} dH`, then push `dU | inputs` through the DAG
  (`impulse_linear` = apply Jacobians). Returns an `ImpulseDict`.
- **Nonlinear** (`solve_impulse_nonlinear`): **Newton on the full nonlinear DAG**. Iterates
  `U += H_U_factored.apply(residuals)` (quasi-Newton with a *fixed* factorised `H_U` from SS)
  until targets `< tol=1e-8` (`maxit=30`). Each iter calls `impulse_nonlinear` which, for a
  HetBlock, runs true backward+forward nonlinear passes (`backward_nonlinear`, `forward_nonlinear`).
- `ss_initial` kwarg: start the economy from a distribution different from the terminal SS (MIT
  shock with mismatched initial `Dbeg`).

`ImpulseDict` supports arithmetic (`+ - * /`), `.pack()/.unpack()`, and `.get(k)` (returns a zero
path if absent — handy for targets).

---

## 9. `@solved` blocks (embedded mini-models & the lag idiom)

```python
@sj.solved(unknowns={'i': 0.0}, targets=['i_resid'], solver='broyden_custom')
def monetary(i, pi, rstar, phi, rho_i):
    i_resid = rho_i * i(-1) + (1 - rho_i) * (rstar + phi * pi) - i
    return i_resid
```
- A `SolvedBlock` wraps a block + its own `unknowns`/`targets`. Its `.jacobian` returns the *GE*
  `G` of the mini-model as its curlyJ; its `.impulse_nonlinear` solves the internal transition.
- **Why `@solved` and not `@simple` for lagged self-reference:** an equation where a variable
  appears on **both sides** (e.g. interest-rate smoothing `i = ρ·i(-1) + …`) can't be written as a
  direct assignment `i = …` inside `@simple` — referencing `i(-1)` on the RHS of an assignment to
  `i` triggers a Python `UnboundLocalError`. Instead write it as a **residual** (`i_resid = … - i`,
  `return i_resid`) and let the SolvedBlock find the `i` path that zeroes it. The lagged term is
  fine on the RHS of a residual because `i` is a block *input* there, not a local being assigned.
- `dissolve=[name]` at SS time turns a SolvedBlock's unknowns into ordinary calibration (solver
  `'solved'`), i.e. "don't solve this internally, it's pinned."

---

## 10. Transitions / distribution operators (`het_support.py`)

- `Markov(Pi, i)`: exogenous transition on state axis `i`. `.forward(D)` = `Pi.T · D`;
  `.expectation(X)` = `Pi · X` (both along axis `i`). `.stationary()` = power iteration.
- `PolicyLottery1D/2D`: endogenous transition from interpolating policy onto grid.
- `CombinedTransition([exog, endog])`: chains stages; `.forward` applies in order, `.expectation`
  in reverse. This is the per-period law of motion.
- Shockable variants (`ForwardShockable*`, `ExpectationShockable*`) supply the derivatives the
  fake-news algorithm needs (effect of a policy/transition shock on next-period `D`, and of an exog
  shock on expectations).

---

## 11. Core data classes

- `SteadyStateDict`: two-level dict — `.toplevel` (aggregates/scalars) and
  `.internals[block_name]` (grids, `D`, `Dbeg`, policies, individual arrays). `_vector_valued()`
  flags multidim entries (excluded from default Jacobian outputs). Index with a list of names to
  subset.
- `JacobianDict`: nested `J[output][input] -> (T×T array | SimpleSparse | IdentityMatrix)`.
  `@` = compose / apply / remap. `.pack(T)`/`.unpack(...)` ↔ dense block matrix.
  `.identity(ks)`, `.addinputs()`. `[outs, ins]` slicing.
- `ImpulseDict`: dict of length-`T` paths + arithmetic; `.T` is the horizon.
- `FactoredJacobianDict`: LU-factored `H_U` for fast repeated `-H_U^{-1} @ x`.

---

## 12. Permanent-heterogeneity via `remap` (β-heterogeneity idiom)

To give agents different discount factors (or any parameter), **reuse one HetBlock under renamed
I/O** and aggregate:

```python
hh = household.add_hetinputs([income, make_grids])
to_map = ['beta', *hh.outputs]                                  # rename param + all outputs
hh_patient   = hh.remap({k: k + '_patient'   for k in to_map}).rename('hh_patient')
hh_impatient = hh.remap({k: k + '_impatient' for k in to_map}).rename('hh_impatient')

@simple
def aggregate(A_patient, A_impatient, C_patient, C_impatient, mass_patient):
    C = mass_patient * C_patient + (1 - mass_patient) * C_impatient
    A = mass_patient * A_patient + (1 - mass_patient) * A_impatient
    return C, A

model = create_model([hh_patient, hh_impatient, firm, mkt_clearing, aggregate])
```
`remap` deep-copies the block and composes `self.M` with the renaming bijection, so `beta_patient`
and `beta_impatient` are independent inputs and `A_patient`/`A_impatient` independent outputs, all
sharing the same solved policy machinery. Alternative to remap: physically add a discount-factor
axis to the state (`(n_e, n_beta, n_a)`) with an identity transition — but then **every array must
physically broadcast that axis** (numba `reshape` in the interpolation/lottery step reads flat
memory, so a merely size-1 / logically-broadcast axis causes out-of-bounds reads; force
materialisation, e.g. `+ 0.0 * beta_grid[None,None,:,None]`).

---

## 13. Breaking DAG cycles with steady-state constants

Using a household *aggregate* output as an input to an *upstream* block creates a cycle
(household depends on upstream, upstream depends on household → topological sort fails). Standard
fix: compute the needed quantity **once from the solved steady state as a plain calibration
constant** and feed that constant in, rather than the live aggregate. Example: Laspeyres/expenditure
weights in a headline-inflation aggregator are computed from SS nominal shares and passed as fixed
`omega` constants, so the aggregator doesn't depend on the current-period household block.

---

## 14. Grids & discretization (`sj.grids`, `utilities/discretize.py`)

- `grids.markov_rouwenhorst(rho, sigma, N)` → `(e_grid, pi_stationary, Pi)`; `e_grid` is
  **mean-normalised to 1** under the stationary distribution. Preferred for persistent AR(1)s.
- `grids.markov_tauchen(rho, sigma, N, ...)` → Tauchen alternative.
- `grids.asset_grid(amin, amax, n)` / `grids.agrid(amax, n)` — double-exponentially spaced grid,
  dense near the borrowing constraint. (`sj.agrid` at top level is deprecated; use `sj.grids.*`.)

---

## 15. Gotchas checklist (fast triage)

- **`return` must be one flat line**; names parsed verbatim from source (§2). A multi-line tuple
  silently loses outputs.
- **HetBlock outputs are auto-capitalized aggregates** (`a→A`). Don't re-create an aggregate that
  already exists; grep the hetblock outputs/hetoutputs first (§3).
- **Each variable output by exactly one block**, or `combine` raises "output twice".
- **Cycles fail loudly** at `create_model`; break them with SS constants (§13).
- **Lagged self-reference** (`i = …i(-1)…`) → use `@solved` with a residual, never `@simple` (§9).
- **Extra state axes must be physically broadcast** for numba reshape in `upperenv`/lottery (§12).
- **`resolve_ss` / re-solving**: params that only affect *dynamics* (e.g. `rho_i`) can be patched
  into `ss.copy()`; params that move the *steady state* (e.g. a markup) require a full re-solve of
  `solve_steady_state` before recomputing Jacobians.
- **Walras' law is a check, not a target.** If asset market clears to ~1e-12 but goods market
  doesn't, the household aggregate budget is inconsistent — suspect a double-counted price
  (e.g. income already includes `w`, don't multiply by `w` again) or a missing income component.
- **`np.vdot(D, x)` is the aggregation** everywhere; individual arrays and `D` must share shape.
- **`Js=` reuse**: pass cached partial Jacobians across `solve_jacobian`/impulse calls to avoid
  recomputing the expensive HetBlock fake-news step.

---

## 16. Minimal working template (Krusell–Smith, verified end-to-end)

```python
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import het, simple, create_model

def hh_init(a_grid, y, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)   # must be named Va (return-line rule, §2)
    return Va

@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def household(Va_p, a_grid, y, r, beta, eis):
    c_nextgrid = (beta * Va_p) ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c

def make_grids(rho, sigma, nS, amax, nA):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)
    a_grid = sj.grids.agrid(amax=amax, n=nA)
    return e_grid, Pi, a_grid

def income(w, e_grid):
    y = w * e_grid
    return y

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt

hh = household.add_hetinputs([make_grids, income])
model = create_model([hh, firm, mkt_clearing], name="KS")

calibration = dict(eis=1, delta=0.025, alpha=0.11, rho=0.966, sigma=0.5,
                   L=1.0, nS=7, nA=500, amax=200, Z=0.85, K=3.0)
ss = model.solve_steady_state(
    calibration,
    unknowns={'beta': 0.98, 'Z': 0.85, 'K': 3.0},
    targets={'r': 0.01, 'Y': 1.0, 'asset_mkt': 0.0},
    solver='hybr')

G = model.solve_jacobian(ss, unknowns=['K'], targets=['asset_mkt'],
                         inputs=['Z'], T=300)
dZ  = 0.01 * ss['Z'] * 0.8 ** np.arange(300)
dY  = G['Y']['Z'] @ dZ          # IRF of Y to the TFP path
```

---

## 17. StageBlock (multi-stage heterogeneous agents)

A `StageBlock` generalises `HetBlock`: instead of one monolithic backward step, the within-period
problem is split into an ordered list of **stages**, each with its own backward step, its own
reported outputs, and its own law of motion for the distribution. Use it when the period has
internal structure the fake-news algorithm must see stage-by-stage: **discrete choice** (occupation,
tenure, brown-vs-green durable adoption) interleaved with continuous choice, multiple sequential
sub-decisions, or several exogenous transitions applied at different points.

Import (not top-level): `from sequence_jacobian.blocks.stage_block import StageBlock` and
`from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D, Continuous2D, LogitChoice`.

### Construction
```python
hh = StageBlock([stage_0, stage_1, ...], backward_init=..., hetinputs=..., name=...)
```
- Stages are listed in **chronological (forward) order** — the order the distribution moves through
  them within a period. Backward iteration runs them in reverse automatically.
- Cyclic wiring: stage `i` receives, as its incoming backward variable, the `backward_outputs` of
  stage `i+1` (mod N). The last stage wraps to the first → next period. So the backward variable
  circulates: `... → stage_i → ... → stage_{N-1} → stage_0(next period) → ...`.
- `backward_init` seeds the first stage's `backward_outputs` (same return-line naming rule as §2).
- `add_hetinputs([...])` works as in HetBlock (grids, income, run before the stages).
- Aggregation & **capitalization** as in HetBlock: each stage's `report` outputs are aggregated over
  that stage's distribution and exposed CAPITALIZED (`a→A`, `c→C`). A hetoutput on a stage (e.g.
  `t_E`) becomes `T_E` — so a "missing" aggregate may already be produced by a stage.
- Protected names: a stage can't be named `D`/`law_of_motion`; can't report `d`/`law_of_motion`;
  reported outputs must be lowercase. Inputs, outputs, and backward variables must not overlap.

### Stage types (`blocks/support/stages.py`)
| Stage | Constructor | Role |
|---|---|---|
| `Continuous1D` | `Continuous1D(backward, policy, f, name=, hetoutputs=)` | One continuous endogenous choice (EGM). `f` returns the backward var + reported policies; law of motion = 1D policy lottery. |
| `Continuous2D` | `Continuous2D(backward, policy=(p1,p2), f, ...)` | Two continuous choices; 2D lottery. |
| `Exogenous` (via `ExogenousMaker`) | `ExogenousMaker(markov_name, index, name=, hetoutputs=)` | Applies a Markov matrix `markov_name` along state axis `index`. Reports nothing; its law of motion is the Markov. Built lazily via `.make_stage(next_backward)`. |
| `LogitChoice` | `LogitChoice(value, backward, index, taste_shock_scale, f=None, name=, hetoutputs=)` | **Discrete choice** with T1EV taste shocks. `f` = flow utility (single output). Produces choice-probability law of motion `DiscreteChoice`; updates value function `value` (`EV`) and takes expectations of `backward` vars. `index` = state axis the choice moves. |

`ExogenousMaker` is a *maker*, not a stage: `make_all_into_stages` finds the first real `Stage`,
then calls `.make_stage(next_stage.backward_outputs)` on the makers so each exogenous stage inherits
the right backward variables. You can therefore drop `ExogenousMaker(...)` straight into the list.

### Key difference from HetBlock: the Euler expectation is its own stage
In a `HetBlock`, the backward function takes `Va_p` and the block takes the exogenous expectation
internally. In a `StageBlock`, the **exogenous transition is an explicit `Exogenous` stage**, so the
continuous-choice function takes the *already-expectation-taken* continuation value `Va` (not
`Va_p`) and applies the discount itself (`beta * Va`). This is why stage functions reference `Va`,
not `Va_p`.

### Internals layout (nested per stage)
`ss.internals['hh']` holds the hetinput grids at top level plus one subdict **per stage**:
```python
ss.internals['hh']['consav']   # -> {'Va', 'a', 'c', 'law_of_motion', 'D'}
```
Each stage subdict has that stage's individual arrays, its `law_of_motion` (a `LawOfMotion`), and
`D` = the **beginning-of-that-stage** distribution. Aggregation of a stage's report uses that
stage's own `D`.

### Steady state & forward flow
- Backward: iterate full backward sweeps (all stages, reversed) until the first stage's backward
  outputs converge (`backward_tol=1e-9`, `maxit=5000`).
- Forward: `forward_step` applies each stage's law of motion in chronological order,
  `D_beg → lom_0 → lom_1 → ... → D_end`; `D_end` becomes next period's `D_beg`. The steady state
  stores the beginning-of-stage `D` for every stage.

### Fake-news algorithm, staged (`_jacobian`)
Same four steps as HetBlock (§4), adapted: step 1 walks **backward through stages** collecting the
shock to each stage's law of motion (`backward_step_shock` per stage) and the direct (policy)
contribution to `curlyY`; then walks **forward through stages** accumulating the distributional
perturbation `dD` (`dD = lom @ dD + dlom @ D`) and its contribution to `curlyY`. Steps 2–4
(`expectation_vectors`, `build_F`, `J_from_F`) are shared with `HetBlock` (literally
`HetBlock.build_F` / `HetBlock.J_from_F`). `LogitChoice.backward_step_shock` implements the discrete
-choice derivative: `dP = P·(dV − dEV)/scale`, the envelope `dEV = Σ P·dV`.

### Gotchas specific to StageBlock
- **Chronological order matters** and is not auto-detected: `[exog, consav]` ≠ `[consav, exog]`.
  Put stages in the order the distribution physically flows within the period.
- **Stage functions take the post-expectation backward var** (`Va`, not `Va_p`); apply `beta`
  inside the continuous-choice stage.
- **Per-stage `D`**: to aggregate or inspect a stage's outcome, use `ss.internals[name][stage]['D']`,
  not a single block-level distribution — there isn't one.
- `backward_init` must produce the **first stage's** backward outputs, named exactly (return-line
  rule).
- No `remap` support hardened yet for hetinputs on StageBlock (`TODO` in source) — prefer building
  separate StageBlocks for permanent heterogeneity rather than remapping.

### Verified minimal StageBlock (2-stage SIM — identical SS/IRF to monolithic HetBlock)
```python
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import simple, create_model, interpolate, misc
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D

def hh_init(a_grid, y, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)          # name Va (return-line rule)
    return Va

def consav(Va, a_grid, y, r, beta, eis):              # takes Va (post-expectation), not Va_p
    c_nextgrid = (beta * Va) ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c

hh = StageBlock(
    [ExogenousMaker(markov_name='Pi', index=0, name='exog'),   # chronological order:
     Continuous1D(backward='Va', policy='a', f=consav, name='consav')],  # exog THEN choice
    backward_init=hh_init, name='hh')

def make_grids(rho, sigma, nS, amax, nA):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho=rho, sigma=sigma, N=nS)
    a_grid = sj.grids.agrid(amax=amax, n=nA)
    return e_grid, Pi, a_grid

def income(w, e_grid):
    y = w * e_grid
    return y

hh = hh.add_hetinputs([make_grids, income])

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt

model = create_model([hh, firm, mkt_clearing], name="KS-staged")

calibration = dict(eis=1, delta=0.025, alpha=0.11, rho=0.966, sigma=0.5,
                   L=1.0, nS=7, nA=500, amax=200, Z=0.85, K=3.0)
ss = model.solve_steady_state(
    calibration, unknowns={'beta': 0.98, 'Z': 0.85, 'K': 3.0},
    targets={'r': 0.01, 'Y': 1.0, 'asset_mkt': 0.0}, solver='hybr')

G  = model.solve_jacobian(ss, unknowns=['K'], targets=['asset_mkt'], inputs=['Z'], T=300)
dZ = 0.01 * ss['Z'] * 0.8 ** np.arange(300)
dY = G['Y']['Z'] @ dZ
# ss.internals['hh']['consav'] -> {'Va','a','c','law_of_motion','D'}
```

---

## 18. API quick-reference

```
sj.simple                              -> @simple decorator
sj.het(exogenous, policy, backward,    -> @het decorator
       backward_init=, hetinputs=, hetoutputs=)
sj.solved(unknowns, targets, solver=)  -> @solved decorator (single SimpleBlock mini-model)
sj.combine([...], name=)               -> CombinedBlock
sj.create_model([...], name=)          -> CombinedBlock (Model alias)

StageBlock([stages...], backward_init=, hetinputs=, name=)   # multi-stage HetBlock (§17)
  from sequence_jacobian.blocks.stage_block import StageBlock
  from sequence_jacobian.blocks.support.stages import (
      ExogenousMaker, Continuous1D, Continuous2D, LogitChoice)
  ExogenousMaker(markov_name, index, name=)          # exogenous Markov stage
  Continuous1D(backward, policy, f, name=)           # 1 continuous choice (EGM)
  Continuous2D(backward, policy=(p1,p2), f, name=)   # 2 continuous choices
  LogitChoice(value, backward, index, taste_shock_scale, f=, name=)  # discrete choice (T1EV)

block.add_hetinputs([f, ...])          -> HetBlock/StageBlock w/ pre-step individual inputs
block.add_hetoutputs([f, ...])         -> HetBlock (or per-Stage) post-step outputs (aggregated+CAP)
block.remap({old: new})                -> renamed copy (heterogeneity)
block.rename(name) / .rename(suffix=)  -> renamed copy
block.solved(unknowns, targets)        -> wrap as SolvedBlock

block.steady_state(calibration)                                  -> SteadyStateDict (partial eq)
block.solve_steady_state(calib, unknowns, targets, solver=)      -> SteadyStateDict (general eq)
block.jacobian(ss, inputs, outputs, T=)                          -> JacobianDict (partial)
block.solve_jacobian(ss, unknowns, targets, inputs, T=300)       -> JacobianDict G (general eq)
block.impulse_linear(ss, inputs, outputs=)                       -> ImpulseDict (partial)
block.solve_impulse_linear(ss, unknowns, targets, inputs)        -> ImpulseDict (general eq)
block.solve_impulse_nonlinear(ss, unknowns, targets, inputs,
                              ss_initial=)                        -> ImpulseDict (nonlinear GE)
block.partial_jacobians(ss, inputs, outputs, T)                  -> {name: JacobianDict} cache

sj.grids.markov_rouwenhorst(rho, sigma, N) -> (e_grid, pi, Pi)   # e_grid mean 1
sj.grids.markov_tauchen(rho, sigma, N)     -> (e_grid, pi, Pi)
sj.grids.asset_grid(amin, amax, n) / agrid(amax, n)
sj.interpolate.interpolate_y(x, xq, y)     # EGM interpolation
sj.misc.setmin(a, floor)                   # in-place clip (borrowing constraint)

ss['X']                                    # aggregate
ss.internals['block']['D']                 # HetBlock distribution & grids
ss.internals['block'][stage]['D']          # StageBlock: per-stage distribution
G['out']['shock']                          # T×T GE Jacobian
G['out']['shock'] @ dshock                 # IRF path
```

---

## 19. Idioms from the Auclert–Rognlie–Straub energy replication code

Read off the *Managing an Energy Shock* replication notebook. These are structural patterns, not
API details; they are what makes a multi-variant quantitative model maintainable in SSJ.

### 19.1 `dissolve` — suppress a `@solved` block's inner solver at the steady state
```python
ss = model.solve_steady_state(calib, unknowns, targets,
                              dissolve=['unions', 'UIP', 'CA', 'piW_to_W', 'pitop'])
```
`dissolve` names `@solved` blocks whose internal unknowns should be read from the calibration
instead of being solved internally (solver `'solved'`). Use it when the inner unknown is pinned
by a normalisation at the ss (`piw = 0`, `P = 1`, `Q = 1`), when it duplicates an outer unknown,
or when the inner residual is identically zero at the ss for any value of the unknown (a unit
root — see Sec. 9). `dissolve` affects the **steady state only**; the block solves normally along
the transition. The dissolved variables must appear in the calibration dict with their ss values.

### 19.2 Nested models: a household *group* as a subgraph
`sj.create_model` output is an ordinary `CombinedBlock` and drops straight into a larger
`sj.combine`. ARS build the whole β-heterogeneous household group as its own model, then treat it
as one block:
```python
group_vars = ['C', 'A', 'MPC', 'c_ss_grid', 'C2', 'LOGC', 'DIFF_C', ...]
hh_list = [hh_ha.rename(suffix=f'_{i}')
                .remap({x: f'{x}_{i}' for x in group_vars})
                .remap({'beta_g': f'beta_{i}'})
           for i in range(3)]

@simple
def group_betas(beta_spread, beta_max):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread/2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2

@simple
def aggregate_groups(C_0, C_1, C_2, A_0, A_1, A_2, MPC_0, MPC_1, MPC_2):
    C = (C_0+C_1+C_2)/3; A = (A_0+A_1+A_2)/3; MPC = (MPC_0+MPC_1+MPC_2)/3
    return C, A, MPC

hh_ha = create_model(hh_list + [group_betas, aggregate_groups])
model = combine([hh_ha, hh_outputs, firm, mkt, unions, ...])
```
`rename(name=None, suffix=None)` (verified signature): the `suffix` form renames the block and its
`internals` key together, so per-type steady-state policies are read as
`ss.internals['hh_0']['c']`. Note that ARS run **three separate HetBlocks**, not one block with a
`beta` axis in the state — see Sec. 12 for why that is the safer of the two implementations.

At the ss, `beta_max` is the unknown and `beta_spread` is held fixed (0.06 quarterly with
`beta_max ≈ 0.984`). Solving `beta_spread` endogenously needs a second target, typically an
aggregate wealth-to-GDP ratio.

### 19.3 Model variants selected by a calibration constant
Branching on a *parameter* inside `@simple` is legal: the branch resolves once at evaluation and
the parameter never receives a Jacobian. This carries several model closures in one codebase.
```python
@simple
def eqm_cond(cE, prodE, PEstar, PEstar_shock, E_supply, E_supply_elasticity):
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock      # price-taking SOE: world price exogenous
    else:
        E_clearing = (cE + prodE) - E_supply    # quantity closure: price clears the market
    return E_clearing
```
Also used for `if prodE_share == 0:` (energy in production on/off) and `if eta_E == 1:`
(Cobb–Douglas vs CES: the `1/(1-eta)` CES aggregator is singular at unit elasticity).
**Never** branch on an endogenous *variable* — that is a kink, and the first-order machinery will
silently linearise around whichever branch the steady state happens to sit in.

### 19.4 Shock inversion: solve for the shock that delivers a target path
Anything expressible as a residual can be inverted by adding an unknown and a target to the
transition solve. Passing unknowns/targets as **sets** and combining with `|` keeps variants
readable.
```python
unknowns_td = {'y', 'pi', 'r', 'tauY', 'PEstar', 'w'}
targets_td  = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing', 'w_res'}

# "what supply path reproduces this world-price path?"  PEstar_diff = PEstar - PEstar_shock
irf = model.solve_impulse_linear(ss,
        unknowns_td | {'E_supply_shock'},
        targets_td  | {'PEstar_diff'},
        shocks)
E_shock = {'E_supply_shock': irf['E_supply_shock']}      # reuse as a shock elsewhere
```
Same construction gives a **flexible-price counterfactual**: add `ishock` as an unknown and `piw`
as a target — "choose the real-rate path that holds wage inflation at zero". And a **freeze**:
add the instrument as an unknown and `X - X.ss` as a target to hold `X` fixed along the
transition.

### 19.5 The `recalib` wrapper: what needs a re-solve
```python
def recalib(shocks, resolve_ss=False, **kwargs):
    ss_here = ss_baseline.copy()
    ss_here.update(kwargs)                 # ResultDict: .update(dict), NOT kwargs
    if resolve_ss:
        ss_here = model.solve_steady_state(ss_here, unknowns_ss, targets_ss,
                                           solver="broyden_custom", dissolve=dissolve)
    else:
        ss_here = model.steady_state(ss_here, dissolve=dissolve)
    test_targets(ss_here, RESID)                                   # Sec. 6
    irf = model.solve_impulse_linear(ss_here, unknowns_td, targets_td, shocks)
    test_targets(irf, RESID)
    return irf
```
- Parameters affecting **only dynamics** (`rho_i`, `phi_pi`, `theta_w`): patch `ss.copy()`,
  `resolve_ss=False`.
- Parameters affecting the **steady state** (`markup_ss`, `alpha_E`, any `delta`, any adjustment
  cost): `resolve_ss=True`, and `calibration`, `unknowns_ss`, `targets_ss` must all be passed
  explicitly.
- The trap: patching an ss-relevant parameter without re-solving returns a plausible IRF computed
  around the wrong point. `test_targets` catches it only if the parameter enters a residual —
  often it does not. Keep the two paths visibly separate rather than inferring which is needed.

### 19.6 Analytic non-regression assertions
ARS pin their model against closed-form limits and `assert` them in the notebook. This is the
cheapest protection against a silent respecification, and worth reproducing for any model with a
known analytic limit:
```python
analytic = calibration['alpha_E'] / (1 - calibration['alpha_E'] - calibration['alpha_F_tilde']) \
           * ra_list[0]['PEstar_shock']
assert np.allclose(ra_list[0]['y'], analytic, atol=1e-4)     # RA and HA coincide in this limit
assert np.allclose(ha_list[0]['y'], analytic, atol=1e-4)
assert np.allclose(ra_list[1]['y'], 0.5 * analytic, atol=1e-4)   # scales linearly in chi
```
Note what this particular assertion says economically: in the frictionless limit (flexible
prices, no importer frictions, real-rate rule) the output response to an energy price *increase*
is **positive** and proportional to the openness parameter. A negative output response in that
class of model comes from real wage rigidity, low substitution elasticity and importer frictions
— not from the open-economy structure per se. Worth remembering before attributing a sign
difference to a modelling choice.

### 19.7 Guarding Jacobian lookups
Variables with no exposure to a shock are **absent** from the `JacobianDict`, not zero-filled:
```python
def g(k, dshock, T):
    return G[k]['C_E_B_S'] @ dshock if (k in G.nesteddict and 'C_E_B_S' in G[k]) \
           else np.zeros(T)
```
Related: `solve_jacobian` returns the **unknowns** too, even though they are not in
`model.outputs`. When filtering requested outputs use `set(model.outputs) | set(unknowns)`.

---

### Verified in this environment
`sequence-jacobian 1.0.0`, `numba 0.66`, `numpy`. Sec. 2 (`.ss`), Sec. 6 (3-tuple unknowns, `provide_solver_default` raising on multi-unknown brackets, the absent target validation) and Sec. 19.2 (`rename(name=, suffix=)`) were each reproduced directly against the installed package. Both the monolithic HetBlock template (§16) and
the 2-stage StageBlock template (§17) solve to machine-precision market clearing
(`asset_mkt ~3e-15`, `goods_mkt ~3e-9`) and produce the identical canonical TFP IRF
(`dY[0]/Y = +1.00%`, `dC[0]/C = +0.37%`), confirming the StageBlock decomposition reproduces the
monolithic solution.
