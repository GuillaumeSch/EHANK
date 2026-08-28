---
name: sequence-jacobian
description: >
  Hard-won gotchas and patterns for building HANK / heterogeneous-agent models with the
  `sequence-jacobian` (SSJ) Python library. Use whenever writing or debugging SSJ code:
  @het / @simple / @solved blocks, StageBlock, hetinputs, steady_state /
  solve_steady_state, jacobian / solve_jacobian, remap, discount-factor heterogeneity,
  dissolve, model variants, shock inversion, or when hitting cryptic SSJ errors (empty
  SimpleSparse, topological cycles, bracket failures, silent non-convergence). This is a
  troubleshooting index: match the symptom, apply the fix.
---

# SSJ — gotchas & patterns

Each entry: **symptom → cause → fix (+ minimal snippet)**. Ordered by how much damage it does.

Entries marked **[verified]** were reproduced in this environment (`sequence-jacobian 1.0.0`,
numba 0.66). Entries marked **[ARS]** are idioms read off the Auclert–Rognlie–Straub *Managing
an Energy Shock* replication notebook.

---

## 0. `solve_steady_state` NEVER checks the steady state it returns **[verified]**

**This is the single most dangerous behaviour in the library.** Read `Block.solve_steady_state`:
it calls `solve_for_unknowns(...)` and then `return ss`. There is no validation step.
`run_consistency_check()` exists in `blocks/support/steady_state.py` but is **called from
nowhere**, and the `ctol=1e-9` entry in `solve_steady_state_options` is dead code.

Consequence: a failed solve returns a `SteadyStateDict` that looks completely normal.

**[verified]** In a Krusell–Smith test with unknowns `{'beta': (0.95, 0.98, 0.999), 'K': (5., 8., 15.)}`:

```
solve_steady_state returned normally, no error, no warning
  asset_mkt  = -5.0001e+00      <-- A = 0.0000, K = 5.0001
  goods      = -1.3393e-02
```

The solver walked both unknowns onto the lower bound of their brackets and stopped. Why it
looks like success: bounded unknowns are handled by `constrained_method="linear_continuation"`,
which *adds a penalty* to the residual outside the bounds (`residual_with_linear_continuation`).
Broyden converges on the **penalised** objective while the **true** targets are wide open.

Same model, a *tighter* bracket, raises loudly instead:
```
ValueError: No convergence after 100 iterations
```
So the two failure modes are inconsistent: sometimes it screams, sometimes it lies.

**Fix — always assert the targets yourself.** This is exactly why ARS wrote `test_targets`;
port it and call it after *every* `steady_state`, `solve_steady_state` and
`solve_impulse_linear`. Residual variables are ordinary block outputs, so the same function
works on an `ImpulseDict`.

```python
def test_targets(d, names, tol=1e-8, noisy=False):
    """Works on SteadyStateDict and ImpulseDict alike."""
    for k in names:
        v = np.max(np.abs(d[k]))
        assert v < tol, f"{k}: {v:.3e}"
        if noisy:
            print(f"  {k:20s} {v:.2e}")

RESID = ['GBC', 'goods_clearing', 'labor_mkt', 'wnkpc', 'brown_energy_mkt',
         'nfa_u_res', 'nfa_res']          # nfa_res holds by Walras: assert it too
test_targets(ss, RESID)
test_targets(irf, RESID)
```
Put the Walras-residual variables (the ones *not* used as targets) in the list too — those are
the ones that silently absorb specification errors.

---

## 1. Bounded unknowns: 3-tuples need an explicit `solver=` **[verified]**

`solve_steady_state` accepts three forms per unknown:

| form | meaning |
|---|---|
| `x: 1.0` | scalar initial value |
| `x: (lo, hi)` | **discouraged** — warns, and just averages to `(lo+hi)/2` as an init value |
| `x: (lo, init, hi)` | lower bound, initial value, upper bound (asserts `lo < init < hi`) |

- **Symptom:** `ValueError: Unable to find a compatible multi-dimensional solver with provided
  'unknowns'.`
- **Cause:** with `solver=""` (the default), `provide_solver_default` inspects the unknowns; for
  `len(unknowns) > 1` it requires every value to be a `Real` scalar and raises otherwise. So
  3-tuples + more than one unknown is rejected *before* the solve begins. **[verified]**
- **Fix:** pass the solver explicitly. Do **not** patch `site-packages` (the ARS notebook says
  "comment out lines 36–39 in steady_state.py" — unnecessary and non-portable):

```python
ss = model.solve_steady_state(
        calib,
        unknowns={'beta': (0.96, 0.985, 0.999), 'vphi': (0.1, 1.0, 2.0)},
        targets={'asset_mkt': 0., 'wnkpc': 0.},
        solver="broyden_custom")            # <-- required with 3-tuples, >1 unknown
```
With exactly **one** unknown, a 3-tuple is fine and defaults to `brentq`. **[verified]**

Note the trade-off: brackets buy you robustness against the solver wandering into a region
where the household block won't converge, at the cost of failure mode §0(b). Brackets +
`test_targets` is the only safe combination.

---

## 2. `.ss` — steady-state value of a variable, inside a block **[verified]**

Undocumented elsewhere in our notes and used constantly in ARS and in `blocks_soe.py`.
Inside `@simple` / `@solved`, `x.ss` is the steady-state level of `x`. It does **not** create a
new input (`f.inputs` is unchanged) and it is a constant, so it carries no Jacobian.

```python
@sj.simple
def debt_premium(nfa_u, psi_nfa):
    r_prem = -psi_nfa * (nfa_u(-1) - nfa_u.ss)     # deviation from own ss
    return r_prem
```
Two standard uses:
- **deviation-form rules** (`w - w.ss`, `pE_P - pE_P.ss`, `D_B - D_B.ss`);
- **first-order linearisation by hand** — replacing a live aggregate by its ss level, see §7.

Careful: `x.ss` is the ss of the *model you solved*. If you patch a calibration and re-solve,
`.ss` moves with it; if you patch `ss.copy()` without re-solving, it does not. This is the usual
source of "why did my counterfactual not change".

---

## 3. `dissolve=[...]` — don't run a `@solved` block's inner solver at the SS **[ARS]**

`dissolve` names `@solved` (or `@sj.combine`d) blocks whose internal unknowns should be treated
as **given calibration** at steady-state time instead of being solved internally.

```python
ss = model.solve_steady_state(calib, unknowns, targets,
                              dissolve=['unions', 'UIP', 'CA', 'piW_to_W', 'pitop'])
```
Use it when:
- the inner unknown is **also** an outer unknown or is pinned by a normalisation
  (e.g. `piw = 0`, `P = 1`, `Q = 1` at the ss) — solving it internally is redundant and often
  ill-conditioned (a unit root, see §6);
- the inner residual is identically zero at the ss for any value of the unknown.

The dissolved variable must be present in the calibration dict with its ss value. `dissolve`
affects the **steady state only** — the block is solved normally along the transition.

---

## 4. Model variants selected by a calibration constant **[ARS]**

Branching on a *parameter* inside `@simple` is legal and is the cleanest way to carry model
variants in one codebase: the branch is resolved once when the block is evaluated, and the
parameter never gets a Jacobian.

```python
@sj.simple
def eqm_cond(cE, prodE, PEstar, PEstar_shock, E_supply, E_supply_elasticity):
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock        # price-taking SOE: price exogenous
    else:
        E_clearing = (cE + prodE) - E_supply      # quantity closure: price clears
    return E_clearing
```
ARS use the same trick for `if prodE_share == 0:` (energy in production on/off) and
`if eta_E == 1:` (Cobb–Douglas vs CES — see §10, CES formulas are singular at unit elasticity).

**Do not** branch on an endogenous *variable*: that is a kink, not a variant, and the
first-order machinery will silently linearise around whichever branch the ss happens to be in.

---

## 5. Shock inversion: back out the shock path that delivers a target path **[ARS]**

To ask "what quantity path would generate *this* price path?", add an unknown and a target to
the transition solve. This is a general and underused idiom.

```python
unknowns_td = {'y', 'pi', 'r', 'tauY', 'PEstar', 'w'}
targets_td  = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing', 'w_res'}

# baseline: PEstar exogenous. Inverted: solve for the supply path E_supply_shock
# that reproduces the given PEstar path (PEstar_diff = PEstar - PEstar_shock).
irf = model.solve_impulse_linear(
        ss,
        unknowns_td | {'E_supply_shock'},        # sets: union to add one unknown
        targets_td  | {'PEstar_diff'},
        shocks)
E_shock = {'E_supply_shock': irf['E_supply_shock']}   # reuse as a shock elsewhere
```
Passing `unknowns`/`targets` as **sets** and combining with `|` makes variants readable and is
supported. Same idea gives you a flexible-price counterfactual: add `ishock` as an unknown and
`piw` as a target ("choose the real rate that keeps wage inflation at zero").

Our own `main_soe.py` already uses this for the no-adoption freeze
(`U + ['psi_g']`, `TG + ['D_B_target']`). It generalises: **anything you can write as a residual
can be inverted.**

---

## 6. `@solved` needs a determinate residual and, in 1-D, a sign-changing bracket

- **Symptom:** `f(a) and f(b) must have different signs`, or `Unable to find a compatible
  one-dimensional solver`.
- **Cause:** the block's ss residual is identically 0 for all values of the unknown (a unit
  root). Classic case: a nominal wage law of motion `W_res = W/W(-1) - 1 - pi_w`, zero for *any*
  `W` at the ss (nominal indeterminacy).
- **Fixes:** either `dissolve` the block at the ss (§3, this is what ARS do for `piW_to_W` and
  `pitop`), or keep it out of the model entirely and **post-process** it: cumulate `pi_w` to get
  `d log W`, then `d log P = d log W − d log w`.
- Multi-unknown `@solved` blocks need `solver="broyden_custom"`; `brentq` is 1-D only.

```python
@sj.solved(unknowns={'J': 15., 'j': 15.}, targets=['Jres', 'jres'], solver="broyden_custom")
def income(y, w, Z, J, j, rante, markup_ss, ...):
    dividend = (markup_ss - 1) * w * n
    Jres = dividend + J(1)/(1 + rante) - J        # forward asset-pricing recursion
    jres = J(1)/(1 + rante) - j
    return jres, Jres, atw_n, dividend, gdp, atw, n, btw_n
```

**Why `@solved` and not `@simple` for lagged self-reference:** an equation where a variable
appears on both sides (`i = rho*i(-1) + ...`) cannot be a direct assignment inside `@simple` —
Python raises `UnboundLocalError`. Write a residual (`i_resid = ... - i`) and let the
SolvedBlock find the path.

---

## 7. First-order trick to avoid a household → upstream cycle

If an upstream block needs a household aggregate `X_t` multiplied by a first-order object (a
price gap `= pEstar − 1`), the product's deviation is **second order**. Replace `X_t` by the
constant `X_ss`: exact to first order, and it deletes the household→upstream edge.

```python
subsidy_cost = tauE * price_gap * cE_ss        # cE_ss constant, NOT cE_t
```
Same family as §13 in the reference (SS constants to break cycles) and §2 (`.ss`).

---

## 8. `backward_init` / return-line parsing

SSJ parses the **source text** of the `return` line to name outputs. A `return <expression>`
(not a bare variable) breaks naming.

- **Symptom:** `KeyError: 'Va'` inside `backward_steady_state` at `exog.expectation(ss[k])`.
- **Fix:** assign to the named backward variable and return it by name, on ONE flat line.

```python
def hh_init(coh, r, eis):
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va                     # NOT: return (1+r)*(0.1*coh)**(-1/eis)
```
Same rule for every hetinput / hetoutput / backward function: `return a, b, c` on one line,
never a multi-line parenthesised tuple.

---

## 9. Empty `SimpleSparse` crash

- **Symptom:** `ValueError: not enough values to unpack (expected 2, got 0)` in
  `sparse_jacobians.py` (`zip(*self.elements.items())`), during `J @ total_Js`.
- **Cause:** a variable whose *composed* Jacobian is identically zero is consumed downstream.
  Two routes: (a) an instrument scaled by a parameter currently 0 (`tauE*price_gap`, `tauE=0`);
  (b) exact cancellation along an identity (a price index `p_c = P_hh/P ≡ 1` when there is no
  subsidy — partials cancel to an empty operator).
- **Fixes (pick one):**
  - make the degenerate variable a **calibration constant**, so SSJ attaches no Jacobian, and
    split into two model assemblies differing by one block (§4 is the tidier version of this);
  - keep the channel live (nonzero parameter) if the experiment uses it;
  - **don't** return an identically-zero intermediate as a block output — inline it.

---

## 10. Break self-referential blocks with unknown + target

Canonical for the government budget (debt `B` depends on its own lag and on tax/spending that
depend on `B`), and for any `A → nfa → r → A` loop.

- **Symptom:** `Topological sort failed: cyclic dependency tax_rule -> government -> tax_rule`.
- **Fix:** promote the looping variable to a **model unknown** and add its defining equation as
  a **target**. It becomes exogenous to the DAG; the outer Newton pins it.

```python
@sj.simple
def debt_premium(nfa_u, psi_nfa):            # reads the now-exogenous unknown
    r_prem = -psi_nfa * (nfa_u(-1) - nfa_u.ss)
    return r_prem

@sj.simple
def nfa_consistency(nfa_u, nfa):             # its defining equation, as a target
    nfa_u_res = nfa_u - nfa
    return nfa_u_res

G = model.solve_jacobian(ss, unknowns=U + ['nfa_u'],
                         targets=TG + ['nfa_u_res'], inputs=['C_E_B_S'], T=300)
```
This is the same fix as §13 of the reference but for a *dynamic* loop rather than a
steady-state weight.

---

## 11. Fixed het-level array input (transfer indexed to a steady-state policy)

For a transfer indexed to each household's counterfactual ss energy use
`cE_i,ss = alpha_E * c_ss(a,e)` (a full grid array): two-pass.

```python
ss = model.solve_steady_state(...)                            # transfer inactive at ss (tE=0)
ss['cE_ss_grid'] = alpha_E * ss.internals['hh']['c']          # read ss policy, inject
G  = model.solve_jacobian(ss, ...)                            # held fixed (not in inputs)
```
With **remapped** household copies, index the internals by the renamed block **[ARS]**:
```python
for i in range(3):
    ss[f'cE_ss_grid_{i}'] = ss.internals[f'hh_{i}']['c'] * ss['cE'] / ss['C']
```
The array's value is irrelevant at the ss because the scale multiplying it is 0 there, so the ss
need not know it in advance.

---

## 12. Discount-factor heterogeneity via `rename(suffix=)` + `remap` **[ARS]**

Permanent types, equal shares, identity transition. **Do not add a `beta` axis to the state
space.** ARS replicate the block instead — this is the reference implementation:

```python
group_vars = ['C', 'A', 'MPC']                       # every aggregate you want per-type
hh_list = [hh.rename(suffix=f'_{i}')
             .remap({x: f'{x}_{i}' for x in group_vars})
             .remap({'beta_g': f'beta_{i}'})
           for i in range(3)]

@sj.simple
def group_betas(beta_spread, beta_max):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread/2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2

@sj.simple
def aggregate_groups(C_0, C_1, C_2, A_0, A_1, A_2, MPC_0, MPC_1, MPC_2):
    C = (C_0+C_1+C_2)/3; A = (A_0+A_1+A_2)/3; MPC = (MPC_0+MPC_1+MPC_2)/3
    return C, A, MPC

hh_ha = sj.create_model(hh_list + [group_betas, aggregate_groups])
model = sj.combine([hh_ha, firm, mkt, ...])          # models nest inside models
```
Two things worth noting: `rename(name=None, suffix=None)` **[verified]** — the `suffix` form
renames the block *and* its internals key in one call; and `sj.create_model` output can be
dropped straight into a bigger `sj.combine`, so the household group is a self-contained subgraph.

At the ss, calibrate `beta_max` as the unknown and hold `beta_spread` fixed (ARS: 0.06 with
`beta_max ≈ 0.984`, quarterly); solving `beta_spread` endogenously needs a second target, e.g.
aggregate wealth-to-GDP.

**Alternative (avoid unless forced):** physically add a discount axis to the state,
`(n_e, n_beta, n_a)` with an identity transition. Then **every array must physically broadcast
that axis** — numba `reshape` in the interpolation/lottery step reads flat memory, so a merely
size-compatible / logically-broadcast axis causes silent out-of-bounds reads. Force
materialisation: `+ 0.0 * beta_grid[None, None, :, None]`.

**Watch the asset grid.** A patient type (high `beta(1+r)`) needs a large `max_a` or the forward
iteration diverges (`No convergence after 100000 forward iterations`). Scan `beta → A` on a grid
before choosing the ss bracket.

---

## 13. Recalibration wrapper: what needs a re-solve and what doesn't **[ARS]**

```python
def recalib(shocks, resolve_ss=False, **kwargs):
    ss_here = ss_baseline.copy()
    ss_here.update(kwargs)                       # ResultDict: .update(dict), NOT kwargs
    if resolve_ss:
        ss_here = model.solve_steady_state(ss_here, unknowns_ss, targets_ss,
                                           solver="broyden_custom", dissolve=dissolve)
    else:
        ss_here = model.steady_state(ss_here, dissolve=dissolve)
    test_targets(ss_here, RESID)                 # §0 — non-negotiable
    irf = model.solve_impulse_linear(ss_here, unknowns_td, targets_td, shocks)
    test_targets(irf, RESID)
    return irf
```
- Parameters affecting **only the dynamics** (`rho_i`, `phi_pi`, `theta_w`): patch `ss.copy()`,
  `resolve_ss=False`.
- Parameters affecting the **steady state** (`markup_ss`, `psi_g`, `alpha_E`, any `delta`):
  require `resolve_ss=True`, and `calibration`, `unknowns_ss`, `targets_ss` must all be passed
  explicitly.
- Silent trap: patching an ss-relevant parameter without re-solving gives a *plausible* IRF
  computed around the wrong point. §0's assertions catch it only if the parameter enters a
  residual — often it doesn't. Keep the two paths visibly separate.

---

## 14. API / calibration foot-guns

- `markov_rouwenhorst(rho, sigma, N)`: `sigma` is the cross-sectional std of **log** income;
  the grid is normalised to mean 1.
- CES price/demand formulas with `1/(1-eta)` are **singular at unit elasticity**. To sweep an
  average elasticity `chi` through 1, hold one elasticity off 1 and back out the others (any
  decomposition giving the same `chi` yields the same first-order IRFs). Or branch on the
  parameter, §4.
- `solve_jacobian` **returns the unknowns** (e.g. `Y`, `B`) even though they are not in
  `model.outputs`. When filtering requested outputs use `set(model.outputs) | set(unknowns)`,
  or you will silently zero them out.
- `ss.copy()` returns a `ResultDict`: use `sse.update({'insE': 0.5})` (a dict), not kwargs.
- hetoutputs declared in a `StageBlock` / `@het` block are lowercase at household level
  (`t_E`) and SSJ aggregates them over the stationary distribution into a **capitalised**
  aggregate (`T_E`), available to other blocks as a normal variable. Before concluding an
  aggregate is "missing", grep the hetoutputs.
- `G['out']['shock'] @ dshock` gives the IRF path; guard with
  `if k in G.nesteddict and 'shock' in G[k]` — variables with no exposure to the shock are
  absent from the JacobianDict, not zero-filled.

---

## 15. A hetoutput's nonlinear dependence on the numeraire (`p_num`) corrupts the
linearized Jacobian **[verified]**

**Symptom:** `assets_clearing` (a Walras's-law residual, not itself a solved target under
`TARGETS_TD`) jumps from ~2e-7 to 1e-4–1e-3 on `solve_impulse_linear`, even though: the formula
is verified correct in levels (bit-identical at every date to the version that works), the
isolated Python function is smooth at every finite-difference step tested (1e-4, 1e-6, 1e-8),
and the residual barely changes when the model's discrete choice is made 10x smoother
(`taste_shock` 0.05 → 0.5) — so it is neither a formula bug nor a curvature/logit artifact.

**Cause, empirically isolated (E-HANK household block, StageBlock + LogitChoice + Continuous1D):**
giving a **hetoutput** (a `Continuous1D`/stage `hetoutputs=[...]` function, evaluated on the
already-converged distribution) any **nonlinear** (power-law) dependence on `p_num` breaks
something in how SSJ composes the linearized Jacobian for that hetoutput. This is *not* about
`p_num` per se — it is already a hetinput argument elsewhere (`energy_price_bundle`, `hh_income`)
and a *linear* hetoutput argument elsewhere (`compute_weighted_mpc`: `mpc * p_num * e_grid`) with
no issue in either case. Moving the `p_num` division into the hetinput and handing the hetoutput
pre-divided prices does NOT fix it — it trades this residual for a different (larger) one from
making the hetinput depend on an extra nested-`@solved`-block output (`pHF_P`). The two failure
modes were confirmed independently via a hybrid test (declare the extra argument but leave it
unused in the formula — the residual persists at the same magnitude as when it IS used, ruling
out the arithmetic and pointing at the argument's mere presence / how it's wired).

**Fix:** keep any hetoutput that must divide by `p_num` doing so implicitly, by working in the
SAME base as its raw inputs (CPI-relative here) and converting to the numeraire only in the ONE
place that actually needs it (the budget constraint / Euler equation inside `consav`, itself a
backward function, not a hetoutput). Concretely: compute the CES demand system entirely
CPI-relative (`durable_shares` uses `p_rel`, `pE_B_P`, `pHF_P` as given, no `p_num`), and only
divide by `p_num` once, at the very end, for `p_rel_num` used by the budget constraint. Do not
"clean up" this into a single numeraire-native basis throughout the household block — it is
verified numerically inert for the real economy (the demand ratio is base-invariant) but
measurably corrupts the linearization.

**Not fully root-caused**: likely relates to how SSJ's fake-news/Jacobian machinery times or
composes a hetoutput's dependence on a variable that is *also* used with an explicit lag
elsewhere in the DAG (`p_num` is `pH_P`, used contemporaneously in the hetoutput but as
`pH_P(-1)` in `numeraire_core`'s `r_num`) — but this was not confirmed against SSJ internals,
only against the household block's own behavior. Treat as an empirical constraint, not a
diagnosed SSJ bug with a known internal cause.

## Gotcha: nonlinear inner-block maxit (solve_impulse_nonlinear)

`solve_impulse_nonlinear` inner `sj.solved` blocks (here the retail-energy NKPC,
`energyPrices` / `energyPrices_inner`) default to `maxit=30` at the CLASS level:
`Block.solve_impulse_nonlinear_options = dict(tol=1e-8, maxit=30, verbose=True)`.
The `maxit=` kwarg passed to the top-level solve sets only the OUTER Newton and
does NOT reach these inner blocks; per-block `options=` do not reach them either.
Raise the class default in place before solving:

    from sequence_jacobian.blocks.block import Block
    Block.solve_impulse_nonlinear_options['maxit'] = 500   # instances share the dict

Even so: the ex-post cap does not converge at eps=0.75, and nothing converges at
eps=1.0 inside a warm grid loop (state carried in the model between solves). Cap
robustness runs at eps<=0.5; the size sweep in nl_investigate.py stops at 0.75.
