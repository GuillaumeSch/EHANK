---
name: sequence-jacobian
description: >
  Hard-won gotchas and patterns for building HANK / heterogeneous-agent models with the
  `sequence-jacobian` (SSJ) Python library. Use whenever writing or debugging SSJ code:
  @het / @simple / @solved blocks, hetinputs, steady_state / solve_steady_state,
  jacobian / solve_jacobian, remap, discount-factor heterogeneity, or when hitting cryptic
  SSJ errors (empty SimpleSparse, topological cycles, bracket failures). This is a
  troubleshooting index: match the symptom, apply the fix.
---

# SSJ — gotchas & patterns

Each entry: **symptom → cause → fix (+ minimal snippet)**. Ordered by how often they bite.

## 1. `backward_init` / return-line parsing
SSJ parses the *source text* of the `return` line to name outputs. A `return <expression>`
(not a bare variable) breaks naming.

- **Symptom:** `KeyError: 'Va'` inside `backward_steady_state` at `exog.expectation(ss[k])`.
- **Fix:** assign to the named backward variable and `return` it by name; keep it one flat line.
```python
def hh_init(a_grid, e_grid, r, Z, eis):
    coh = (1+r)*a_grid[None,:] + Z*e_grid[:,None]
    Va = (1+r)*(0.1*coh)**(-1/eis)
    return Va                     # NOT: return (1+r)*(0.1*coh)**(-1/eis)
```
Same rule for hetinput/backward returns: always `return a, b, c` on ONE line, never a
multi-line parenthesised tuple — SSJ matches names verbatim to the return text.

## 2. Empty `SimpleSparse` crash
- **Symptom:** `ValueError: not enough values to unpack (expected 2, got 0)` in
  `sparse_jacobians.py` (`zip(*self.elements.items())`), during `J @ total_Js`.
- **Cause:** a variable whose *composed* Jacobian is identically zero is consumed by a
  downstream block. Two ways this happens: (a) an instrument scaled by a parameter that is
  currently 0 (`tauE*price_gap` with `tauE=0`); (b) exact cancellation along an identity
  (e.g. a household price index `p_c=P_hh/P` that ≡ 1 by the CPI identity when there is no
  subsidy — the partials w.r.t. its inputs cancel to an empty operator).
- **Fixes (pick one):**
  - Make the degenerate variable a **calibration constant** (a scalar in the calib dict), so
    SSJ attaches *no* Jacobian to it. Then split into two model assemblies that differ by one
    block (e.g. a `subsidy_price` block that makes `p_c` live vs. passing `p_c=1.0` as a param).
  - Keep the channel live (nonzero parameter) if the experiment genuinely uses it.
  - **Don't** return an identically-zero intermediate as a block output; inline it instead
    (e.g. fold `subsidy_cost = tauE*price_gap*CE_ss` directly into `budget_res`).

## 3. Break self-referential blocks with unknown + target (government budget)
Canonical for the government budget constraint (debt `B` depends on its own lag and on
tax/spending that depend on `B`).
- **Symptom:** `Topological sort failed: cyclic dependency tax_rule -> government -> tax_rule`,
  or a fragile `@solved` that composes badly.
- **Fix:** promote the looping variable to a **model unknown** and add its defining equation as
  a **target** (residual = 0). It becomes exogenous to the DAG; the outer Newton pins it.
```python
@simple
def tax_rule(B, psiB):
    tauL = psiB * B(-1)                 # reads the lag of the now-exogenous unknown
    return tauL

@simple
def government(B, rante, ..., tauL, wN):
    budget_res = B - ((1+rante(-1))*B(-1) + spending - tauL*wN)   # target = 0
    return budget_res

# GE solve:
G = model.solve_jacobian(ss, unknowns=['Y','B'],
                         targets=['goods_clearing','budget_res'], inputs=['pEstar'], T=300)
```
At the ss, set `B=0.0` in the calib (a provided value) and solve only the remaining ss
unknown (e.g. `beta`); `budget_res=0` holds automatically at `B=0`.

## 4. `@solved` needs a sign-changing bracket; unit-root residuals can't be bracketed
- **Symptom:** `f(a) and f(b) must have different signs`, or `Unable to find a compatible
  one-dimensional solver`.
- **Cause:** the block's steady-state residual is identically 0 for all values of the unknown
  (a unit root). Classic case: a nominal wage law of motion `W_res = W/W(-1) - 1 - pi_w`,
  which is 0 for *any* `W` at the ss (nominal indeterminacy).
- **Fix:** don't put such a block in the model. **Post-process** it after the real solve:
  cumulate `pi_w` to get `d log W`, then `d log P = d log W − d log w`. Only genuinely
  determinate `@solved` blocks (asset price `j`, forward wage NKPC `pi_w`) belong in the model.
- For `@solved` unknowns, provide a bracket `{'x': (lo, hi)}` that actually straddles a sign
  change (a scalar init like `{'x': 0.0}` is rejected by the 1-D solver).

## 5. First-order trick to avoid a household → upstream cycle
If an upstream block (e.g. the government budget) needs an aggregate household output `X_t`
multiplied by a first-order object (a price gap `= pEstar − 1`), the product's deviation is
**second order**. Replace `X_t` with the constant `X_ss`: exact to first order, and it removes
the contemporaneous household→government edge (and the associated empty-Jacobian risk).
```python
subsidy_cost = tauE * price_gap * CE_ss     # CE_ss constant, NOT CE_t
```

## 6. Fixed het-level array input (transfer indexed to a steady-state policy)
For a transfer indexed to each household's *counterfactual ss* energy use
`cE_i,ss = alpha_E * c_ss(a,e)` (a full grid array): two-pass.
```python
ss = model.solve_steady_state(...)                 # transfer inactive at ss (tE=0)
ss['xfer_base'] = alpha_E * ss.internals['household']['c']   # read ss policy, inject
G = model.solve_jacobian(ss, ...)                  # xfer_base held fixed (not in inputs)
```
The array's value is irrelevant at the ss because the transfer scale multiplying it is 0
there, so the ss need not know it in advance.

## 7. Discount-factor heterogeneity via `remap`
Permanent types with equal shares and identity transition: remap one household block into N
copies and average.
```python
to_map = ['beta', 'A', 'C']
hh_lo  = hh.remap({k: k+'_lo'  for k in to_map}).rename('hh_lo')
hh_mid = hh.remap({k: k+'_mid' for k in to_map}).rename('hh_mid')
hh_hi  = hh.remap({k: k+'_hi'  for k in to_map}).rename('hh_hi')
@simple
def aggregate(A_lo, A_mid, A_hi, C_lo, C_mid, C_hi):
    A = (A_lo+A_mid+A_hi)/3; C = (C_lo+C_mid+C_hi)/3
    return A, C
```
Watch the asset grid: a patient type (high `beta(1+r)`) needs a large `amax` or the forward
iteration won't converge (`No convergence after 100000 forward iterations`). Scan `beta→A`
first to bracket the ss unknown.

## 8. API / calibration foot-guns
- `markov_rouwenhorst(rho, sigma, N)`: `sigma` is the cross-sectional std of **log** income;
  the grid is normalised to mean 1.
- CES price/demand formulas with `1/(1-eta)` are **singular at unit elasticity**. To sweep an
  average elasticity `chi` through 1, hold one elasticity off 1 and back out the others
  (any decomposition giving the same `chi` yields the same first-order IRFs).
- `solve_jacobian` **returns the unknowns** (e.g. `Y`, `B`) even though they are not in
  `model.outputs`. When filtering requested outputs, use `set(model.outputs) | set(unknowns)`,
  or you will silently zero them out.
- `ss.copy()` returns a `ResultDict`: use `sse.update({'insE': 0.5})` (a dict), not kwargs.
- Provide `solve_steady_state` brackets that straddle a sign change; if unsure, scan the
  unknown→target map on a grid first.
