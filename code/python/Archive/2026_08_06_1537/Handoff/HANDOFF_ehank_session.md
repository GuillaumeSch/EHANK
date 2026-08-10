# E-HANK — Session handoff (Boris exercises + Version A)

Continuity note for resuming in a fresh conversation. Pairs with the existing
`HANDOFF_ehank.md`, `CLAUDE.md`, `SSJ_SKILL.md`. Discussion in French, code and
identifiers in English, as usual. Deadline: **1 Oct 2026** (SECO).

---

## 0. One-line status

Implemented and tested five additions requested by Boris (#3 monetary shock,
#4 untargeted transfer, #1 green subsidy, #5 ETS/carbon tax steady state,
Version A ex-ante/ex-post), plus three analysis deliverables (persistence sweep,
5-feature summary table, welfare-optimal carbon figure) and a ready-to-paste
LaTeX Section 5. Baseline non-regression is **bit-identical** throughout
(`psi_g=0.53804040`, `D^G_ss=0.05`, price-shock `y(0)=-2.7512%`,
peak `D^G=8.87pp`). All work is in the sandbox copy; **not yet merged into the
real repo** at `.../SOE_Durable_07_24_1445/`.

Everything below refers to the calibration **as uploaded** (`taste_shock=0.05`),
which differs from the numbers quoted in the current `.tex`
(`taste_shock=1e-2`, peak `D^G` 9.46/9.18). The tables are `\input`-generated,
so they self-update on whichever calibration is active; the prose in Section 5
quotes the `taste_shock=0.05` numbers.

---

## 1. Files changed (sandbox) and what to merge

All in the flat package. Deliverable copies are in `/mnt/user-data/outputs/ehank/`.

**Modified core files**
- `blocks.py`
  - `energyPrices`: `+tau_b` arg; now also returns `pE_B_pretax_P` (pre-carbon-tax
    consumer brown price). `pE_B_P = pE_B_pretax_P*(1+tau_b)`.
  - `green_energy_price`: `+pE_P, +tau_g`; anchored on the **pre-tax** base
    `pE_P.ss*(1+tau_g)` (decouples green price from the brown carbon tax).
  - `energy_gap`: now uses **pre-tax** prices on both legs (`pE_B_pretax_P`,
    `pE_G_P/(1+tau_g)`) — the carbon tax is a domestic wedge, must not enter the
    BoP import bill.
  - `CESprices`: inner_nest anchored on `pE_B_pretax_P` (not `pE_B_P`).
  - `fiscal`: carbon revenue/rebate **folded in** (so it enters the asset
    identity via B). `+tau_b, tau_g, pE_g_ratio, CE_DUR_G, Trebate, pE_B_pretax_P`.
    `R_carbon = tau_b*pE_B_pretax_P*CE_DUR_B + tau_g*(pE_g_ratio*pE_P.ss)*CE_DUR_G`;
    `Subsidy_green = (s_g - s_g.ss)*psi_g*D_SWITCH` (transitory part only,
    debt-financed); permanent part `s_g.ss*psi_g*D_SWITCH` carbon-financed;
    `Trebate_res = Trebate - (R_carbon - green_subsidy_p)`; carbon nets out of
    `B_res` (budget-neutral). Returns `..., R_carbon, Trebate_res`.
  - `carbon_sector`: **removed** (was a separate balanced account; that broke
    Walras — see §3).
- `household.py`
  - `energy_price_bundle`: `+pE_B_pretax_P`; the CES identity is anchored on the
    pre-tax price while `pE_d` for brown stays the tax-inclusive `pE_B_P`
    → carbon tax has real incidence (`p_rel(brown)>1`), not absorbed by the CPI.
  - `hh_income`: `+s_g, +Trebate`. `Tswitch = -(1-s_g)*psi_g*PAYS_SWITCH`;
    `Trebate` added to `coh` (uniform lump sum).
- `calibration.py`
  - New `ETS = dict(tau_b=0, tau_g=0, s_g_ets=0)` layer; `POLICY` gains `s_g=0`.
  - `set_energy_grids_flat(calib, ss)`: fills `cE_ss_grid_i` with the uniform
    scalar `CE_DUR_B.ss` → untargeted transfer, same envelope, flat incidence.
  - `make_calibration(numeraire, booking, ets=False, **ov)`: `c.update(ETS)`;
    pops `Trebate` when `ets=True` (becomes a model unknown, mirroring `Tgreen`
    under domestic booking); `_derived` seeds `Trebate:0.0`.
- `model.py`
  - `shock_mon(size=0.0025, half_life=4)` → `{'ishock': path}`; `shock_green(size,
    half_life)` → `{'s_g': path}` (transitory, 0 at SS).
  - `build_model(numeraire, booking, ets=False)`: `ets` documented only; carbon
    is in `fiscal`, so `build_model(...)` is the same model either way (no separate
    block). One model serves baseline and ETS.
  - `td_unknowns_targets(booking, ets)`, `ss_unknowns_targets(booking, ets)`,
    `ss_unknowns_targets_fixed_psi(booking, ets)`: add `Trebate`/`Trebate_res`
    when `ets=True`.
  - `run(..., ets=False, ets_kwargs=None)`: closure now `inelastic` iff
    `shock_kind=='supply'`; dispatches `price`/`monetary`/`supply`; layers
    `shock_green` for `policy=='green'`; `transfer_flat` uses the flat grid
    filler; `ets=True` solves the prepared SS (psi_g fixed at the no-ETS baseline,
    D^G floats) with `ets_kwargs=dict(tau_b, tau_g, s_g_ets, recycle)`.
  - `POLICIES` gains `transfer_flat` and `green`; `MODELS`/`MONETARY`/`ENERGY_
    CLOSURE` unchanged.
- `welfare.py`
  - `cev_total(base_ss, pre_ss, irf)`: total CEV of a full scenario vs a common
    reference `base_ss` (standing gap `pre_ss` vs `base_ss` + crisis transition,
    incl. labour-disutility level and deviation). Use this for the ex-ante/ex-post
    comparison; `cev` is unchanged.

**New runners**
- `run_exante_expost.py` — Version A: crisis comparison (LF, cap, transfer,
  ETS, ETS+cap), standing-CEV `tau_b` sweep, and the welfare-optimal carbon
  figure. Emits `output/tab_exante_expost_import.tex`,
  `fig_exante_expost_import.pdf`, `fig_carbon_optimal_import.pdf`.
- `run_persistence.py` — Route A: half-life sweep via the GE Jacobian `G`
  (solve `G` once per SS, apply to every shock). Emits
  `tab_persistence_import.tex`, `fig_persistence_import.pdf`.
- `run_summary_table.py` — two-panel 5-feature table (Panel A price-shock
  policies incl. flat transfer / green subsidy / ETS; Panel B monetary shock).
  Emits `tab_summary_import.tex`.

**New LaTeX**
- `section5_exante.tex` — Section 5, `\input`-ing the three tables and two
  figures. Paste after Experiments (Sec. 4), before Relation to the literature.
  Citation keys `\citet{ARS2024,Bayer2026,Langot2026}` match the manuscript.

---

## 2. Headline numbers (taste_shock=0.05, import booking, H=24)

**Price-shock policies (total CEV vs no-tax SS):**
| scenario | y(0)% | ΣC% | peak D^G | gross fisc | CEV% |
|---|---|---|---|---|---|
| Laissez-faire | −2.75 | −105.6 | 8.87 | 0 | −1.871 |
| Ex-post cap (τ^E=1) | −1.66 | −78.9 | −0.04 | 53.9 | −1.147 |
| Slutsky transfer | −0.63 | −43.4 | 9.15 | 49.3 | −0.816 |
| **Flat transfer** | −0.31 | −37.3 | 9.10 | 49.3 | **−0.317** |
| Green subsidy | −3.19 | −118.5 | 18.02 | 2.1 | −2.032 |
| Ex-ante ETS (τ_b=0.10) | −2.85 | −106.5 | 14.98 | 0.1 | −1.852 |

CEV by type — Slutsky (−0.68, −0.89, −0.88) vs flat (+0.10, −0.28, −0.77).

**Monetary shock (100bp ann. tightening):** Taylor i(0)=+0.36, y(0)=−0.85,
π(0)=−0.43, C(0)=−0.95; real-rate rule i(0)=+0.63, y(0)=−1.13, C(0)=−1.27.

**Standing carbon sweep (no crisis):** τ_b 0/0.05/0.10/0.20/0.30 →
standing CEV 0 / +0.029 / **+0.043** / +0.027 / −0.043; D^G_ss 5.0/6.8/8.9/14.2/20.1%.
Optimum **τ_b\*≈10%**.

**Persistence (Route A):** greening dividend (crisis-only) +0.011 (HL4) →
−0.024 (HL16) → −0.089 (HL48); crosses 0 near HL≈12. Cap protection grows
+0.24 → +1.36.

---

## 3. Key learnings this session (keep these — they cost time)

1. **The ETS Walras leak and its fix (most important).** A permanent carbon tax
   on brown energy leaked into `assets_clearing` (−0.567 at τ_b=0.10, linear in
   τ_b). Ruled out: not the carbon revenue flow (160× too small), not the
   numeraire (identical leak under core and cpi), not subsistence (`cbarE=0`).
   **Root cause:** `energy_price_bundle` builds `p_rel(brown)=1` *structurally*
   by anchoring on `pE_B_P` itself, and `CESprices` normalises the CPI on the
   same `pE_B_P`; so the tax that inflates `pE_B_P` was fully absorbed by the CPI
   renormalisation (pHF_P falls), brown households stayed at `p_rel=1` and never
   paid the tax in real terms, yet R_carbon collected and rebated it.
   **Fix:** anchor both `CESprices` and `energy_price_bundle` on the **pre-tax**
   price `pE_B_pretax_P`, keeping `pE_d=pE_B_P` (tax-inclusive) for brown. The
   tax then has real incidence (`p_rel(brown)>1`), the revenue is real, Walras
   closes. Bit-identical at τ_b=0.

2. **Carbon must enter an active clearing condition.** A separate balanced
   `carbon_sector` (R in, Trebate out, net zero) does **not** fix Walras, because
   it sits outside the asset identity. Folding R_carbon/Trebate into `fiscal`
   (revenue in `taxation`, rebate in `spending`) puts it on the B axis. Necessary
   but not sufficient — item 1 was the real fix.

3. **Carbon tax base under a cap.** `R_carbon` must use the **pre-tax consumer**
   brown price `pE_B_pretax_P`, not the world price `pE_P`. Under a cap the
   consumer pre-tax price is pinned, so ETS and cap compose (households are taxed
   on what they actually pay). Using `pE_P` broke ETS+cap (`assets_clearing=1.18`).

4. **Green subsidy must be transitory** to keep the E2/E3 comparison like-for-
   like: `s_g=0` at SS, fed as a path (`shock_green`). A permanent `s_g` in a
   static POLICIES override would shift the SS.

5. **Flat vs targeted transfer** differ *only* in the household incidence vector
   (`cE_ss_grid` targeted vs uniform `CE_DUR_B.ss` flat); the fiscal envelope is
   already computed on the aggregate, so both cost the same. The flat transfer
   dominates on utilitarian CEV because homothetic CES makes energy use track
   wealth → energy-indexing is regressive. **Caveat:** conditional on homothetic
   demand; a necessity (declining Engel curve) could flip it.

6. **Route A mechanism** (paper-relevant): the adoption margin is an endogenous
   crisis absorber that works better from a browner start; pre-greening spends
   it, so the greening dividend turns negative for persistent shocks.

7. **Solver limit:** τ_b>0.30 breaks the fixed-psi SS brentq bracket (D^G too
   high). To go higher, widen brackets on `P`/`B`/`inom` or use broyden.

8. **Persistence sweeps: use the GE Jacobian.** Persistence moves only the shock,
   not the SS. Compute `G = model.solve_jacobian(ss, u, t, inputs=['PEstar_
   shock'], outputs=[...], T=T)` once per SS, then `irf[o] = G[o]['PEstar_shock']
   @ shock` — matches `solve_impulse_linear` exactly and is ~10× faster.

---

## 4. Open items / next steps (priority order)

1. **Merge to the real repo** (`.../SOE_Durable_07_24_1445/`) and re-run the
   existing non-regression suite (`tests.py`, the four post-session checks:
   envelope, monotonicity, stock-flow, initial-conditions). The block-signature
   changes (energyPrices return arity, fiscal signature) touch the DAG, so
   re-validate the phase-2 nesting test in particular.
2. **Domestic-booking robustness** of the three Version-A deliverables. Version A
   used import booking; the welfare/CEV ranking is claimed booking-robust but is
   only verified under import. `run(..., ets=True, booking='domestic')` is
   untested — the ETS + domestic green_sector interaction needs a Walras check
   (both `Tgreen` and `Trebate` become unknowns simultaneously).
3. **Homotheticity caveat** on the flat-transfer result: add a non-homothetic
   energy nest (or at least a sensitivity note) so §5.1's claim is bounded. This
   is the most exposed result to a referee.
4. **Nonlinearity (N4).** All CEVs are first-order. At shock size 1.0 the logit
   tail (`taste_shock=0.05` → σ/σ_grid large) makes linearisation unreliable;
   the ex-ante/ex-post ranking should be re-checked at reduced shock size
   (0.25–0.5) or with a nonlinear solve before the welfare numbers go in the
   paper as levels.
5. **σ_ε identification** (pre-existing): still needs a second empirical moment
   (Muehlegger–Rapson 2022; Beresteanu–Li 2011). The τ_b\* result is
   qualitative; its magnitude moves with σ_ε.
6. Optional: regenerate all three deliverables under `taste_shock=1e-2` to match
   the manuscript's other tables, or migrate the manuscript to `taste_shock=0.05`
   — pick one and be consistent.

---

## 5. How to reproduce (sandbox)

```bash
pip install --break-system-packages sequence-jacobian==1.0.0 numba
python run_summary_table.py     # Panel A/B table
python run_exante_expost.py     # Version A + carbon-optimal figure
python run_persistence.py       # Route A
```

Non-regression one-liner (must stay bit-identical):
```python
from model import build_model, run; import numpy as np
M = build_model('core', booking='import')
ss, irf = run(M, shock_kind='price', policy='none')
assert abs(float(ss['psi_g'])-0.53804040) < 1e-6
assert abs(100*float(np.asarray(irf['y'])[0]) + 2.7512) < 1e-3
```

---

## 6. Paper positioning reminder

Singular contribution unchanged: **Bayer, Langot and ARS freeze energy
technology; E-HANK adds the one margin they omit.** The new results sharpen it —
the adoption margin is an endogenous shock absorber (Route A), energy-indexed
targeting is regressive vs a flat transfer, and a permanent carbon tax is a
weak substitute for ex-post insurance. No "nesting/reconciling" language.
