# Fiscal-policy extension (Auclert et al. 2023, Section 5) — single-beta

Government added to the baseline HA model. Implemented and verified: steady state, transition
dynamics, and Fig. 15 / Fig. 16 (single-beta). Magnitudes are indicative — see caveat.

## Programs
- **Energy subsidy** (price-facing): households face `pE_hh = (1-tauE)pE + tauE*pE_ss`, entering
  as a cost-of-living wedge `p_c = P_hh/P <= 1` carried through the EGM (`u'(c)/p_c = beta(1+r)E[Va']`,
  budget `a' = (1+r)a + income - p_c*c`). Home demand and energy demand carry the `p_c` wedge.
- **Targeted transfers**: `T_i = insE * cE_i,ss * (pE-pE_ss)`, indexed to each household's ss
  energy use `cE_i,ss = alpha_E*c_ss` (fixed grid array injected two-pass).
- **Untargeted transfers**: equal lump-sum, shock-driven.
- Financing: proportional labor tax `tauL = psiB*B(-1)`; the wage NKPC uses the after-tax wage.

## Key SSJ structure (see SSJ_SKILL.md)
- **Government budget via unknown + target**: `B` is a model unknown, the budget constraint is
  the target `budget_res=0`. GE system `[Y, B] / [goods_clearing, budget_res]`. This breaks the
  debt self-reference cleanly (no `@solved` on `B`).
- **Two assemblies differing by one block** to keep every Jacobian non-degenerate: `model_tr`
  (transfers; `p_c=1` as a calibration constant) for nothing/targeted/untargeted, and `model_sub`
  (subsidy; `p_c` a live variable) for the subsidy. Needed because `p_c ≡ 1` when `tauE=0` has an
  identically-zero composed Jacobian that crashes SSJ if consumed downstream.
- Subsidy cost in the budget uses `CE_ss` (first-order exact: `(CE-CE_ss)*price_gap` is 2nd order).

## Steady state
Unchanged: all fiscal flows vanish at the ss (`pE=pE_ss`, `p_c=1`), so `tauL=0`, `B_ss=0`, and
`nfa=A-j-B=0` pins `beta=0.9564`, `A=j=2.913`, `C=Y=1`.

## Results (energy shock +100% impact, rho=0.96)

| | min Y | min C | peak B (%ssY) | peak pi_w (ann) | peak hh-CPI infl (ann) |
|---|---|---|---|---|---|
| nothing    | -3.42% | -6.57% | 0    | 2.14% | 24.97% |
| subsidy    | -3.05% | -5.09% | 21.85 | 2.39% | 17.22% |
| targeted   | -2.80% | -4.86% | 21.85 | 2.92% | 25.75% |
| untargeted | -2.94% | -4.94% | 21.85 | 2.92% | 25.75% |

Matches the paper's Fig. 15/16 message: all three programs cushion output and consumption
(deficit-financed, slowly repaid via `tauL`, small late reversal); the **subsidy tames inflation**
(household energy price ~+50% vs +100%; hh-CPI inflation 17% vs 25%) while **transfers raise wage
inflation** (2.9% vs 2.4%) by stimulating demand.

## Caveat
Built on the Section-3 baseline (flexible home prices, no slow pass-through, no domestic energy
endowment). The paper runs the fiscal experiments on the Section-4 quantitative model, so
magnitudes/shapes differ; the mechanisms and cross-program ordering are the target here.

## Files
- `hh_block_gov.py` — household with fiscal cash-on-hand terms + subsidy price wedge `p_c`
- `blocks_gov.py` — real + government blocks (both `subsidy_price` and `market_energy_price`)
- `main_build_gov.py` — ss + targeted/untargeted transition (earlier, cash-equiv subsidy)
- `gen_fig15_16.py` — two-model assembly, Fig. 15 and Fig. 16
- `SSJ_SKILL.md` — distilled SSJ gotchas & patterns
