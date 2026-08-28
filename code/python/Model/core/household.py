"""E-HANK household with a brown/green durable, built to nest ARS exactly.

DURABLE STATE (4 points) = the PAIR (holding entering the period, holding chosen
last period). Index order:

    0 = BB   brown now, brown before
    1 = BG   brown now, green before
    2 = GB   green now, brown before   <- pays the switching cost psi_g
    3 = GG   green now, green before

Stage order (forward): [dep, prod, durables, consav].

BREAKDOWN MARKOV (`d_markov`), applied in the `dep` stage:
    brown is ABSORBING (no depreciation)      -> BB, BG map to BB w.p. 1
    green breaks down to brown at rate delta_g -> GB, GG map to BB w.p. delta_g,
                                                  and stay green otherwise

NUMERAIRE-GENERIC BUDGET
------------------------
The block never references the CPI directly as the unit of account. It reads
three quantities produced upstream by `numeraire_cpi` or `numeraire_core`:

    p_num       price of the unit of account, RELATIVE TO THE CPI
    r_num       real return in units of account,  1+r_num = (1+r)*p_num(-1)/p_num
    atw_n_num   after-tax labour income in units of account, atw_n / p_num

Assets `a` are denominated in units of account, so the aggregate `A` returned
here is too and must be converted (A_cpi = A * p_num) before it meets the
CPI-denominated objects nfa, j, jF, jE, B in `assets_clearing`.

    p_num = 1     -> ARS's CPI basket numeraire
    p_num = pH_P  -> domestic good numeraire

Everything contractually written in CPI units -- the fiscal transfers epsT and
insE, the ARS energy subsistence term Tf, and the switching cost psi_g -- is
divided by p_num on the way in. Keeping psi_g CPI-denominated is a MODELLING
CHOICE, not a normalisation: denominating it in domestic-good units instead
moves D_GREEN by ~9%. See docs/model.tex.

HOW THE DURABLE ENTERS THE BUDGET (exact CES, not a first-order transfer)
------------------------------------------------------------------------
Each durable type gets its own exact consumption-bundle price index, reusing
ARS's own energy-nest parameters (alpha_E, eta_E) -- no new preference
parameter:

    pE_d(d)  = pE_B_P + IS_GREEN(d) * (pE_G_P - pE_B_P)
    p_rel(d) = [alpha_E*pE_d(d)^(1-eta_E) + (1-alpha_E)*pHF_P^(1-eta_E)]^(1/(1-eta_E))

`p_rel` is CPI-relative and is what the CES demand system (`durable_shares`)
uses; `p_rel_num = p_rel / p_num` is the same bundle price converted to units
of the numeraire and is what the budget constraint uses:

    p_rel_num(d) * c + a' = coh(d)

TWO BASES, ON PURPOSE, NOT BY OVERSIGHT: `durable_shares` stays CPI-relative
and `consav`'s budget constraint is numeraire-relative. A rewrite that made
the whole household block numeraire-native throughout (computing p_rel_num
directly via the general CES formula, in numeraire units, and feeding
`durable_shares` numeraire-relative prices instead) was tried and reverted:
however the p_num-dependence was introduced -- in the hetinput, in the
hetoutput, or split across both -- it corrupted the linearized Jacobian,
producing an assets_clearing (Walras's-law) residual of 1e-4 to 1e-3, against
~2e-7 for this formulation. The economics are unaffected either way (the
demand ratio is base-invariant, verified); this is a numerical-robustness
constraint of the current SSJ version, not a modelling choice. See the note
in `energy_price_bundle` and SSJ_SKILL.md.

EXACT NESTING PROPERTY: for brown states (BB, BG) pE_d = pE_B_P identically,
and CESprices enforces alpha_E*pE_B_P^(1-eta_E) + (1-alpha_E)*pHF_P^(1-eta_E) = 1
at every date, so p_rel(BB) = p_rel(BG) = 1 EXACTLY. Phase-2 nesting is
therefore exact by construction, not merely to solver tolerance.

PRICE-BASE TRAP (numeraire change): every CES demand ratio must have both legs
in the SAME base. `durable_shares` below therefore uses the CPI-relative
`p_rel`, never `p_rel_num`, because pE_d and pHF_P are CPI-relative. Mixing
the two silently rescales cE_dur by p_num and breaks assets_clearing.
"""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D, LogitChoice


# =============================================================================
# 1. DURABLE GRID AND BREAKDOWN MARKOV
# =============================================================================
def make_durable_markov(delta_g):
    """Rows = state today, columns = state next period. Order BB, BG, GB, GG."""
    d_markov = np.array([
        [1.0, 0.0, 0.0, 0.0],                  # BB -> brown absorbing
        [1.0, 0.0, 0.0, 0.0],                  # BG -> brown absorbing
        [delta_g, 0.0, 0.0, 1.0 - delta_g],    # GB -> breaks to brown, else green
        [delta_g, 0.0, 0.0, 1.0 - delta_g],    # GG -> breaks to brown, else green
    ])
    return d_markov


# is the household GREEN this period? (states GB, GG)
IS_GREEN = np.array([0.0, 0.0, 1.0, 1.0])
# does the household PAY the switching cost? (state GB = green chosen, brown held)
PAYS_SWITCH = np.array([0.0, 0.0, 1.0, 0.0])


def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, delta_g):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    d_markov = make_durable_markov(delta_g)
    return e_grid, Pi, a_grid, d_markov


def energy_price_bundle(pB_P, pG_P, p_num):
    p_rel = pB_P + IS_GREEN * (pG_P - pB_P)
    p_rel_num = p_rel / p_num
    return p_rel, p_rel_num


def hh_income(e_grid, atw_n_num, r_num, p_num, pE_B_P, cbarE, scale_w, markup_ss,
              a_grid, n, frisch, ghh_prefs, epsT, cE_ss_grid, insE, pE_P,
              pE_P_ss, psi_g, Tgreen, s_g, Trebate):
    """ARS income plus the switching cost, expressed in units of account.

    The energy-price gap does NOT enter here as an income transfer -- it is
    priced exactly through p_rel_num(d) in consav's budget constraint. Tf, Tfiscal,
    Tswitch and Tgreen are CPI-denominated by construction and are converted by
    dividing by p_num.

    Tgreen is the domestic green-sector rebate (booking='domestic'): the
    competitive green sector supplies green energy and installs green durables
    at zero profit and rebates the proceeds lump-sum. It is 0 under the
    import booking (green energy imported, switching cost booked as import).
    Households still PAY the switching cost (Tswitch) and the green energy
    price (via p_rel) in both bookings; only the BoP counterpart differs.
    """
    atw_n = atw_n_num * p_num
    Tf = - pE_B_P * cbarE * (atw_n * markup_ss) * scale_w - pE_B_P * cbarE * (1 - scale_w)
    # Slutsky transfer: lump sum indexed to pre-crisis BROWN energy (cE_ss_grid,
    # collapsed over the durable axis). Added to the (e,a) cash-on-hand BEFORE
    # the durable broadcast, so it is identical across durable states -- a
    # household that switches to green keeps the same transfer and the switching
    # margin is undistorted. epsT is the untargeted lump sum.
    Tfiscal = epsT + insE * (pE_P - pE_P_ss) * cE_ss_grid
    # Green/adoption subsidy: the government pays a fraction s_g of the switching
    # cost, so the household pays only (1-s_g)*psi_g. s_g = 0 at the SS.
    Tswitch = - (1 - s_g) * psi_g * PAYS_SWITCH

    # Trebate is the carbon-revenue lump-sum rebate (ets=True); 0 otherwise.
    coh = ((1 + r_num) * a_grid + atw_n_num * e_grid[:, np.newaxis]
           + (Tf + Tfiscal + Tgreen + Trebate) / p_num)
    coh = coh[np.newaxis, ...] + (Tswitch / p_num)[:, np.newaxis, np.newaxis]

    n_ss = 1
    ghh = ghh_prefs * 1 / (1 + 1 / frisch) * (n ** (1 + 1 / frisch) - n_ss ** (1 + 1 / frisch))
    return coh, ghh


def hh_init(coh, r_num, eis):
    Va = (1 + r_num) * (0.1 * coh) ** (-1 / eis)
    V = Va / (1 - 0.98)
    return Va, V


# =============================================================================
# 2. STAGES
# =============================================================================
dep_stage = ExogenousMaker(markov_name='d_markov', index=0, name='dep')
prod_stage = ExogenousMaker(markov_name='Pi', index=1, name='prod')


def util_l(V, green_block, coh):
    """Flow payoff of choosing durable d given the durable held, on (d | d_).
    coh[2] (state GB) is net of psi_g, so the feasibility guard is a genuine
    budget check and is numeraire-invariant (a sign test)."""
    gb = -green_block
    infeasible = coh[2] <= 0.0
    gb_masked = np.where(infeasible, -1e10, gb)

    flow_u = np.full((4, 4) + V.shape[1:], -1e10)
    flow_u[0, 0] = 0.
    flow_u[0, 1] = 0.
    flow_u[2, 0] = gb_masked
    flow_u[2, 1] = gb_masked
    flow_u[3, 2] = 0.
    flow_u[3, 3] = 0.
    return flow_u


durables_stage = LogitChoice(value='V', backward='Va', index=0, name='durables',
                             taste_shock_scale='taste_shock', f=util_l)


def consav(V, Va, a_grid, r_num, beta_g, eis, coh, ghh, p_rel_num):
    """EGM step with a type-specific bundle price, in units of account.
    Budget:   p_rel_num(d) * c + a' = coh(d)
    Envelope: Va = (1+r_num) * uc / p_rel_num
    Euler:    uc = beta * p_rel_num * Va_cont"""
    p_rel_bc = p_rel_num[:, np.newaxis, np.newaxis]

    uc_nextgrid = beta_g * p_rel_bc * Va
    c_nextgrid = uc_nextgrid ** (-eis) + ghh          # bundle quantity c, endogenous grid
    a = sj.interpolate.interpolate_y(p_rel_bc * c_nextgrid + a_grid, coh, a_grid)
    np.maximum(a, a_grid[0], out=a)   # setmin() is hardcoded 2-D; state is 3-D here
    c = (coh - a) / p_rel_bc
    # Floor consumption net of ghh away from 0 (feasibility guard in util_l).
    c_safe = np.maximum(c - ghh, 1e-10)
    Va = (1 + r_num) * c_safe ** (-1 / eis) / p_rel_bc
    if eis == 1:
        u = np.log(c_safe)
    else:
        u = c_safe ** (1 - 1 / eis) / (1 - 1 / eis)
    V = u + beta_g * sj.interpolate.interpolate_y(a_grid, a, V)
    return Va, V, a, c


def compute_weighted_mpc(c, a_grid, r_num, e_grid, p_num):
    """MPC out of a marginal unit of CPI income.

    dc/dcoh is measured in units of account, so multiplying by p_num makes the
    statistic numeraire-invariant: it equals dc/dcoh_cpi in both bases and
    reduces to the ARS definition when p_num = 1.
    """
    mpc = np.empty_like(c)
    post_return = (1 + r_num) * a_grid
    mpc[..., 1:-1] = (c[..., 2:] - c[..., :-2]) / (post_return[2:] - post_return[:-2])
    mpc[..., 0] = (c[..., 1] - c[..., 0]) / (post_return[1] - post_return[0])
    mpc[..., -1] = (c[..., -1] - c[..., -2]) / (post_return[-1] - post_return[-2])
    mpc = mpc * p_num * e_grid[:, np.newaxis]
    return mpc


def durable_shares(c, p_rel, pE_B_P, pE_G_P, pHF_P, alpha_E, eta_E):
    """Population shares, switching flow, and the household's EXACT CES demand
    split, by durable type. These aggregate (over the stationary distribution
    and across beta types) into CE_DUR_B / CE_DUR_G / CHF_DUR, which
    blocks.hh_outputs_dur uses to build cH/cF/cE.

    Why this is needed: ARS's own hh_outputs computes cE = alpha_E *
    pE_B_P^(-eta_E) * C, i.e. the CES demand evaluated at the AGGREGATE price
    index. With durable-type-specific prices that is wrong by Jensen: the true
    aggregate is sum_d of the type-specific demands.

    PRICE BASE: pE_d and pHF_P are CPI-relative, so the denominator must be
    the CPI-relative p_rel, NOT p_rel_num. Mixing bases rescales every demand
    by p_num and breaks assets_clearing under the core numeraire -- verified
    concretely (see `energy_price_bundle`), not just asserted.
    """
    d_green = np.zeros_like(c) + IS_GREEN[:, np.newaxis, np.newaxis]
    d_switch = np.zeros_like(c) + PAYS_SWITCH[:, np.newaxis, np.newaxis]

    pE_d = pE_B_P + IS_GREEN * (pE_G_P - pE_B_P)
    p_rel_bc = p_rel[:, np.newaxis, np.newaxis]
    cE_dur = alpha_E * (pE_d[:, np.newaxis, np.newaxis] / p_rel_bc) ** (-eta_E) * c
    cHF_dur = (1 - alpha_E) * (pHF_P / p_rel_bc) ** (-eta_E) * c

    cE_dur_b = cE_dur * (1.0 - IS_GREEN[:, np.newaxis, np.newaxis])
    cE_dur_g = cE_dur * IS_GREEN[:, np.newaxis, np.newaxis]
    # Bundle consumption split by current technology (for the cross-sectional
    # decomposition of C: C = C_BROWN + C_GREEN, masses 1-D_GREEN / D_GREEN).
    c_green = c * IS_GREEN[:, np.newaxis, np.newaxis]
    c_brown = c * (1.0 - IS_GREEN[:, np.newaxis, np.newaxis])
    c_switch = c * PAYS_SWITCH[:, np.newaxis, np.newaxis]   # new adopters (state GB)
    p_times_c = p_rel_bc * c
    return d_green, d_switch, cE_dur_b, cE_dur_g, cHF_dur, c_green, c_brown, c_switch, p_times_c


def flow_utility(c, ghh, eis):
    """Per-period felicity, aggregated by SSJ into UTIL over the distribution.
    Discounted with each type's own beta this gives utilitarian welfare, and
    hence the consumption-equivalent variation (see ehank/welfare.py)."""
    c_safe = np.maximum(c - ghh, 1e-10)
    if eis == 1:
        util = np.log(c_safe)
    else:
        util = c_safe ** (1 - 1 / eis) / (1 - 1 / eis)
    return util


consav_stage = Continuous1D(backward=['V', 'Va'], policy='a', f=consav, name='consav',
                            hetoutputs=[compute_weighted_mpc, durable_shares, flow_utility])


# =============================================================================
# 3. ASSEMBLY (beta heterogeneity via the ARS rename/remap idiom)
# =============================================================================
hh_one = StageBlock([dep_stage, prod_stage, durables_stage, consav_stage], name='hh',
                    backward_init=hh_init,
                    hetinputs=[make_grids, energy_price_bundle, hh_income])

GROUP_VARS = ['C', 'A', 'MPC', 'cE_ss_grid', 'D_GREEN', 'D_SWITCH', 'CE_DUR_B', 'CE_DUR_G', 'CHF_DUR', 'UTIL', 'C_GREEN', 'C_BROWN', 'C_SWITCH', 'P_TIMES_C']


@sj.simple
def group_betas(beta_spread, beta_max):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread / 2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2


@sj.simple
def aggregate_groups(C_0, C_1, C_2, A_0, A_1, A_2, MPC_0, MPC_1, MPC_2,
                     D_GREEN_0, D_GREEN_1, D_GREEN_2, D_SWITCH_0, D_SWITCH_1, D_SWITCH_2,
                     CE_DUR_B_0, CE_DUR_B_1, CE_DUR_B_2, CE_DUR_G_0, CE_DUR_G_1, CE_DUR_G_2,
                     CHF_DUR_0, CHF_DUR_1, CHF_DUR_2,
                     C_GREEN_0, C_GREEN_1, C_GREEN_2, C_BROWN_0, C_BROWN_1, C_BROWN_2,
                     C_SWITCH_0, C_SWITCH_1, C_SWITCH_2,
                     P_TIMES_C_0, P_TIMES_C_1, P_TIMES_C_2,
                     beta_0, beta_1, beta_2):
    C = (C_0 + C_1 + C_2) / 3
    A = (A_0 + A_1 + A_2) / 3
    MPC = (MPC_0 + MPC_1 + MPC_2) / 3
    D_GREEN = (D_GREEN_0 + D_GREEN_1 + D_GREEN_2) / 3
    D_SWITCH = (D_SWITCH_0 + D_SWITCH_1 + D_SWITCH_2) / 3
    CE_DUR_B = (CE_DUR_B_0 + CE_DUR_B_1 + CE_DUR_B_2) / 3
    CE_DUR_G = (CE_DUR_G_0 + CE_DUR_G_1 + CE_DUR_G_2) / 3
    CHF_DUR = (CHF_DUR_0 + CHF_DUR_1 + CHF_DUR_2) / 3
    C_GREEN = (C_GREEN_0 + C_GREEN_1 + C_GREEN_2) / 3
    C_BROWN = (C_BROWN_0 + C_BROWN_1 + C_BROWN_2) / 3
    C_SWITCH = (C_SWITCH_0 + C_SWITCH_1 + C_SWITCH_2) / 3
    beta = (beta_0 + beta_1 + beta_2) / 3
    P_times_C = (P_TIMES_C_0 + P_TIMES_C_1 + P_TIMES_C_2) / 3
    return C, A, MPC, D_GREEN, D_SWITCH, CE_DUR_B, CE_DUR_G, CHF_DUR, C_GREEN, C_BROWN, C_SWITCH, P_times_C, beta


def hh_ha_durable(n_beta=3):
    hh_list = [hh_one.rename(suffix=f'_{i}')
                     .remap({x: f'{x}_{i}' for x in GROUP_VARS})
                     .remap({'beta_g': f'beta_{i}'})
               for i in range(n_beta)]
    return sj.create_model(hh_list + [group_betas, aggregate_groups], name='hh_ha_durable')
