"""E-HANK household with a brown/green durable, nesting the ARS household.

Durable state: 0=BB, 1=BG, 2=GB (pays switching cost psi_g), 3=GG
(holding now, holding chosen last period). Stage order: dep, prod, durables, consav.
"""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D, LogitChoice


def make_durable_markov(delta_g):
    """Rows = state today, columns = state next period. Order BB, BG, GB, GG."""
    d_markov = np.array([
        [1.0, 0.0, 0.0, 0.0],                  # BB -> brown absorbing
        [1.0, 0.0, 0.0, 0.0],                  # BG -> brown absorbing
        [delta_g, 0.0, 0.0, 1.0 - delta_g],    # GB -> breaks to brown, else green
        [delta_g, 0.0, 0.0, 1.0 - delta_g],    # GG -> breaks to brown, else green
    ])
    return d_markov


# is the household green this period? (states GB, GG)
IS_GREEN = np.array([0.0, 0.0, 1.0, 1.0])
# does the household pay the switching cost? (state GB = green chosen, brown held)
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
    """ARS income plus the switching cost, in units of account."""
    atw_n = atw_n_num * p_num
    Tf = - pE_B_P * cbarE * (atw_n * markup_ss) * scale_w - pE_B_P * cbarE * (1 - scale_w)
    # Slutsky transfer, indexed to pre-crisis brown energy; epsT is the untargeted sum.
    Tfiscal = epsT + insE * (pE_P - pE_P_ss) * cE_ss_grid

    Tswitch = - (1 - s_g) * psi_g * PAYS_SWITCH

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


dep_stage = ExogenousMaker(markov_name='d_markov', index=0, name='dep')
prod_stage = ExogenousMaker(markov_name='Pi', index=1, name='prod')


def util_l(V, green_block, coh):
    """Flow payoff of choosing durable d given the durable held."""
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
    """EGM consumption-savings step with a type-specific bundle price."""
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
    """MPC out of a marginal unit of CPI income."""
    mpc = np.empty_like(c)
    post_return = (1 + r_num) * a_grid
    mpc[..., 1:-1] = (c[..., 2:] - c[..., :-2]) / (post_return[2:] - post_return[:-2])
    mpc[..., 0] = (c[..., 1] - c[..., 0]) / (post_return[1] - post_return[0])
    mpc[..., -1] = (c[..., -1] - c[..., -2]) / (post_return[-1] - post_return[-2])
    mpc = mpc * p_num * e_grid[:, np.newaxis]
    return mpc


def durable_shares(c, p_rel, pE_B_P, pE_G_P, pHF_P, alpha_E, eta_E, psi_g_bar,
                   atw_n_num, e_grid):
    """Population shares, switching flow, and CES demand by durable type."""
    d_green = np.zeros_like(c) + IS_GREEN[:, np.newaxis, np.newaxis]
    d_switch = np.zeros_like(c) + PAYS_SWITCH[:, np.newaxis, np.newaxis]


    lab_inc = np.zeros_like(c) + atw_n_num * e_grid[np.newaxis, :, np.newaxis]
    # lab_inc = np.zeros_like(c) + e_grid[np.newaxis, :, np.newaxis]
    lab_inc_green = lab_inc * IS_GREEN[:, np.newaxis, np.newaxis]
    lab_inc_brown = lab_inc * (1.0-IS_GREEN[:, np.newaxis, np.newaxis])

    cHF_switch = psi_g_bar * d_switch

    pE_d = pE_B_P + IS_GREEN * (pE_G_P - pE_B_P)
    p_rel_bc = p_rel[:, np.newaxis, np.newaxis]
    cE_dur = alpha_E * (pE_d[:, np.newaxis, np.newaxis] / p_rel_bc) ** (-eta_E) * c
    cHF = (1 - alpha_E) * (pHF_P / p_rel_bc) ** (-eta_E) * c

    cE_b = cE_dur * (1.0 - IS_GREEN[:, np.newaxis, np.newaxis])
    cE_g = cE_dur * IS_GREEN[:, np.newaxis, np.newaxis]
    # split bundle consumption by current technology
    c_green = c * IS_GREEN[:, np.newaxis, np.newaxis]
    c_brown = c * (1.0 - IS_GREEN[:, np.newaxis, np.newaxis])
    c_switch = c * PAYS_SWITCH[:, np.newaxis, np.newaxis]   # new adopters (state GB)
    p_times_c = p_rel_bc * c

    # group specific cHF values 
    cHF_green =  cHF * IS_GREEN[:, np.newaxis, np.newaxis]
    cHF_brown = cHF * (1.0 - IS_GREEN[:, np.newaxis, np.newaxis])

    return d_green, d_switch, cE_b, cE_g, cHF, c_green, c_brown, c_switch, p_times_c, cHF_switch, cHF_green, cHF_brown, lab_inc, lab_inc_brown, lab_inc_green


def flow_utility(c, ghh, eis):
    """Per-period felicity, aggregated by SSJ into UTIL."""
    c_safe = np.maximum(c - ghh, 1e-10)
    if eis == 1:
        util = np.log(c_safe)
    else:
        util = c_safe ** (1 - 1 / eis) / (1 - 1 / eis)
    return util


consav_stage = Continuous1D(backward=['V', 'Va'], policy='a', f=consav, name='consav',
                            hetoutputs=[compute_weighted_mpc, durable_shares, flow_utility])


hh_one = StageBlock([dep_stage, prod_stage, durables_stage, consav_stage], name='hh',
                    backward_init=hh_init,
                    hetinputs=[make_grids, energy_price_bundle, hh_income])

GROUP_VARS = ['C', 'A', 'MPC', 'cE_ss_grid', 'D_GREEN', 'D_SWITCH', 'CE_B', 'CE_G', 'CHF', 'UTIL', 'C_GREEN', 'C_BROWN', 'C_SWITCH', 'P_TIMES_C', 'CHF_SWITCH', 'CHF_GREEN', 'CHF_BROWN', 'LAB_INC', 'LAB_INC_GREEN', 'LAB_INC_BROWN']


@sj.simple
def group_betas(beta_spread, beta_max):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread / 2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2


@sj.simple
def aggregate_groups(C_0, C_1, C_2, A_0, A_1, A_2, MPC_0, MPC_1, MPC_2,
                     D_GREEN_0, D_GREEN_1, D_GREEN_2, D_SWITCH_0, D_SWITCH_1, D_SWITCH_2,
                     CE_B_0, CE_B_1, CE_B_2, CE_G_0, CE_G_1, CE_G_2,
                     CHF_0, CHF_1, CHF_2,
                     C_GREEN_0, C_GREEN_1, C_GREEN_2, C_BROWN_0, C_BROWN_1, C_BROWN_2,
                     CHF_GREEN_0, CHF_GREEN_1, CHF_GREEN_2,
                     CHF_BROWN_0, CHF_BROWN_1, CHF_BROWN_2,
                     C_SWITCH_0, C_SWITCH_1, C_SWITCH_2,
                     P_TIMES_C_0, P_TIMES_C_1, P_TIMES_C_2,
                     CHF_SWITCH_0, CHF_SWITCH_1, CHF_SWITCH_2,
                     LAB_INC_0, LAB_INC_1, LAB_INC_2, 
                     LAB_INC_GREEN_0, LAB_INC_GREEN_1, LAB_INC_GREEN_2, 
                     LAB_INC_BROWN_0, LAB_INC_BROWN_1, LAB_INC_BROWN_2, 
                     beta_0, beta_1, beta_2):
    C = (C_0 + C_1 + C_2) / 3
    A = (A_0 + A_1 + A_2) / 3
    MPC = (MPC_0 + MPC_1 + MPC_2) / 3
    D_GREEN = (D_GREEN_0 + D_GREEN_1 + D_GREEN_2) / 3
    D_SWITCH = (D_SWITCH_0 + D_SWITCH_1 + D_SWITCH_2) / 3
    CE_B = (CE_B_0 + CE_B_1 + CE_B_2) / 3
    CE_G = (CE_G_0 + CE_G_1 + CE_G_2) / 3
    CHF = (CHF_0 + CHF_1 + CHF_2) / 3
    C_GREEN = (C_GREEN_0 + C_GREEN_1 + C_GREEN_2) / 3
    C_BROWN = (C_BROWN_0 + C_BROWN_1 + C_BROWN_2) / 3
    CHF_GREEN = (CHF_GREEN_0 + CHF_GREEN_1 + CHF_GREEN_2) / 3
    CHF_BROWN = (CHF_BROWN_0 + CHF_BROWN_1 + CHF_BROWN_2) / 3
    C_SWITCH = (C_SWITCH_0 + C_SWITCH_1 + C_SWITCH_2) / 3
    beta = (beta_0 + beta_1 + beta_2) / 3
    P_times_C = (P_TIMES_C_0 + P_TIMES_C_1 + P_TIMES_C_2) / 3
    CHF_SWITCH = (CHF_SWITCH_0 + CHF_SWITCH_1 + CHF_SWITCH_2) / 3
    C_GREEN_PC = C_GREEN/D_GREEN # per capita cons of brown users
    C_BROWN_PC = C_BROWN/(1-D_GREEN) # per capita cons of brown users
    C_CHECK = C_GREEN + C_BROWN
    LAB_INC = (LAB_INC_0 + LAB_INC_1 + LAB_INC_2)/3
    LAB_INC_GREEN = (LAB_INC_GREEN_0 + LAB_INC_GREEN_1 + LAB_INC_GREEN_2)/3
    LAB_INC_BROWN = (LAB_INC_BROWN_0 + LAB_INC_BROWN_1 + LAB_INC_BROWN_2)/3
    
    return C, A, MPC, D_GREEN, D_SWITCH, CE_B, CE_G, CHF, C_GREEN, C_BROWN, C_GREEN_PC, C_BROWN_PC, CHF_GREEN, CHF_BROWN, C_SWITCH, P_times_C, CHF_SWITCH, C_CHECK, LAB_INC, LAB_INC_GREEN, LAB_INC_BROWN, beta


def hh_ha_durable(n_beta=3):
    hh_list = [hh_one.rename(suffix=f'_{i}')
                     .remap({x: f'{x}_{i}' for x in GROUP_VARS})
                     .remap({'beta_g': f'beta_{i}'})
               for i in range(n_beta)]
    return sj.create_model(hh_list + [group_betas, aggregate_groups], name='hh_ha_durable')
