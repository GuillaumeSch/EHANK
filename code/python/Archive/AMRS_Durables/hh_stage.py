"""E-HANK household as a StageBlock, built to nest Auclert-Rognlie-Straub exactly.

PHASE 1 (this file): no durable dimension. Stages are
    [prod (exogenous Pi), consav (EGM on a)]
which must reproduce the ARS `@sj.het` household to machine precision.

The durable dimension is added in phase 2 as two extra stages in front, so the
backward order becomes
    [dep (green -> brown breakdown), prod, durable (logit choice), consav].
"""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D


# =============================================================================
# 1. HETINPUTS  (verbatim from ARS hh_income / make_grids)
# =============================================================================
def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    return e_grid, Pi, a_grid


def hh_income(e_grid, atw_n, r, pEhh_P, cbarE, scale_w, markup_ss, a_grid, n,
              frisch, ghh_prefs, epsT, cE_ss_grid, insE, pE_P, pE_P_ss):
    Tf = - pEhh_P * cbarE * (atw_n * markup_ss) * scale_w - pEhh_P * cbarE * (1 - scale_w)
    Tfiscal = epsT + insE * (pE_P - pE_P_ss) * cE_ss_grid
    coh = (1 + r) * a_grid + atw_n * e_grid[:, np.newaxis] + Tf + Tfiscal
    n_ss = 1
    ghh = ghh_prefs * 1 / (1 + 1 / frisch) * (n ** (1 + 1 / frisch) - n_ss ** (1 + 1 / frisch))
    return coh, ghh


# =============================================================================
# 2. BACKWARD INIT
# =============================================================================
def hh_init(coh, r, eis):
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


# =============================================================================
# 3. STAGES
# =============================================================================
# Stage: idiosyncratic productivity (exogenous Markov on axis 0).
# Placed BEFORE consav in the list, so in backward order the expectation over
# next-period e is taken before the EGM step -- exactly what `Va_p` means in
# the monolithic HetBlock.
prod_stage = ExogenousMaker(markov_name='Pi', index=0, name='prod')


def consav(Va, a_grid, r, beta_g, eis, coh, ghh):
    """One EGM step. Identical algebra to the ARS backward function."""
    uc_nextgrid = beta_g * Va
    c_nextgrid = uc_nextgrid ** (-eis) + ghh
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * (c - ghh) ** (-1 / eis)
    return Va, a, c


def compute_weighted_mpc(c, a_grid, r, e_grid):
    """MPC out of wealth, symmetric differences, weighted by productivity."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpc[..., 1:-1] = (c[..., 2:] - c[..., :-2]) / (post_return[2:] - post_return[:-2])
    mpc[..., 0] = (c[..., 1] - c[..., 0]) / (post_return[1] - post_return[0])
    mpc[..., -1] = (c[..., -1] - c[..., -2]) / (post_return[-1] - post_return[-2])
    mpc = mpc * e_grid[:, np.newaxis]
    return mpc


consav_stage = Continuous1D(backward='Va', policy='a', f=consav, name='consav',
                            hetoutputs=[compute_weighted_mpc])


# =============================================================================
# 4. BLOCK ASSEMBLY + beta heterogeneity (ARS rename/remap idiom)
# =============================================================================
hh_one = StageBlock([prod_stage, consav_stage], name='hh',
                    backward_init=hh_init,
                    hetinputs=[make_grids, hh_income])

GROUP_VARS = ['C', 'A', 'MPC', 'cE_ss_grid']


def build_hh_group(n_beta=3):
    """Replicate the household into `n_beta` permanent discount-factor types."""
    hh_list = [hh_one.rename(suffix=f'_{i}')
                     .remap({x: f'{x}_{i}' for x in GROUP_VARS})
                     .remap({'beta_g': f'beta_{i}'})
               for i in range(n_beta)]
    return hh_list


@sj.simple
def group_betas(beta_spread, beta_max):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread / 2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2


@sj.simple
def aggregate_groups(C_0, C_1, C_2, A_0, A_1, A_2, MPC_0, MPC_1, MPC_2,
                     beta_0, beta_1, beta_2):
    C = (C_0 + C_1 + C_2) / 3
    A = (A_0 + A_1 + A_2) / 3
    MPC = (MPC_0 + MPC_1 + MPC_2) / 3
    beta = (beta_0 + beta_1 + beta_2) / 3
    return C, A, MPC, beta


def hh_ha_stage(n_beta=3):
    return sj.create_model(build_hh_group(n_beta) + [group_betas, aggregate_groups],
                           name='hh_ha_stage')
