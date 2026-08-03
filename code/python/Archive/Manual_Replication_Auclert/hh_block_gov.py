"""
Household block WITH government and a price-facing energy subsidy (Auclert et al. 2023,
Section 5), single-beta.

Cash-on-hand components (market-real, deflated by the market CPI P):
    Z * e_i           after-tax labor income
    tE * xfer_base    targeted transfer, indexed to ss energy use (xfer_base = alpha_E c_ss)
    T_unt             untargeted lump-sum transfer

Energy subsidy enters as a cost-of-living wedge p_c = P_hh / P <= 1 (P_hh = household price
index at the subsidized energy price pE_hh). One unit of the consumption aggregate costs p_c
in market-real terms, so the EGM carries p_c through the FOC and budget:
    u'(c)/p_c = beta (1+r) E[Va']            (Euler with price wedge)
    a' = (1+r)a + income - p_c * c            (budget)
When tau_E = 0, p_c == 1 identically (the CPI identity), so this reduces to the baseline.
"""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import het


def hh_init(a_grid, e_grid, r, Z, eis, p_c):
    coh = (1 + r) * a_grid[np.newaxis, :] + Z * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis) / p_c
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def household(Va_p, a_grid, e_grid, r, Z, beta, eis, tE, T_unt, xfer_base, p_c):
    c_nextgrid = (p_c * beta * Va_p) ** (-eis)
    coh = ((1 + r) * a_grid[np.newaxis, :] + Z * e_grid[:, np.newaxis]
           + tE * xfer_base + T_unt)
    a = sj.interpolate.interpolate_y(p_c * c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = (coh - a) / p_c
    Va = (1 + r) * c ** (-1 / eis) / p_c
    return Va, a, c


def grids(rho_e, sigma_e, nS, amax, nA):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho=rho_e, sigma=sigma_e, N=nS)
    a_grid = sj.grids.agrid(amax=amax, n=nA)
    return e_grid, Pi, a_grid
