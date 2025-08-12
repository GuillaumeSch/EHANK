# Standard libraries
from IPython.display import display, Math
# Numerical computing
import numpy as np
from numba import njit
# Sequence-Jacobian framework
from sequence_jacobian import grids, interpolate
#from sequence_jacobian.blocks.stage_block import StageBlock
import sequence_jacobian as sj
# Custom utilities
from SSJ_Fun.utils import make_d_grid, LogitChoiceDurables, ExogenousMaker, Continuous1D_Durables, StageBlockDurables
from Fun.my_funs import *

def make_strictly_decreasing(uc):
    uc_fixed = uc.copy()
    shape = uc.shape
    ndim = uc.ndim

    # Iterate over all indices except the last one
    it = np.nditer(uc[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index  # Tuple of all dimensions except the last
        row = uc[idx]  # This is a 1D array (the last axis)

        # Fix infinite values at the start
        if np.isinf(row[0]):
            first_finite = np.argmax(~np.isinf(row))
            if first_finite > 0:
                decrement = 1.0
                for k in range(first_finite - 1, -1, -1):
                    row[k] = row[k + 1] + decrement

        # Make strictly decreasing
        for k in range(1, row.shape[0]):
            if row[k] >= row[k - 1]:
                row[k] = row[k - 1] - 1e-8

        # Assign back to uc_fixed
        uc_fixed[idx] = row

        it.iternext()

    return uc_fixed


#%% Stage 1 - Productivity shock (Expected value function given initial state of individual prod. level e_)

#Initialize Stage 1a
prod_stage = ExogenousMaker(markov_name='e_markov', index=1, name='prod')

#Initialize Stage 1b
depreciation_stage = ExogenousMaker(markov_name='d_markov', index=0, name='durable')

#%% Stage 2 - Discrete choice (Labor participation)

#Initialize Stage 2
durables_stage = LogitChoiceDurables(value='V', backward='Va', index=0, name='durables',
                           taste_shock_scale='taste_shock')

#%% Stage 3 - Consumption-Savings Continuous Choice
#Discrete Choice - Endogenous Grid point Method. Performs single step of backward iteration.
def dcegm(V, Va, a_grid, disp_inc, adj_matrix, z_grid, r, T, beta, eis, shifters, bundle_price):
    """DC-EGM algorithm"""
    n_d = adj_matrix.shape[0] #Number of discrete choices
    # use all FOCs on endogenous grid
    W = beta * V                                                  # end-of-stage vfun
    W = np.stack([W] * n_d, axis=0)                               # Add first dimension to match the dimensions
    
    uc_endo = (beta * Va)[np.newaxis, ...] * bundle_price[..., np.newaxis ,np.newaxis, np.newaxis]     # FOC
    c_endo = uc_endo** (-eis)                                     # Euler equation
    a_endo = (c_endo * bundle_price[..., np.newaxis, np.newaxis, np.newaxis]
              + a_grid[np.newaxis, np.newaxis, np.newaxis, ...]
              + adj_matrix[..., np.newaxis,np.newaxis]
              - z_grid[np.newaxis, np.newaxis, ..., np.newaxis]
              - T[np.newaxis, np.newaxis, ..., np.newaxis]
              ) / (1 + r)     # budget constraint

    # Mark the presence of each durable 
    d_type = np.zeros_like(a_endo)
    for d in range(0, n_d):
        d_type[d, :, :, :] = d
    # Mark the presence of price of consumption bundle
    p_c_type = np.zeros_like(a_endo)
    for d in range(0, n_d):
        p_c_type[d, :, :, :] = bundle_price[d]

    # interpolate with upper envelope, enforce borrowing limit
    V, c, a = upperenv(W, a_endo, disp_inc, a_grid, d_type, p_c_type, eis, shifters)

    # update Va on exogenous grid
    uc = c ** (-1 / eis)                                          # Euler equation
    uc = make_strictly_decreasing(uc)                             # Correct for the infinite values.
    Va = (1 + r) * uc                                             # envelope condition

    return V, Va, a, c


#Simple wrapper to make it independent of the size of the state space. Temporarily collapse states associated with all other stages into a single axis.
def upperenv(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args):
    # collapse (d_tilde, d, z, a) into (b, a)
    shape = W.shape
    W = W.reshape((-1, shape[-1]))
    a_endo = a_endo.reshape((-1, shape[-1]))
    d_type = d_type.reshape((-1, shape[-1]))
    p_c_type = p_c_type.reshape((-1, shape[-1]))
    disp_inc = disp_inc.reshape((-1, shape[-1]))
    V, c, a = upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args)

    # report on (d_tilde, d, z, a)
    return V.reshape(shape), c.reshape(shape), a.reshape(shape)


#Core upper envelope step:
# Consider every segment of the endogenous grid $(a_{j}^{endo}, a_{j+1}^{endo})$ and find all the exogenous gridpoints $a^{grid}_i$ that fall into that segment.
# Interpolate there to get a candidate solution $a_i$.
# Since the endogenous grid is non-monotonic, the same point $a^{grid}_i$ may be bracketed by another segment $(a_{\tilde j}^{endo}, a_{\tilde j+1}^{endo}).$
# When this happens, we keep the solution that gives higher value.
@njit
def upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args):
    """Interpolate value function and consumption to exogenous grid."""
    n_b, n_a = W.shape
    a = np.zeros_like(W)
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        d = int(d_type[ib,0])
        p_c = p_c_type[ib,0]
        # loop over segments of endogenous asset grid from EGM (not necessarily increasing)
        for ja in range(n_a - 1):
            a_low, a_high = a_endo[ib, ja], a_endo[ib, ja + 1]
            W_low, W_high = W[ib, ja], W[ib, ja + 1]
            ap_low, ap_high = a_grid[ja], a_grid[ja + 1]

           # loop over exogenous asset grid (increasing)
            for ia in range(n_a):
                acur = a_grid[ia]
                coh_cur = disp_inc[ib, ia]

                interp = (a_low <= acur <= a_high)
                extrap = (ja == n_a - 2) and (acur > a_endo[ib, n_a - 1])

                # exploit that a_grid is increasing
                if (a_high < acur < a_endo[ib, n_a - 1]):
                    break

                if interp or extrap:
                    W0 = interpolate.interpolate_point(acur, a_low, a_high, W_low, W_high)
                    a0 = interpolate.interpolate_point(acur, a_low, a_high, ap_low, ap_high)
                    c0 = (coh_cur - a0)/p_c
                    V0 = util(c0, d, *args) + W0

                    # upper envelope, update if new is better
                    if V0 > V[ib, ia]:
                        a[ib, ia] = a0
                        c[ib, ia] = c0
                        V[ib, ia] = V0

        # Enforce borrowing constraint
        ia = 0
        while ia < n_a and a_grid[ia] <= a_endo[ib, 0]:
            a[ib, ia] = a_grid[0]
            c[ib, ia] = max(0.0001,disp_inc[ib, ia]/p_c) # Correct for negative values. Replace by small consumption (Unlikely to choose this consumption)
            V[ib, ia] = util(c[ib, ia], d, *args) + W[ib, 0]
            ia += 1

    return V, c, a

# %% Utility function
@njit
def util(c, d, eis, shifters):
    """
    General utility function for arbitrary discrete states.

    Parameters:
    - c: consumption (scalar)
    - d: discrete state index (integer)
    - eis: elasticity of intertemporal substitution
    - shifters: 1D array of utility shifters per d-state
    """
    # Basic bounds check (without exception)
    #if d < 0 or d >= shifters.shape[0]:
    #    return -1e10  # or some other penalizing value instead of raising an error

    if eis == 1.0:
        u = np.log(c) + shifters[d]
    else:
        u = c ** (1 - 1 / eis) / (1 - 1 / eis) + shifters[d]

    return u


#Report the aggregate demand for d
def D_demand(c):
    shape = c.shape
    D = shape[0]
    dd_tilde_list = []
    dd_list = []
    for d in range(D):
        dd_tilde = np.zeros(shape, dtype=c.dtype)
        dd = np.zeros(shape, dtype=c.dtype)
        dd_tilde[d, ...] = 1
        dd[:, d, ...] = 1
        dd_tilde_list.append(dd_tilde)
        dd_list.append(dd)
    # Dynamically assign to variables in local scope
    out_vars = []
    for i in range(D):
        globals()[f'dd_tilde_{i}'] = dd_tilde_list[i]
        globals()[f'dd_{i}'] = dd_list[i]
        out_vars.append(dd_tilde_list[i])
    for i in range(D):
        out_vars.append(dd_list[i])

    d_t_N, d_N, d_t_BN, d_BN, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GO, d_GO = (dd_tilde_0, dd_0, dd_tilde_1, dd_1, dd_tilde_2, dd_2, dd_tilde_3, dd_3, dd_tilde_4, dd_4)
    #d_t_N, d_N, d_t_BN, d_BN, d_t_BM, d_BM, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GM, d_GM, d_t_GO, d_GO = (dd_tilde_0, dd_0, dd_tilde_1, dd_1, dd_tilde_2, dd_2, dd_tilde_3, dd_3, dd_tilde_4, dd_4, dd_tilde_5, dd_5, dd_tilde_6, dd_6)

    d_B = d_BN + d_BO
    d_G = d_GN + d_GO
#return d_t_N, d_N, d_t_BN, d_BN, d_t_BM, d_BM, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GM, d_GM, d_t_GO, d_GO
    return d_t_N, d_N, d_t_BN, d_BN, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GO, d_GO, d_B, d_G
#return dd_tilde_0, dd_0, dd_tilde_1, dd_1, dd_tilde_2, dd_2, dd_tilde_3, dd_3, dd_tilde_4, dd_4


def compute_distr(c):
    distr = np.ones_like(c)
    return distr

#Initialize Stage 3
consav_stage = Continuous1D_Durables(backward=['V', 'Va'], policy='a', f=dcegm,
                            name='consav', hetoutputs=[D_demand, compute_distr])

# %% Other basic necessary functions
# hh_init: function that constructs the initial guess for backward variables
def hh_init(disp_inc, a_grid, eis, shifters):
    V = util(disp_inc-np.min(disp_inc)+1, 0, eis, shifters)         #Avoid strange behaviour due to negative values. Not too important as only for first guess.
    V = np.mean(V, axis=0)                                          #Get rid of first dimension
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (a_grid[1] - a_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (a_grid[-1] - a_grid[-2])
    return V, Va

#construct Markov process for productivity, for depreciation of durables and the assets grid
def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, n_b, n_g, lifetime_b, lifetime_g):
    e_grid, e_dist, e_markov = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = grids.agrid(max_a, n_a, min_a)
    d_grid, d_markov, d_grid_name = make_d_grid(n_b, n_g, lifetime_b, lifetime_g)
    return e_grid, e_dist, e_markov, a_grid, d_grid, d_markov, d_grid_name

#def income_grid(e_grid, tau, w, N):
def income_grid(e_grid, Z):
    #z_grid = (1 - tau) * w * N * e_grid
    z_grid = Z * e_grid
    return z_grid

def transfers(e_dist, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    #tax_rule, div_rule = e_grid, e_grid
    tax_rule, div_rule = np.ones_like(e_grid), np.ones_like(e_grid)               #Lump-Sum
    div = Div / np.sum(e_dist * div_rule) * div_rule
    tax = Tax / np.sum(e_dist * tax_rule) * tax_rule
    T = div - tax
    return T

#Construct the adjustment costs matrix between durables
def adj_costs(p_d, chi):
    adj_matrix = p_d[:, None] - (1 - chi) * p_d
    np.fill_diagonal(adj_matrix, 0)                            # set diagonal to 0 (no cost if no switching)
    return adj_matrix

#Define the disposable income
def disp_inc_f(a_grid, z_grid, T, r, adj_matrix):                 #Disposable income for consumption and assets after buying the durable good
    # Disposable income is:
    # asset income          + labor income         - durable adjustment cost
    disp_inc = (
        (1 + r) * a_grid[np.newaxis, np.newaxis, np.newaxis, :]           # asset income
        + z_grid[np.newaxis, np.newaxis, ..., np.newaxis]                 # labor income
        + T[np.newaxis, np.newaxis, ..., np.newaxis]                      # Transfers
        - adj_matrix[..., np.newaxis, np.newaxis]                         # adjustment costs
    )                                                         # on (nd, nd, e, a)
    return disp_inc

#Construct the utility shifter for durables

def make_shifters(n_b, n_g, gamma_b, gamma_g, dep_util_frac_b, dep_util_frac_g):
    dep_rate_b = 1 - (dep_util_frac_b) ** (1 / (n_b - 1))        # Depreciation rate for good b
    vintages_b = np.arange(n_b)
    gammas_b_vector = gamma_b * (1 - dep_rate_b) ** vintages_b
    dep_rate_g = 1 - (dep_util_frac_g) ** (1 / (n_g - 1))        # Depreciation rate for good g
    vintages_g = np.arange(n_g)
    gammas_g_vector = gamma_g * (1 - dep_rate_g) ** vintages_g
    # Combine
    shifters = np.array([0.0] + list(gammas_b_vector) + list(gammas_g_vector))
    return shifters

def make_prices_durables(p_b, dep_frac_b, n_b, p_g, dep_frac_g, n_g):
    dep_rate_b = 1 - (dep_frac_b) ** (1 / (n_b - 1))        # Depreciation rate for good b
    vintages_b = np.arange(n_b)
    p_b_vector = p_b * (1 - dep_rate_b) ** vintages_b
    dep_rate_g = 1 - (dep_frac_g) ** (1 / (n_g - 1))        # Depreciation rate for good g
    vintages_g = np.arange(n_g)
    p_g_vector = p_g * (1 - dep_rate_g) ** vintages_g
    # Combine
    p_d = np.array([0.0] + list(p_b_vector) + list(p_g_vector))
    return p_d

#Price of the household consumption bundle
def make_consu_bundle_price(p_core, n_b, p_e_b, n_g, p_e_g, tau_b, tau_g, xi, nu):
    p_e_total_b = np.ones(n_b) * (1 + tau_b) * p_e_b
    p_e_total_g = np.ones(n_g) * (1 + tau_g) * p_e_g
    p_e = np.concatenate([[0], p_e_total_b, p_e_total_g])
    if nu != 1:
        bundle_price = (xi * p_core**(1-nu) + (1-xi) * p_e**(1-nu))**(1/(1-nu))
    else:
        bundle_price = p_core**xi * p_e**(1-xi)
    return bundle_price


#%% Assemble the HH block (staged block)
hh = StageBlockDurables([depreciation_stage, prod_stage, durables_stage, consav_stage], name='hh',
                backward_init=hh_init,
                hetinputs=[make_grids, income_grid, transfers, adj_costs, 
                           disp_inc_f, make_shifters, make_prices_durables, 
                           make_consu_bundle_price])