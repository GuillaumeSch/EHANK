#%%
# Standard libraries
from dataclasses import dataclass
from typing import Tuple, Callable

# Numerical computing
import numpy as np
from numba import njit

# Sequence-Jacobian framework
from sequence_jacobian import grids, interpolate
from sequence_jacobian.blocks.support.stages import ExogenousMaker

# Custom utilities
from SSJ_Fun.utils import make_d_grid_simple, LogitChoiceDurables, Continuous1D_Durables, StageBlockDurables, ExogenousMaker
from Fun.my_funs import *

#%%--------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
#1a. Productivity stage
prod_stage = ExogenousMaker(markov_name='e_markov', index=1, name='prod')
#1b. Durable depreciation stage
depreciation_stage = ExogenousMaker(markov_name='d_markov', index=0, name='durable')
#2. Discrete choice over durables stage
durables_stage = LogitChoiceDurables(value='V', backward='Va', index=0, name='durables', taste_shock_scale='taste_shock')

#%%--------------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------------
#@njit
def util(c, gamma):
    if gamma == 1.0:
        u = np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
    return u

#%% Stage 3 - Consumption-Savings Continuous Choice
#Discrete Choice - Endogenous Grid point Method. Performs single step of backward iteration.
def dcegm(V, Va, a_grid, disp_inc, adj_matrix, z_grid, r, T, beta, gamma, p_bundle):
    """DC-EGM algorithm"""
    n_d = adj_matrix.shape[0] #Number of discrete choices
    # use all FOCs on endogenous grid
    W = beta * V                                                  # end-of-stage vfun
    W = np.stack([W] * n_d, axis=0)                               # Add first dimension to match the dimensions
    #uc_endo = (beta * Va)[np.newaxis, ...] * p_bundle[..., np.newaxis ,np.newaxis, np.newaxis]     # FOC
    uc_endo = (beta * Va)[np.newaxis, ...]
    c_endo = uc_endo** (-1/gamma)                                    
    
    a_endo = (
        p_bundle[..., np.newaxis, np.newaxis, np.newaxis] * c_endo
        + a_grid[np.newaxis, np.newaxis, np.newaxis, ...]
        + adj_matrix[..., np.newaxis, np.newaxis]
        - z_grid[np.newaxis, np.newaxis, ..., np.newaxis]
        - T[np.newaxis, np.newaxis, ..., np.newaxis]
    ) / (1 + r)

    # Prepare indices for upper envelope
    d_type = np.arange(n_d)[:, None, None, None] * np.ones_like(a_endo)
    p_c_type = p_bundle[:, None, None, None] * np.ones_like(a_endo)

    # interpolate with upper envelope, enforce borrowing limit
    V, c, a = upperenv(W, a_endo, disp_inc, a_grid, d_type, p_c_type, gamma)

    # update Va on exogenous grid
    uc = np.maximum(1e-8, c) ** (-gamma) # Marginal Utility of Consu
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
#@njit
def upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args):
    """Interpolate value function and consumption to exogenous grid."""
    n_b, n_a = W.shape
    a = np.zeros_like(W)
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        d, p_c = int(d_type[ib,0]), p_c_type[ib,0]
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
                    #c0 = (coh_cur - a0)/p_c
                    c0 = max(1e-5, (coh_cur - a0)/p_c) # Correct for negative values. Replace by small consumption (Unlikely to choose this consumption)
                    V0 = util(c0, *args) + W0

                    # upper envelope, update if new is better
                    if V0 > V[ib, ia]:
                        a[ib, ia] = a0
                        c[ib, ia] = c0
                        V[ib, ia] = V0

        # Enforce borrowing constraint
        ia = 0
        while ia < n_a and a_grid[ia] <= a_endo[ib, 0]:
            a[ib, ia] = a_grid[0]
            c[ib, ia] = max(1e-5,disp_inc[ib, ia]/p_c) # Correct for negative values. Replace by small consumption (Unlikely to choose this consumption)
            V[ib, ia] = util(c[ib, ia], *args) + W[ib, 0]
            ia += 1

    return V, c, a


#%%--------------------------------------------------------------------------
# Hetoutputs for the Continous Choice stage. 
# Can use vectors as input, but not as output.
# Will aggregate all the outputs across the distribution.
# ---------------------------------------------------------------------------

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
    d_t_B, d_B, d_t_G, d_G = (dd_tilde_0, dd_0, dd_tilde_1, dd_1)
    return d_t_B, d_B, d_t_G, d_G

#Report the decomposition of the consumption bundle into core and energy, and the tax paid on energy consumption. Also report the decomposition of energy consumption into brown and green.
def decomposition_consu_bundle(c, p_core, p_bundle, p_e, nu, xi, tau_vec):
    c_core = xi * (p_core/p_bundle[...,np.newaxis,np.newaxis, np.newaxis])**(-nu)*c
    mask = p_e == 0
    c_E = np.zeros_like(c)
    non_zero_mask = ~mask
    c_E[non_zero_mask] = (1-xi) * ((1 + tau_vec[non_zero_mask,np.newaxis,np.newaxis,np.newaxis]) * p_e[non_zero_mask,np.newaxis,np.newaxis,np.newaxis]/
                          p_bundle[non_zero_mask,np.newaxis,np.newaxis,np.newaxis])**(-nu)*c[non_zero_mask]
    
    c_E_b, c_E_g = [np.zeros_like(c_E) for _ in range(2)]
    c_E_b[0,...], c_E_g[1,...] = c_E[0,...], c_E[1,...]
        
    t_E = c_E * tau_vec[...,np.newaxis, np.newaxis, np.newaxis] * p_e[...,np.newaxis, np.newaxis, np.newaxis]

    return c_core, c_E, c_E_b, c_E_g, t_E



#Initialize Stage 3
consav_stage = Continuous1D_Durables(backward=['V', 'Va'], policy='a', f=dcegm,
                            name='consav', hetoutputs=[D_demand, decomposition_consu_bundle])

          
#%%--------------------------------------------------------------------------
# Hetinputs (grid construction, income, transfers, prices...)
# ---------------------------------------------------------------------------
def hh_init(disp_inc, a_grid, gamma):
    V = util(disp_inc-np.min(disp_inc)+1,gamma)         #Avoid strange behaviour due to negative values. Not too important as only for first guess.
    V = np.mean(V, axis=0)                                          #Get rid of first dimension
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (a_grid[1] - a_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (a_grid[-1] - a_grid[-2])
    return V, Va

#construct Markov process for productivity, for depreciation of durables and the assets grid
def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, delta_g):
    e_grid, e_dist, e_markov = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = grids.agrid(max_a, n_a, min_a)
    d_grid, d_markov, d_grid_name = make_d_grid_simple(delta_g)
    return e_grid, e_dist, e_markov, a_grid, d_grid, d_markov, d_grid_name

def income_grid(e_grid, w, N):
    z_grid = w * N * e_grid
    return z_grid

def create_vectors(tau_b, tau_g, 
                   p_e_b, p_e_g,
                   nu, xi, p_core):
    #Quantities of durables vector
    d = np.array([1,1])
    #Prices of energy vector
    p_e = np.array([p_e_b, p_e_g])
    #Tax vector and tax-adjusted price of energy
    tau_vec = np.array([tau_b, tau_g])
    p_tilde_e = (1 + tau_vec) * p_e
    #Price of the household consumption bundle
    if nu != 1:
        p_bundle = (xi * p_core**(1-nu) + (1-xi) *p_tilde_e**(1-nu))**(1/(1-nu))
    else:
        p_bundle = p_core**xi * p_tilde_e**(1-xi)
    
    return tau_vec, p_e, d, p_tilde_e, p_bundle

def transfers(e_dist, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    #tax_rule, div_rule = e_grid, e_grid
    tax_rule, div_rule = np.ones_like(e_grid), np.ones_like(e_grid)               #Lump-Sum
    div = Div / np.sum(e_dist * div_rule) * div_rule
    tax = Tax / np.sum(e_dist * tax_rule) * tax_rule
    T = div - tax
    return T

#Construct the adjustment costs to go from Brown to Green
def adj_costs(psi_g):
    adj_matrix = np.array([
        [0.0,     0.0],  # brown -> green
        [psi_g,   0.0]     # green -> anything
    ])
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

#%% Assemble the HH block (staged block)
hh = StageBlockDurables([depreciation_stage, prod_stage, durables_stage, consav_stage], name='hh',
                backward_init=hh_init,
                hetinputs=[make_grids, income_grid, transfers, adj_costs, 
                           disp_inc_f, create_vectors])