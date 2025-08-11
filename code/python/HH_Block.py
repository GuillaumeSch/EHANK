#%% Import packages
#import time
#start = time.time()

# Standard libraries
import inspect
from IPython.display import display, Math
# Numerical computing
import numpy as np
from numba import njit
from scipy.interpolate import interp1d, griddata
from copy import deepcopy
# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
# Sequence-Jacobian framework
from sequence_jacobian import grids, interpolate
#from sequence_jacobian.blocks.stage_block import StageBlock
import sequence_jacobian as sj
# Custom utilities
from SSJ_Fun.utils import make_d_grid, LogitChoiceDurables, ExogenousMaker, Continuous1D_Durables, StageBlockDurables
from Fun.my_funs import *

#%% Interactive plot
#%matplotlib qt

#%% Some useful functions for debugging

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
#`value`: name of value function, SSJ has to know which object to apply the logsum formula to
#`backward`: names of other variables that have to be propagated backward, typically this is partial value function needed for EGM in continuous choice stage
#`index`: axis of correspoinding state
#`name`: name of stage
#`taste_shock_scale`: name of $\sigma_\varepsilon$ parameter, needed for all formulas
#`f`: (optional) function that implements additive utility cost on expanded state $(n| n_-, z, a_-)$. This is useful to implement costs that depend on origin as well as destination $(n|n_-)$. Setting some costs to infinity implements constraints on discrete choice (more on this below).

durables_stage = LogitChoiceDurables(value='V', backward='Va', index=0, name='durables',
                           taste_shock_scale='taste_shock')

#%% Stage 3 - Consumption-Savings Continuous Choice
#Discrete Choice - Endogenous Grid point Method. Performs single step of backward iteration.
def dcegm(V, Va, a_grid, disp_inc, adj_matrix, z_grid, r, T, beta, eis, shifters):
    """DC-EGM algorithm"""
    n_d = adj_matrix.shape[0] #Number of discrete choices
    # use all FOCs on endogenous grid
    W = beta * V                                                  # end-of-stage vfun
    W = np.stack([W] * n_d, axis=0)                               # Add first dimension to match the dimensions
    uc_endo = beta * Va                                           # envelope condition
    c_endo = uc_endo** (-eis)                                     # Euler equation
    a_endo = (c_endo[np.newaxis, ...]
              + a_grid[np.newaxis, np.newaxis, np.newaxis, ...]
              + adj_matrix[..., np.newaxis,np.newaxis]
              - z_grid[np.newaxis, np.newaxis, ..., np.newaxis]
              - T[np.newaxis, np.newaxis, ..., np.newaxis]
              ) / (1 + r)     # budget constraint

    # Mark the presence of each durable
    d_type = np.zeros_like(a_endo)
    for d in range(0, n_d):
        d_type[d, :, :, :] = d

    # interpolate with upper envelope, enforce borrowing limit
    V, c, a = upperenv(W, a_endo, disp_inc, a_grid, d_type, eis, shifters)

    # update Va on exogenous grid
    uc = c ** (-1 / eis)                                          # Euler equation
    uc = make_strictly_decreasing(uc)                             # Correct for the infinite values.
    Va = (1 + r) * uc                                             # envelope condition

    return V, Va, a, c



#Simple wrapper to make it independent of the size of the state space. Temporarily collapse states associated with all other stages into a single axis.
def upperenv(W, a_endo, disp_inc, a_grid, d_type, *args):
    # collapse (d_tilde, d, z, a) into (b, a)
    shape = W.shape
    W = W.reshape((-1, shape[-1]))
    a_endo = a_endo.reshape((-1, shape[-1]))
    d_type = d_type.reshape((-1, shape[-1]))
    disp_inc = disp_inc.reshape((-1, shape[-1]))
    V, c, a = upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, *args)

    # report on (d_tilde, d, z, a)
    return V.reshape(shape), c.reshape(shape), a.reshape(shape)


#Core upper envelope step:
# Consider every segment of the endogenous grid $(a_{j}^{endo}, a_{j+1}^{endo})$ and find all the exogenous gridpoints $a^{grid}_i$ that fall into that segment.
# Interpolate there to get a candidate solution $a_i$.
# Since the endogenous grid is non-monotonic, the same point $a^{grid}_i$ may be bracketed by another segment $(a_{\tilde j}^{endo}, a_{\tilde j+1}^{endo}).$
# When this happens, we keep the solution that gives higher value.
@njit
def upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, *args):
    """Interpolate value function and consumption to exogenous grid."""
    n_b, n_a = W.shape
    a = np.zeros_like(W)
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        #d = min(ib * 2 // n_b, 2 - 1)
        d = int(d_type[ib,0])
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
                    c0 = coh_cur - a0
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
            c[ib, ia] = max(0.0001,disp_inc[ib, ia]) # Correct for negative values. Replace by small consumption (Unlikely to choose this consumption)
            V[ib, ia] = util(c[ib, ia], d, *args) + W[ib, 0]
            ia += 1

    return V, c, a

# %% Utilitz function
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
    V = (V[0,:,:,:] + V[1,:,:,:])/2                                 #Get rid of first dimension
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


#%% Assemble the HH block (staged block)
hh = StageBlockDurables([depreciation_stage, prod_stage, durables_stage, consav_stage], name='hh',
                backward_init=hh_init,
                hetinputs=[make_grids, income_grid, transfers, adj_costs, disp_inc_f, make_shifters, make_prices_durables])

print(hh)
print(f"Inputs: {hh.inputs}")
print(f"Outputs: {hh.outputs}")

#%%
# -------------------------------
# --Solving the baseline hh block --
# -------------------------------

#%% Calibration
# === Calibration dictionary ===
cali = {}
cali["baseline"] = {
    # Preferences and taste shocks
    "taste_shock": 1e-1,       # Idiosyncratic taste shock
    "vphi": 0.0,               # Value function penalty parameter
    "beta": 0.97,              # Discount factor
    "eis": 0.5,                # Elasticity of intertemporal substitution
    "r": 0.02 / 4,             # Interest rate (quarterly)
    # Productivity process
    "rho_e": 0.95,             # Persistence of productivity shocks
    "sd_e": 0.5,               # Std. deviation of productivity shocks
    "n_e": 5,                  # Number of productivity grid points
    # Asset grid
    "min_a": 0.0,              # Minimum asset level
    "max_a": 100,              # Maximum asset level
    "n_a": 20,                 # Number of asset grid points
    # Labor market
    #"w": 1.0,                  # Wage level
    "N": 1.0,                  # Labor supply
    "tau":0,                   # Labor income tax
    # Durable goods
    "p_b": 0.80,               # Initial price of brown durable
    "dep_frac_b": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_b": 2,                  # Number of brown vintages
    "p_g": 0.90,               # Initial price of green durable
    "dep_frac_g": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_g": 2,                  # Number of green vintages
    #"n_d": 1 + n_b + n_g,      # Total durable states
    "chi": 0.5,                # Resale loss (fraction)
    "gamma_b": 1.0,            # Utility from brown durable
    "dep_util_frac_b": 1,    # Depreciation utility brown (Fraction of oldest vintage relative to newest)
    "gamma_g": 1.2,            # Utility from green durable
    "dep_util_frac_g": 1,    # Depreciation utility green (Fraction of oldest vintage relative to newest)
    "lifetime_b": 60,          # Average lifetime of brown durables (quarters)
    "lifetime_g": 60,          # Average lifetime of green durables (quarters)
    # Firms
    "alpha": 1,                # Share of labor in prod. function
    "Div": 0,                  # Dividends from firms
    "Tax": 0.5,                # Total tax
    #Government
    #"Y" : 1,                   # Output
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in cali["baseline"].items():
    globals()[k] = v


# %% Only useful for debugging
# e_grid, e_dist, e_markov, a_grid, d_grid, d_markov, d_grid_name = make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, n_b, n_g, lifetime_b, lifetime_g)
# z_grid = income_grid(e_grid, tau, w, N)
# T = transfers(e_dist, Div, Tax, e_grid)
# adj_matrix = adj_costs(p_d, chi)
# disp_inc = disp_inc_f(a_grid, z_grid, T, r, adj_matrix)
# shifters = make_shifters(n_b, n_g, gamma_b, gamma_g)

# V, Va = hh_init(disp_inc, a_grid, eis, shifters)

# #%% Baseline model

# ss = dict()
# ss['baseline'] = hh.steady_state(cali['baseline'])
# print(ss['baseline']['A'])
# print('Proportion of people with a brown car at the end of the period (choice variable)',ss['baseline']['DD_TILDE_1'])
# print('Proportion of people with a green car at the end of the period (choice variable)',ss['baseline']['DD_TILDE_2'])
# print('Proportion of people with a brown car at the beginning of the period (state variable)',ss['baseline']['DD_1'])
# print('Proportion of people with a green car at the beginning of the period (state variable)',ss['baseline']['DD_2'])
# print('Ratio of DD_1/DD_TILDE_1: ',ss['baseline']['DD_1'] / ss['baseline']['DD_TILDE_1'])
# print('Ratio of DD_2/DD_TILDE_2: ',ss['baseline']['DD_2'] / ss['baseline']['DD_TILDE_2'])
# print(ss['baseline']['C'])
# #%% Policy functions
# policy_functions(ss, amax=150, d_tilde_list=ss['baseline'].internals['hh']['d_grid'] ,d_list = [0],ie_list=[0], figsize=0.8, models = ['baseline'])

# #%% Comparative statics of SS - 2d
# results = analyze_steady_state('chi', np.linspace(0.1, 0.5, 3), cali, hh, n_d)

# #%% Comparative statics of SS - 3d
# CS_dep_rate_chi_2 = analyze_steady_state_3d(
#     param1='dep_rate',
#     values1=np.linspace(0.05, 0.75, 5),
#     param2='chi',
#     values2=np.linspace(0.1, 0.8, 5),
#     cali=cali,
#     hh=hh,
#     n_d=n_d
#     )

# #%% Comparative statics of SS - 3d
# CS_lifetime = analyze_steady_state_3d(
#     param1='lifetime_b',
#     values1=np.linspace(30, 90, 5),
#     param2='lifetime_g',
#     values2=np.linspace(30, 90, 5),
#     cali=cali,
#     hh=hh,
#     n_d=n_d
#     )

# #%% Comparative statics of SS - 3d
# #Would be nice to analzye how it changes by changing gamma_g and gamma_b
# CS_gammas = analyze_steady_state_3d(
#     param1='gamma_b',
#     values1=np.linspace(0.9, 1.1, 5),
#     param2='gamma_g',
#     values2=np.linspace(0.9, 1.1, 5),
#     cali=cali,
#     hh=hh,
#     n_d=n_d
#     )
# # %%



#%% Add other blocks

@sj.simple
def fiscal(B, r, G, Y):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    Z = Y - Tax
    deficit = G - Tax
    return Tax, deficit, Z


@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt

@sj.simple
def prod(N, alpha):
    Y = N**alpha
    w = N**(alpha-1)
    return Y, w

#%% Create the model
ha = sj.create_model([hh, fiscal, mkt_clearing, prod], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))
# %% Evalaute model with basic calibration
cali['no_ss'] = deepcopy(cali['baseline'])
cali['no_ss']['r'] = cali['baseline']['r'] + 0
cali['no_ss']['G'] = cali['baseline']['G'] + 0.0
cali['no_ss']['beta'] = cali['baseline']['beta'] + 0.00
cali['no_ss']['B'] = cali['baseline']['B'] + 0
cali['no_ss']['N'] = cali['baseline']['N'] + 0.0




no_ss = ha.steady_state(cali['no_ss'])
# Print the result
print("Evaluating steady state with arbitrary calibration (no equilibrium solving):")
print(f"  Given beta = {cali['no_ss']['beta']}")
print(f"  Given r    = {cali['no_ss']['r']}")
print(f"  Given G    = {cali['no_ss']['G']}")
print(f"  Given B    = {cali['no_ss']['B']}")
print("Resulting market clearing residuals:")
print(f"  Goods market:  {np.round(no_ss['goods_mkt'], 5)}")
print(f"  Asset market:  {np.round(no_ss['asset_mkt'], 5)}")
print(f"  Tax:  {np.round(no_ss['Tax'], 5)}")






#%% Find the values for SS
#unknowns_ss = {'beta': 0.97, 'G': 0.3}
#unknowns_ss = {'r':0.005, 'G': 0.3}
#unknowns_ss = {'r':0.005, 'beta': 0.97}
unknowns_ss = {'beta':0.91}
targets_ss = {'asset_mkt'}

ss = ha.solve_steady_state(cali['baseline'], unknowns_ss, targets_ss, solver='hybr')
print(f"To attain SS, we need beta={np.round(ss['beta'],4)}")
print(f"To attain SS, we need Y={np.round(ss['Y'],4)}")

print(f"Check: Goods market clearing: {np.round(ss['goods_mkt'],5)}")
print(f"Check: Assets market clearing: {np.round(ss['asset_mkt'],5)}")



#%% Calibrate to attain the empirical fractions of cars 

targets_ss = {'asset_mkt': 0.,
    'D_N': 1-0.075 - 0.55,
    'D_G': 0.075,
    'D_B':0.55,
    #'D_BN':0.007,
    }  # <-- with a dict rather than a list, we can specify specific targets for output variables

unknowns_ss = {
    'beta': (0.80, 0.881, 0.95),
    'p_b': (0.01, 0.273, 10),
    'p_g': (0.01, 0.9, 10),
    'gamma_g': (0,1.243,100),
    #'dep_util_frac_b': (0.1,0.99,1)
}

ss_DD = ha.solve_steady_state(cali['baseline'], unknowns_ss, targets_ss, solver = 'broyden_custom')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

#%%
cali['ss_DD'] = ha.steady_state(ss_DD)
cali['ss_DD_mod'] = deepcopy(cali['ss_DD'])
cali['ss_DD_mod']['dep_util_frac_b'] = cali['ss_DD_mod']['dep_util_frac_b']*0.50
#cali['ss_DD_mod']['gamma_b'] = cali['ss_DD_mod']['gamma_b']*1.50


print('Original model:')
display_ss_durables(ha.steady_state(cali['ss_DD']))
print('Modified model:')
display_ss_durables(ha.steady_state(cali['ss_DD_mod']))


#%%
for key, value in cali['ss_DD'].items():
    if 'D_' in key:
        print(key, ":", np.round(value, 2))


#%% Use the ss
cali['ss'] = ha.steady_state(ss)
#%%
for key, value in ss.items():
    print(key, ":", np.round(value,2))

# %%
T = 300
#breakpoint()
J_ha = hh.jacobian(ss, inputs=['r'], T=T)

# %%
s_to_plot = [0, 50, 100, 150]
for s in s_to_plot:
   plt.plot(J_ha['A']['r'][:, s], label =f's={s}')
plt.legend()
plt.show()

#%%
# %% IRFs
T = 300  # <-- the length of the IRF
rho_r = 0.8
eR = 0.01
rho_B = 0.8
eB = 0.01*0
dr = eR * rho_r ** np.arange(T)
dB = eB * rho_B ** np.arange(T)
shocks = {"r": dr, "B": dB}
unknowns_td = ['N']
targets_td = ["asset_mkt"]
irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)
irfs_alt = ha.solve_impulse_linear(ss_DD, unknowns_td, targets_td, shocks)
show_irfs([irfs, irfs_alt], ["N","w","C","Y", "A", "goods_mkt", "asset_mkt"],  labels=["Default Calib","Calibrated"], figsize=(18,3))
show_irfs([irfs, irfs_alt], ["D_N","D_BO","D_BN","D_GO","D_GN"],  labels=["Default Calib","Calibrated"], figsize=(18,3))


#%% Compute and plot directly
plot_linear_irfs(
    shocks_list=['r'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"goods_mkt"],
    ha=ha,
    ss=ss,
    outputs=["N", "G", "Tax","r", "B", "w", "C", "Y", "A", "goods_mkt", "asset_mkt"]
)

#%%
plot_linear_irfs(
    shocks_list=['tau'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"goods_mkt"],
    ha=ha,
    ss=ss,
    outputs=["N", "G", "Tax","r", "B", "w", "C", "Y", "A", "goods_mkt", "asset_mkt"]
)

# %% IRFs
T = 300  # <-- the length of the IRF
rho_r = 0.8
dr = 0.01 * rho_r ** np.arange(T)
shocks = {"G": dr}
unknowns_td = ['N']
targets_td = ["asset_mkt"]
irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)

# %% IRFs
T = 300  # <-- the length of the IRF
dB = 0.01 * 0.8 ** np.arange(T)
shocks = {"G": dr, "B": dB}
unknowns_td = ['N']
targets_td = ["asset_mkt"]
irfs_B = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)

#%% Plot IRFs
show_irfs([irfs, irfs_B], ["r","C","Y", "A", "goods_mkt", "asset_mkt", "B"],  labels=["..."], figsize=(18,3))



# %%
print(f"Execution time: {time.time() - start:.2f} seconds")
# %%
