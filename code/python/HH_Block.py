"""
================================================================================
Household Block — Durable Goods with Discrete-Continuous Choice
================================================================================

This module defines the heterogeneous household block for the HANK model.
Households are heterogeneous in:
    - Productivity (idiosyncratic AR(1) shock)
    - Asset holdings (one-period bond)
    - Durable good type (Brown or Green, e.g. gas car vs. electric car)

Household problem
-----------------
Each period, households make three nested decisions:

    Stage 0 — Durable depreciation (exogenous Markov process)
        The existing durable may break down stochastically.

    Stage 1 — Productivity realization (exogenous AR(1) Markov process)
        The idiosyncratic productivity shock is realized.

    Stage 2 — Discrete durable choice (Logit / smoothed max)
        Households choose which durable type to hold next period.
        Switching from Green to Brown incurs an adjustment cost (psi_g).

    Stage 3 — Continuous consumption-savings choice (DC-EGM)
        Given the discrete durable choice, households choose how much
        to consume (C) and save (a'). The consumption bundle C aggregates:
            - Core nondurable goods (C_core)
            - Energy services (C_E), which depend on the durable type:
                * Brown durable → brown energy (gasoline) at price p_e_b
                * Green durable → green energy (electricity) at price p_e_g

Solution method
---------------
Stage 3 uses the Discrete-Choice Endogenous Grid Method (DC-EGM).
The upper envelope resolves non-convexities introduced by the discrete choice.

Dependencies: sequence_jacobian, numba, numpy
================================================================================
"""

# =============================================================================
# Imports
# =============================================================================

# Standard library
from dataclasses import dataclass
from typing import Tuple, Callable

# Numerical computing
import numpy as np
from numba import njit

# Sequence-Jacobian framework
from sequence_jacobian import grids, interpolate
from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D


# Custom utilities (grid construction, logit choice, stage block wrappers)
from SSJ_Fun.utils import (
    make_d_grid_simple,
    LogitChoiceDurable,
    StageBlockDurables,
)
from Fun.my_funs import *


# =============================================================================
# Stage Definitions
# =============================================================================
# The household problem is solved by backward induction through four stages.
# Stages 0 and 1 are exogenous (Markov transitions).
# Stages 2 and 3 involve household optimization.

# Stage 0: Durable depreciation — durable good breaks down with some probability
depreciation_stage = ExogenousMaker(
    markov_name="d_markov", index=0, name="durable"
)

# Stage 1: Productivity shock — idiosyncratic AR(1) productivity is realized
prod_stage = ExogenousMaker(
    markov_name="e_markov", index=2, name="prod"
)

# Stage 2: Discrete durable choice — household picks Brown or Green durable
# The Logit smoother (with scale = taste_shock) avoids a hard discrete kink,
# making the model differentiable for the sequence-space Jacobian.
def fake_util_f(V, vphi):
    flow_u = np.array([[0, 0],  
                       [0, 0]])           
    shape = np.zeros((2, 2,) + V.shape[2:])
    flow_u = flow_u[..., np.newaxis, np.newaxis, np.newaxis] + shape
    return flow_u

durables_stage = LogitChoiceDurable(
    value="V",
    backward="Va",
    index=0,
    name="durables",
    taste_shock_scale="taste_shock",
    f=fake_util_f
)


# =============================================================================
# Utility Function
# =============================================================================

@njit
def util(c, gamma):
    """
    CRRA utility function.

    Parameters
    ----------
    c : float or array
        Consumption level. Must be strictly positive.
    gamma : float
        Coefficient of relative risk aversion.
        gamma = 1 → log utility; gamma ≠ 1 → standard CRRA.

    Returns
    -------
    float or array
        Period utility u(c).
    """
    if gamma == 1.0:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)


# =============================================================================
# Stage 3: Continuous Consumption-Savings Choice (DC-EGM)
# =============================================================================

def dcegm(V, Va, a_grid, e_grid, disp_inc, adj_matrix, z_grid, r, T, beta, gamma, p_bundle):
    """
    One backward iteration step using the Discrete-Choice Endogenous Grid
    Method (DC-EGM).

    Given the continuation value function V and its derivative Va (with respect
    to end-of-period assets a'), this function solves for the optimal consumption
    c and savings a' on the exogenous asset grid.

    State space dimensions: (n_d_tilde, n_d, n_e, n_a)
        n_d_tilde : number of discrete durable choices available (Brown, Green)
        n_d       : current durable type held by the household
        n_e       : productivity grid points
        n_a       : asset grid points

    Parameters
    ----------
    V : ndarray, shape (n_d, n_e, n_a)
        Continuation value function (beginning of next period).
    Va : ndarray, shape (n_d, n_e, n_a)
        Derivative of V with respect to end-of-period assets a'.
    a_grid : ndarray, shape (n_a,)
        Exogenous asset grid (common savings grid).
    e_grid : ndarray, shape (n_e,)
        Productivity grid points.
    disp_inc : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Disposable income = (1+r)*a + w*e + T - adj_cost.
    adj_matrix : ndarray, shape (n_d_tilde, n_d)
        Switching cost from current durable d to new durable d_tilde.
    z_grid : ndarray, shape (n_e,)
        Labor income on the grid: z = w * N * e.
    r : float
        Real interest rate.
    T : ndarray, shape (n_e,)
        Net lump-sum transfers (dividends minus taxes), by productivity.
    beta : float
        Household discount factor.
    gamma : float
        CRRA coefficient.
    p_bundle : ndarray, shape (n_d_tilde,)
        Price of the consumption bundle, one entry per durable choice.

    Returns
    -------
    V : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Updated value function on the exogenous grid.
    Va : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Updated marginal value of assets (envelope condition).
    a : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Optimal end-of-period asset choice a'.
    c : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Optimal consumption bundle quantity c.
    uce : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Marginal utility of consumption scaled by productivity (used for
        wage setting block).
    """
    n_d = adj_matrix.shape[0]

    # --- Endogenous grid step (EGM) ---
    # From the Euler equation: u'(c) = beta * Va
    # → c_endo = (beta * Va)^(-1/gamma)
    W = beta * V                                              # Discounted continuation value
    uc_endo = beta * Va                                       # Euler equation RHS
    c_endo = uc_endo ** (-1 / gamma)                          # Implied consumption (endogenous grid)

    # Recover the endogenous asset grid a_endo corresponding to c_endo:
    # budget constraint: (1+r)*a_endo + z + T - adj_cost = p_bundle * c_endo + a'
    a_endo = (
        p_bundle[..., np.newaxis, np.newaxis, np.newaxis] * c_endo
        + a_grid[np.newaxis, np.newaxis, np.newaxis, :]
        + adj_matrix[..., np.newaxis, np.newaxis]
        - z_grid[np.newaxis, np.newaxis, :, np.newaxis]
        - T[np.newaxis, np.newaxis, :, np.newaxis]
    ) / (1 + r)

    # --- Upper envelope ---
    # Because the endogenous grid is non-monotone (due to discrete choice kinks),
    # we use the upper envelope to select the globally optimal solution.
    d_type   = np.arange(n_d)[:, None, None, None] * np.ones_like(a_endo)
    p_c_type = p_bundle[:, None, None, None]        * np.ones_like(a_endo)

    V, c, a = upperenv(W, a_endo, disp_inc, a_grid, d_type, p_c_type, gamma)

    # --- Envelope condition: update Va ---
    uc = np.maximum(1e-8, c) ** (-gamma)   # Marginal utility u'(c)
    uc = make_strictly_decreasing(uc)       # Enforce monotonicity (fix numerical issues)
    Va = (1 + r) * uc                       # Envelope condition: Va = (1+r) * u'(c)

    # Productivity-weighted marginal utility (used in wage NKPC)
    uce = e_grid[np.newaxis, np.newaxis, :, np.newaxis] * uc

    return V, Va, a, c, uce


def upperenv(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args):
    """
    Wrapper around the core upper envelope routine.

    Temporarily reshapes the 4D state space (n_d_tilde, n_d, n_e, n_a) into
    a 2D array (n_states, n_a) to allow a generic looped implementation,
    then restores the original shape.
    """
    shape  = W.shape
    n_flat = -1  # collapse all non-asset dimensions into one

    W        = W.reshape((n_flat, shape[-1]))
    a_endo   = a_endo.reshape((n_flat, shape[-1]))
    d_type   = d_type.reshape((n_flat, shape[-1]))
    p_c_type = p_c_type.reshape((n_flat, shape[-1]))
    disp_inc = disp_inc.reshape((n_flat, shape[-1]))

    V, c, a = upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args)

    return V.reshape(shape), c.reshape(shape), a.reshape(shape)


@njit
def upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, p_c_type, *args):
    """
    Core upper envelope algorithm (Druedahl & Jørgensen, 2017).

    For each non-asset state (collapsed into axis 0), interpolates the value
    function from the endogenous grid onto the exogenous grid and resolves
    non-convexities by keeping only the segment giving the highest value.

    Parameters
    ----------
    W : ndarray, shape (n_states, n_a)
        Continuation value on the endogenous grid.
    a_endo : ndarray, shape (n_states, n_a)
        Endogenous (EGM) asset grid — may be non-monotone.
    disp_inc : ndarray, shape (n_states, n_a)
        Disposable income on the exogenous grid.
    a_grid : ndarray, shape (n_a,)
        Exogenous (common) asset grid — strictly increasing.
    d_type : ndarray, shape (n_states, n_a)
        Discrete durable type index for each state.
    p_c_type : ndarray, shape (n_states, n_a)
        Price of the consumption bundle for each state.
    *args
        Additional arguments passed to util() — typically gamma.

    Returns
    -------
    V : ndarray, shape (n_states, n_a)
        Value function on the exogenous grid.
    c : ndarray, shape (n_states, n_a)
        Optimal consumption on the exogenous grid.
    a : ndarray, shape (n_states, n_a)
        Optimal savings on the exogenous grid.
    """
    n_b, n_a = W.shape
    a = np.zeros_like(W)
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    for ib in range(n_b):
        d, p_c = int(d_type[ib, 0]), p_c_type[ib, 0]

        # Iterate over segments of the (possibly non-monotone) endogenous grid
        for ja in range(n_a - 1):
            a_low,  a_high  = a_endo[ib, ja],   a_endo[ib, ja + 1]
            W_low,  W_high  = W[ib, ja],         W[ib, ja + 1]
            ap_low, ap_high = a_grid[ja],         a_grid[ja + 1]

            for ia in range(n_a):
                acur    = a_grid[ia]
                coh_cur = disp_inc[ib, ia]

                interp = (a_low <= acur <= a_high)
                extrap = (ja == n_a - 2) and (acur > a_endo[ib, n_a - 1])

                # Early exit: exogenous grid is increasing, so once acur
                # exceeds a_high without being in the last segment, skip ahead
                if a_high < acur < a_endo[ib, n_a - 1]:
                    break

                if interp or extrap:
                    # Interpolate continuation value and savings choice
                    W0 = interpolate.interpolate_point(acur, a_low, a_high, W_low, W_high)
                    a0 = interpolate.interpolate_point(acur, a_low, a_high, ap_low, ap_high)

                    # Implied consumption from the budget constraint
                    c0 = max(1e-5, (coh_cur - a0) / p_c)
                    V0 = util(c0, *args) + W0

                    # Upper envelope: keep the solution that delivers higher utility
                    if V0 > V[ib, ia]:
                        a[ib, ia] = a0
                        c[ib, ia] = c0
                        V[ib, ia] = V0

        # Enforce borrowing constraint: for gridpoints at or below the natural
        # borrowing limit, set a' = 0 and consume all disposable income
        ia = 0
        while ia < n_a and a_grid[ia] <= a_endo[ib, 0]:
            a[ib, ia] = a_grid[0]
            c[ib, ia] = max(1e-5, disp_inc[ib, ia] / p_c)
            V[ib, ia] = util(c[ib, ia], *args) + W[ib, 0]
            ia += 1

    return V, c, a


# =============================================================================
# Hetoutputs for Stage 3
# =============================================================================
# These functions are called after the continuous choice stage and aggregated
# across the distribution to produce aggregate quantities.

def D_demand(c):
    """
    Indicator functions for current and target durable type.

    Returns four arrays, each with the same shape as c:
        d_t_B : 1 if the household is choosing Brown next period
        d_B   : 1 if the household currently holds a Brown durable
        d_t_G : 1 if the household is choosing Green next period
        d_G   : 1 if the household currently holds a Green durable

    Aggregating these over the distribution gives the share of households
    with each durable type (D_B, D_G) and the share switching (D_tB, D_tG).
    """
    shape = c.shape
    n_d   = shape[0]   # number of durable types (2: Brown, Green)

    # Build indicator arrays for target durable (d_tilde axis = axis 0)
    # and current durable (d axis = axis 1)
    dd_tilde = [np.zeros(shape, dtype=c.dtype) for _ in range(n_d)]
    dd       = [np.zeros(shape, dtype=c.dtype) for _ in range(n_d)]

    for d in range(n_d):
        dd_tilde[d][d, ...]    = 1   # choosing durable d (target)
        dd[d][:, d, ...]       = 1   # currently holding durable d

    # Unpack into named outputs expected by SSJ
    d_t_B, d_t_G = dd_tilde[0], dd_tilde[1]
    d_B,   d_G   = dd[0],       dd[1]

    return d_t_B, d_B, d_t_G, d_G


def decomposition_consu_bundle(c, p_core, p_bundle, p_e, nu, xi, tau_vec):
    """
    Decompose the consumption bundle c into core goods and energy services.

    The household consumption bundle takes a CES form:
        C = [xi * C_core^((nu-1)/nu) + (1-xi) * C_E^((nu-1)/nu)]^(nu/(nu-1))

    Optimal demand from CES cost minimization:
        C_core = xi  * (p_core / p_bundle)^(-nu) * C
        C_E    = (1-xi) * ((1+tau)*p_e / p_bundle)^(-nu) * C

    Parameters
    ----------
    c : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Total consumption bundle quantity.
    p_core : float
        Price of core nondurable goods (numéraire = 1).
    p_bundle : ndarray, shape (n_d_tilde,)
        Price index of the consumption bundle, one per durable choice.
    p_e : ndarray, shape (n_d_tilde,)
        Energy price faced by each durable type.
    nu : float
        Elasticity of substitution between core goods and energy.
    xi : float
        Share parameter on core goods in the CES aggregator.
    tau_vec : ndarray, shape (n_d_tilde,)
        Energy tax rate by durable type (tau_b for Brown, tau_g for Green).

    Returns
    -------
    c_core : ndarray  — core nondurable consumption
    c_E    : ndarray  — total energy consumption
    c_E_b  : ndarray  — energy consumption of Brown-durable households
    c_E_g  : ndarray  — energy consumption of Green-durable households
    t_E    : ndarray  — energy tax revenue paid by each household
    """
    # Broadcast p_bundle along (n_d, n_e, n_a) dimensions
    p_bun = p_bundle[..., np.newaxis, np.newaxis, np.newaxis]
    p_en  = p_e[...,     np.newaxis, np.newaxis, np.newaxis]
    tau   = tau_vec[..., np.newaxis, np.newaxis, np.newaxis]

    # Core consumption (same CES formula for all durable types)
    c_core = xi * (p_core / p_bun) ** (-nu) * c

    # Energy consumption (zero if energy price is zero, e.g. no energy good)
    c_E = np.where(
        p_e[..., np.newaxis, np.newaxis, np.newaxis] == 0,
        0.0,
        (1 - xi) * ((1 + tau) * p_en / p_bun) ** (-nu) * c,
    )

    # Split energy consumption by durable type
    # Brown-durable households (axis 0, index 0) use brown energy
    # Green-durable households (axis 0, index 1) use green energy
    c_E_b, c_E_g = np.zeros_like(c_E), np.zeros_like(c_E)
    c_E_b[0, ...] = c_E[0, ...]
    c_E_g[1, ...] = c_E[1, ...]

    # Energy tax revenue: tau * p_e * C_E
    t_E = c_E * tau * p_en

    return c_core, c_E, c_E_b, c_E_g, t_E


# Initialize Stage 3 (continuous consumption-savings choice)
consav_stage = Continuous1D(
    backward=["V", "Va"],
    policy="a",
    f=dcegm,
    name="consav",
    hetoutputs=[D_demand, decomposition_consu_bundle],
)


# =============================================================================
# Hetinputs — Grids, Income, Prices, and Transfers
# =============================================================================
# These functions are called once before solving the household block.
# They construct the grids and pre-compute quantities that enter the backward
# recursion as parameters.

def hh_init(disp_inc, a_grid, gamma):
    """
    Initialize the value function V and its derivative Va.

    Uses a simple guess based on the utility of consuming all disposable income
    (shifted to be positive). Va is computed by finite differences on the grid.

    Parameters
    ----------
    disp_inc : ndarray, shape (n_d_tilde, n_d, n_e, n_a)
        Disposable income on the full state grid.
    a_grid : ndarray, shape (n_a,)
        Asset grid.
    gamma : float
        CRRA coefficient.

    Returns
    -------
    V  : ndarray, shape (n_d, n_e, n_a)  — initial value function guess
    Va : ndarray, shape (n_d, n_e, n_a)  — initial derivative guess
    """
    # Shift disposable income to ensure positivity for the log/power utility
    V = util(disp_inc - np.min(disp_inc) + 1, gamma)
    #V = np.mean(V, axis=0)   # Average over the first axis (d_tilde) to match (n_d, n_e, n_a)

    # Finite-difference derivative in the asset dimension
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:]  - V[..., :-2])  / (a_grid[2:]  - a_grid[:-2])
    Va[..., 0]    = (V[..., 1]   - V[..., 0])    / (a_grid[1]   - a_grid[0])
    Va[..., -1]   = (V[..., -1]  - V[..., -2])   / (a_grid[-1]  - a_grid[-2])

    return V, Va


def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, delta_g, delta_b):
    """
    Construct the state-space grids.

    Parameters
    ----------
    rho_e, sd_e, n_e : float, float, int
        Persistence, standard deviation, and grid size for the productivity
        AR(1) process. Discretized via Rouwenhorst.
    min_a, max_a, n_a : float, float, int
        Bounds and grid size for the asset grid (log-spaced).
    delta_g : float
        Depreciation rate of green durables (governs the durable Markov chain).

    Returns
    -------
    e_grid, e_dist, e_markov : productivity grid, stationary distribution, transition matrix
    a_grid                   : asset grid
    d_grid, d_markov, d_grid_name : durable grid, depreciation transition, names
    """
    e_grid, e_dist, e_markov = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = grids.agrid(max_a, n_a, min_a)
    d_grid, d_markov, d_grid_name = make_d_grid_simple(delta_g, delta_b)
    return e_grid, e_dist, e_markov, a_grid, d_grid, d_markov, d_grid_name


def income_grid(e_grid, w, N):
    """
    Compute labor income on the productivity grid.

    z(e) = w * N * e
    where w is the wage, N aggregate labor, and e idiosyncratic productivity.
    """
    z_grid = w * N * e_grid
    return z_grid


def create_vectors(tau_b, tau_g, p_e_b, p_e_g, nu, xi, p_core):
    """
    Pre-compute price vectors and consumption bundle price indices.

    Assembles durable-type-specific vectors for energy prices, tax rates,
    and the composite consumption bundle price index p_bundle, defined by:

        p_bundle = [xi * p_core^(1-nu) + (1-xi) * p_tilde_e^(1-nu)]^(1/(1-nu))

    where p_tilde_e = (1 + tau) * p_e is the tax-inclusive energy price.

    Returns
    -------
    tau_vec   : ndarray (2,)  — tax rates [tau_b, tau_g]
    p_e       : ndarray (2,)  — energy prices [p_e_b, p_e_g]
    d         : ndarray (2,)  — durable quantity vector (both = 1 at SS)
    p_tilde_e : ndarray (2,)  — tax-inclusive energy prices
    p_bundle  : ndarray (2,)  — consumption bundle price index per durable type
    """
    d         = np.array([1, 1])
    p_e       = np.array([p_e_b, p_e_g])
    tau_vec   = np.array([tau_b, tau_g])
    p_tilde_e = (1 + tau_vec) * p_e

    if nu != 1:
        p_bundle = (xi * p_core ** (1 - nu) + (1 - xi) * p_tilde_e ** (1 - nu)) ** (1 / (1 - nu))
    else:
        # Cobb-Douglas limit (nu → 1)
        p_bundle = p_core ** xi * p_tilde_e ** (1 - xi)

    return tau_vec, p_e, d, p_tilde_e, p_bundle


def transfers(e_dist, Div, Tax, e_grid):
    """
    Compute net lump-sum transfers to households by productivity type.

    Transfers are uniform (lump-sum) across productivity types:
        T(e) = Div / sum(e_dist) - Tax / sum(e_dist)

    Parameters
    ----------
    e_dist : ndarray (n_e,)   — stationary productivity distribution
    Div    : float            — aggregate dividends
    Tax    : float            — aggregate lump-sum tax
    e_grid : ndarray (n_e,)   — productivity grid (not used but kept for signature)

    Returns
    -------
    T : ndarray (n_e,)  — net transfer by productivity type
    """
    # Lump-sum rules: uniform incidence across all productivity types
    div_rule = np.ones_like(e_grid)
    tax_rule = np.ones_like(e_grid)

    div = Div / np.sum(e_dist * div_rule) * div_rule
    tax = Tax / np.sum(e_dist * tax_rule) * tax_rule
    
    T = div - tax

    return T

def adj_costs(psi_g):
    """
    Construct the durable switching cost matrix.

    adj_matrix[d_tilde, d] = cost of switching from current durable d
                              to new durable d_tilde.

    Convention:
        d = 0 → Brown (e.g. gas car)
        d = 1 → Green (e.g. electric car)

    Switching from Green to Brown incurs cost psi_g (replacement cost).
    All other transitions are free.

    Returns
    -------
    adj_matrix : ndarray, shape (2, 2)
    """
    adj_matrix = np.array([
        [0.0,   0.0  ],   # Brown → Brown: free  |  Green → Brown: free
        [psi_g, 0.0  ],   # Brown → Green: costs psi_g  |  Green → Green: free
    ])
    return adj_matrix


def disp_inc_f(a_grid, z_grid, T, r, adj_matrix):
    """
    Compute disposable income on the full (n_d_tilde, n_d, n_e, n_a) grid.

    Disposable income is what the household has available to split between
    consumption and savings, after paying (or receiving) the durable
    switching cost:

        disp_inc(d_tilde, d, e, a) =
            (1 + r) * a          [asset income]
            + z(e)               [labor income]
            + T(e)               [lump-sum transfers]
            - adj_cost(d_tilde, d) [durable switching cost]

    Array broadcasting dimensions: (n_d_tilde, n_d, n_e, n_a)
    """
    disp_inc = (
          (1 + r) * a_grid[np.newaxis, np.newaxis, np.newaxis, :]   # (1, 1, 1, n_a)
        + z_grid[np.newaxis, np.newaxis, :, np.newaxis]              # (1, 1, n_e, 1)
        + T[np.newaxis, np.newaxis, :, np.newaxis]                   # (1, 1, n_e, 1)
        - adj_matrix[..., np.newaxis, np.newaxis]                    # (n_d_tilde, n_d, 1, 1)
    )
    return disp_inc


# =============================================================================
# Household Block Assembly
# =============================================================================
# Combine the four stages and hetinputs into a single staged household block.

hh = StageBlockDurables(
    [depreciation_stage, prod_stage, durables_stage, consav_stage],
    name="hh",
    backward_init=hh_init,
    hetinputs=[make_grids, income_grid, transfers, adj_costs, disp_inc_f, create_vectors],
)
# %%
