"""
================================================================================
HANK Model with Durable Goods and Energy Transition
================================================================================

This script builds and solves a Heterogeneous Agent New Keynesian (HANK) model
where households make decisions over:
    - Nondurable consumption (core goods)
    - Energy consumption (brown vs. green)
    - Durable goods (brown vs. green durables, e.g. gas car vs. electric car)
    - Savings (one-period bond)

The model is used to study the distributional and aggregate effects of:
    1. A shock to the price of brown energy (e.g. oil price shock)
    2. A carbon tax on brown energy
    3. A monetary policy shock (interest rate shock)

Structure
---------
    1. Imports and calibration
    2. Isolated analysis of the household block
    3. Full model assembly (HA and HANK versions)
    4. Steady-state computation
    5. Comparative statics
    6. Impulse Response Functions (IRFs)

Dependencies: sequence_jacobian, numpy, matplotlib
================================================================================
"""

# %%
# =============================================================================
# 1. IMPORTS
# =============================================================================

import warnings
import json
from copy import deepcopy

import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import drawdag

# Model blocks
from HH_Block import hh
from Model_Blocks import (
    fiscal, mkt_clearing, prod,
    rsrce_cstrt, nkpc, inflation, taylor_rule
)

# Custom utility functions (plotting IRFs, policy functions, etc.)
from Fun.my_funs import *

# Suppress RuntimeWarnings (e.g. from numerical solver steps)
warnings.filterwarnings("error", category=RuntimeWarning)


# %%
# =============================================================================
# 2. CALIBRATION
# =============================================================================
# All parameters are quarterly unless otherwise noted.

baseline_calibration = {

    # -------------------------------------------------------------------------
    # Preferences
    # -------------------------------------------------------------------------
    "beta":        0.965,      # Household discount factor
    "gamma":       1 / 0.8,   # Coefficient of relative risk aversion (CRRA)
    "taste_shock": 1e-3,       # Idiosyncratic taste shock (smooths discrete durable choice)

    # Labor supply
    "frisch": 1,               # Frisch elasticity of labor supply
    "vphi":   1,               # Scale parameter on disutility of work. Will be calibrated to match WNKPC at SS.

    # -------------------------------------------------------------------------
    # Idiosyncratic Productivity
    # -------------------------------------------------------------------------
    "rho_e": 0.95,             # Persistence of productivity AR(1) process
    "sd_e":  0.50,             # Standard deviation of productivity innovation
    "n_e":   5,                # Number of grid points for productivity

    # -------------------------------------------------------------------------
    # Asset Grid
    # -------------------------------------------------------------------------
    "min_a": 0.0,              # Borrowing constraint (natural borrowing limit = 0)
    "max_a": 100,              # Upper bound of asset grid
    "n_a":   20,               # Number of grid points for assets

    # -------------------------------------------------------------------------
    # Aggregate Labor and Output (normalized at steady state)
    # -------------------------------------------------------------------------
    "N": 1,                    # Aggregate labor supply (normalized to 1)
    "Y": 1,                    # Aggregate output (normalized to 1)
    "Z": 1,                    # Total factor productivity in core goods sector
    "p_core": 1,               # Price of nondurable consumption good (numéraire)
    "Div": 0,                  # Dividends paid to households (zero in baseline)

    # -------------------------------------------------------------------------
    # Durable Goods
    # Brown durables: e.g. internal combustion vehicles (run on gasoline)
    # Green durables: e.g. electric vehicles (run on electricity)
    # -------------------------------------------------------------------------
    "delta_g": 0.05,           # Depreciation rate of green durables (quarterly)
    "psi_g":   0.1,            # Switching cost from brown to green durable

    # -------------------------------------------------------------------------
    # Government
    # -------------------------------------------------------------------------
    "B":   2,                  # Stock of government debt
    "G":   0.1,                # Government spending (exogenous)
    "Tax": 0,                  # Lump-sum tax (endogenous at SS to satisfy GBC)
    "tau": 0,                  # Labor income tax rate

    # -------------------------------------------------------------------------
    # Energy Prices and Policy
    # Brown energy: gasoline; Green energy: electricity
    # -------------------------------------------------------------------------
    "p_e_b": 1.0,              # Consumer price of brown energy (gasoline)
    "p_e_g": 0.8,              # Consumer price of green energy (electricity)
    "tau_b": 0.0,              # Carbon tax on brown energy (0 = no tax at baseline)
    "tau_g": 0.0,              # Subsidy on green energy (0 = no subsidy at baseline)

    # -------------------------------------------------------------------------
    # Consumption Aggregator
    # C = [xi * C_core^((nu-1)/nu) + (1-xi) * C_energy^((nu-1)/nu)]^(nu/(nu-1))
    # -------------------------------------------------------------------------
    "xi":        0.70,         # Share of core goods in consumption bundle
    "nu":        0.4,          # Elasticity of substitution between core and energy
    "markup_ss": 1.2,          # Steady-state price markup in goods market

    # -------------------------------------------------------------------------
    # Financial Environment
    # -------------------------------------------------------------------------
    "r":   0.05 / 4,           # Real interest rate (5% annualized → quarterly)

    # -------------------------------------------------------------------------
    # Monetary Policy (Taylor Rule)
    # i_t = rss + phi_pi * pi_t + ishock_t
    # -------------------------------------------------------------------------
    "rss":    0.05 / 4,        # Steady-state nominal interest rate
    "phi_pi": 1.5,             # Taylor rule coefficient on inflation
    "ishock": 0,               # Monetary policy shock (0 at steady state)

    # -------------------------------------------------------------------------
    # Wage Phillips Curve
    # -------------------------------------------------------------------------
    "piw":     0.0,            # Wage inflation (0 at steady state)
    "theta_w": 0.75,           # Calvo wage stickiness (prob. of not re-optimizing)

    # -------------------------------------------------------------------------
    # Standalone HH block analysis only (not used in full model)
    # -------------------------------------------------------------------------
    "w": 1,                    # Wage (fixed when analyzing HH block in isolation)
}


# %%
# =============================================================================
# 3. HOUSEHOLD BLOCK — ISOLATED ANALYSIS
# =============================================================================
# Solve the household block alone (partial equilibrium) to inspect:
#   - Policy functions (consumption, savings, durable choice)
#   - Steady-state distribution over assets, income, and durable type

hh_sol = hh.steady_state(baseline_calibration)

# --- Print steady-state durable shares ---
print("Steady-state durable shares (partial equilibrium):")
print(f"  Brown durables (D_B): {np.round(hh_sol['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G): {np.round(hh_sol['D_G'] * 100, 3)}%")

# --- Plot household policy functions ---
# ie_list: income grid points to display
# d_list, d_tilde_list: current and target durable state indices
ss_dict_partial = {"baseline": hh_sol}
policy_functions_Simple(
    ss_dict_partial,
    ie_list=[2],
    d_list=[1],
    d_tilde_list=[0, 1],
    xmax=10,
    figsize=0.8,
)


# %%
# =============================================================================
# 4. FULL MODEL ASSEMBLY
# =============================================================================
# Two model variants:
#   - `ha`   : Real (flexible-price) HA model — no nominal rigidities
#   - `hank` : HANK model — adds wage rigidity and monetary policy

ha = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod],
    name="Simple HA Model",
)

hank = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc, inflation, taylor_rule],
    name="HANK Model",
)



# %%
# =============================================================================
# 5. STEADY STATE — HA MODEL
# =============================================================================
# The steady state is found by jointly solving for unknowns such that
# three equilibrium conditions (targets) hold simultaneously.
#
# Unknowns (3):   Tax, beta, N
# Targets  (3):
#   - GBC       : Government budget constraint balances (G + rB = Tax + ΔB)
#   - asset_mkt : Asset market clears (∫ a di = B)
#   - labor_mkt : Labor market clears (N^s = N^d)

unknowns_ss = {
    "Tax":  baseline_calibration["Tax"],
    "beta": 0.97,
    "N":    baseline_calibration["N"],
}
targets_ss = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
}

ss = ha.solve_steady_state(
    baseline_calibration,
    unknowns_ss,
    targets_ss,
    solver="hybr",
)

# --- Print key steady-state values ---
print("\nSteady State — HA Model:")
print(f"  Government debt (B):   {ss['B']:.4f}")
print(f"  Lump-sum tax (Tax):    {ss['Tax']:.4f}")
print(f"  Real interest rate (r):{ss['r']:.4f}")
print(f"  Discount factor (β):   {ss['beta']:.4f}")
print(f"  Brown durables (D_B):  {np.round(ss['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G):  {np.round(ss['D_G'] * 100, 3)}%")

# --- Plot policy functions at general equilibrium SS ---
ss_dict_ge = {"baseline": ss}
policy_functions_Simple(
    ss_dict_ge,
    ie_list=[4],
    d_list=[0],
    d_tilde_list=[0, 1],
    xmax=10,
    figsize=0.8,
)


# %%
# =============================================================================
# 5b. STEADY STATE — HANK MODEL
# =============================================================================
# The HANK model adds wage stickiness, so we need an additional unknown (vphi)
# and an additional target (wage Phillips curve, wnkpc).
#
# Unknowns (4): Tax, beta, vphi, N
# Targets  (4): GBC, asset_mkt, wnkpc, labor_mkt

unknowns_ss_hank = {
    "Tax":  ss["Tax"],
    "beta": ss["beta"],
    "vphi": baseline_calibration["vphi"],
    "N":    baseline_calibration["N"],
}
targets_ss_hank = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "wnkpc":     0.0,   # Wage New Keynesian Phillips Curve: zero at SS
    "labor_mkt": 0.0,
}

ss_hank = hank.solve_steady_state(
    ss,
    unknowns_ss_hank,
    targets_ss_hank,
    solver="hybr",
)

# %%
# =============================================================================
# 6. COMPARATIVE STATICS
# =============================================================================
# Vary psi_g across a grid and re-solve the steady state
# to see how key aggregate variables respond.

param_grid = {"psi_g": np.linspace(0.06, 0.14, 5)}

cs_outputs = ["psi_g", "D_B", "D_G", "r", "Tax", "C", "Y"]

results = comparative_statics_plot(
    ha=hank,
    ss_base=ss_hank,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=cs_outputs,
    plot_deviation=False,
)

# %%
# =============================================================================
# 7. IMPULSE RESPONSE FUNCTIONS (IRFs)
# =============================================================================
# We use the sequence-space Jacobian method to compute linear IRFs.
# Each shock is modeled as an AR(1): x_t = rho * x_{t-1} + e * epsilon_t
#
# Unknowns (transition): Tax, Y, N, w  (solved to satisfy the 4 targets)
# Targets  (transition): asset_mkt, GBC, wnkpc, labor_mkt

unknowns_td = ["Tax", "Y", "N", "w"]
targets_td  = ["asset_mkt", "GBC", "wnkpc", "labor_mkt"]

# Variables to plot and their LaTeX labels
# outputs = [
#     "C", "C_CORE", "C_E", "Y", "w", "N", "N_D",
#     "D_B", "D_G", "G", "B", "Tax", "r", "piw", "i",
#     "rsrce_cstrt", "AD", "AD_CORE", "AD_DURABLES", "AS",
# ]
# names_outputs = [
#     r"Consumption: $C$",
#     r"Core Consumption: $C_{core}$",
#     r"Energy Consumption: $C_E$",
#     r"Output: $Y$",
#     r"Wage: $w$",
#     r"Labor Supply: $N^s$",
#     r"Labor Demand: $N^d$",
#     r"Brown Durable Stock: $D_B$",
#     r"Green Durable Stock: $D_G$",
#     r"Government Spending: $G$",
#     r"Government Debt: $B$",
#     r"Lump-Sum Tax: $Tax$",
#     r"Interest Rate: $r$",
#     r"Wage Inflation: $\pi_w$",
#     r"Nominal Interest Rate: $i$",
# ]
outputs = [
    "C", "Y", "w",
    "D_B", "D_G", "Tax", "r", "piw", "i",
]
names_outputs = [
    r"Consumption: $C$",
    r"Output: $Y$",
    r"Wage: $w$",
    r"Brown Durable Stock: $D_B$",
    r"Green Durable Stock: $D_G$",
    r"Lump-Sum Tax: $Tax$",
    r"Interest Rate: $r$",
    r"Nominal Interest Rate: $i$",
]



# --- Shock 1: Brown energy price shock (e.g. oil price increase) ---
# Interpretation: a 1% rise in the consumer price of gasoline, fading at rate 0.80
IRFs_p_e_b = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e={"p_e_b": 0.01},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
    figsize=(12, 9),
)

# --- Shock 2: Carbon tax shock ---
# Interpretation: a 1% increase in the carbon tax on brown energy, fading at rate 0.80
IRFs_tau_b = plot_linear_irfs(
    shocks_list=["tau_b"],
    e={"tau_b": 0.01},
    rho={"tau_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
    figsize=(12, 9),
)

# --- Shock 3: Monetary policy shock (interest rate shock) ---
# Interpretation: a 1% unexpected increase in the nominal rate, fading at rate 0.80
IRFs_i = plot_linear_irfs(
    shocks_list=["ishock"],
    e={"ishock": 0.01},
    rho={"ishock": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
    figsize=(12, 9),
)

# --- Compare IRFs across shocks ---
# Overlays the brown energy price shock and the carbon tax shock on the same plots
show_irfs(
    [IRFs_p_e_b, IRFs_tau_b],
    outputs,
    titles=names_outputs,
    labels=["Brown Energy Price Shock", "Carbon Tax Shock"],
)



# %%
