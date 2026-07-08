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
    rsrce_cstrt, nkpc, nkpc_ss, inflation, taylor_rule, real_rule, others
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

    # ---------------------- ---------------------------------------------------
    # Preferences
    # -------------------------------------------------------------------------
    "beta":        0.965,      # Household discount factor
    "gamma":       1 / 0.8,   # Coefficient of relative risk aversion (CRRA)
    "taste_shock": 1e-1,       # Idiosyncratic taste shock (smooths discrete durable choice)

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
    "delta_b": 0.0,           # Depreciation rate of brown durables (quarterly)
    "psi_g":   1.5,            # Switching cost from brown to green durable

    # -------------------------------------------------------------------------
    # Government
    # -------------------------------------------------------------------------
    "B":   2,                  # Stock of government debt
    "G_ss":0.1,                # Government spending (exogenous)
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
    "xi":        0.94,         # Share of core goods in consumption bundle
    "nu":        0.40,          # Elasticity of substitution between core and energy
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
# print(f"  Brown durables (D_BB): {np.round(hh_sol['D_BB'] * 100, 3)}%")
# print(f"  Green durables (D_BG): {np.round(hh_sol['D_BG'] * 100, 3)}%")
# print(f"  Brown durables (D_GB): {np.round(hh_sol['D_GB'] * 100, 3)}%")
# print(f"  Green durables (D_GG): {np.round(hh_sol['D_GG'] * 100, 3)}%")
print(f"  Brown durables (D_B): {np.round(hh_sol['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G): {np.round(hh_sol['D_G'] * 100, 3)}%")
print(f"  Green durables (C): {hh_sol['C']}")



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


hank_ss = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc_ss, inflation, taylor_rule, others],
    name="HANK Model S.S",
)

hank = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc, inflation, taylor_rule, others],
    name="HANK Model",
)

hank_real = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc, inflation, real_rule, others],
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
    "psi_g":baseline_calibration["psi_g"]
}
targets_ss = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B":       0.95
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
    "psi_g":ss["psi_g"],
    "vphi": baseline_calibration["vphi"],
    "N":    baseline_calibration["N"],
}
targets_ss_hank = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "wnkpc":     0.0,   # Wage New Keynesian Phillips Curve: zero at SS
    "labor_mkt": 0.0,
    "D_B":       0.95
}

# ss_hank = hank.solve_steady_state(
#     ss,
#     unknowns_ss_hank,
#     targets_ss_hank,
#     solver="hybr",
# )

calib = hank_ss.solve_steady_state(
    ss,
    unknowns_ss_hank,
    targets_ss_hank,
    solver="hybr"
)


ss_hank =  hank.steady_state(calib)
ss_hank_real =  hank_real.steady_state(calib)


# --- Plot policy functions at general equilibrium SS ---
ss_dict_ge = {"baseline": ss_hank}

policy_function_disc(
    ss_dict_ge,
    xmax=20,
    xmin=0,
    d_tilde_list=[0,2],
    d_list=[0],
    ie_list=[2],
    figsize=0.8,
    models=['baseline'],
    title='Durable Adoption Decision: Brown Owners at Median Productivity',
    save_path="../../output/figures/policy_function_durable_brown_ie2.png"
)

policy_function_disc(
    ss_dict_ge,
    xmax=20,
    xmin=0,
    d_tilde_list=[1,3],
    d_list=[2],
    ie_list=[2],
    figsize=0.8,
    models=['baseline'],
    title='Durable Adoption Decision: Green Owners at Median Productivity',
    save_path="../../output/figures/policy_function_durable_green_ie2.png"
)

policy_function_switch_heatmap(ss_dict_ge, save_path="../../output/figures/policy_function_heatmap.png")

plot_distribution(ss_hank, truncate_at=11, normalize=False, save_path="../../output/figures/stationary_distr.png")
plot_distribution(ss_hank, lines_dim=1,truncate_at=11, normalize=False, labels=['Very Low','Low','Middle','High','Very High'], title="Wealth Distribution by Productivity Type (Mass in the Economy)",save_path="../../output/figures/stationary_distr_prod.png")
plot_distribution(ss_hank, lines_dim=0,truncate_at=11, normalize=False, labels=['BB','BG','GB','GG'], title="Wealth Distribution by (collapsed) durable state",save_path="../../output/figures/stationary_distr_durables.png")





# %%
# =============================================================================
# 6. COMPARATIVE STATICS
# =============================================================================
# Vary psi_g across a grid and re-solve the steady state
# to see how key aggregate variables respond.
unknowns_cs = {
    "Tax":  baseline_calibration["Tax"],
    "beta": 0.97,
    "N":    baseline_calibration["N"]
}
targets_cs = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0
}


param_grid = {"psi_g": ss["psi_g"]*np.linspace(0.1, 1.5, 5)}

cs_outputs = ["D_B", "D_G"]

comparative_statics_plot_shares(
    ha=hank,
    ss_base=ss_hank,
    param_grid=param_grid,
    unknowns_ss=unknowns_cs,
    targets_ss=targets_cs,
    outputs=cs_outputs,
    x_label="Green Adoption Cost",
    title="Steady-State Durable Composition Under Varying Adoption Costs",
    save_path="../../output/figures/comp_stat/comp_stat_psig.png"
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

unknowns_td = ["Tax", "Y", "N", "piw"]
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
    "C", "Y", "piw","D_B", "D_G", "Tax", "B"
]
names_outputs = [
    r"Consumption: $C$",
    r"Output: $Y$",
    r"Wage Inflation: $\pi_w$",
    r"Brown Durable Stock: $D_B$",
    r"Green Durable Stock: $D_G$",
    r"Lump-Sum Tax: $Tax$",
    r"Public Debt: $B$"
]



# --- Shock 1: Brown energy price shock (e.g. oil price increase) ---
# Interpretation: a 1% rise in the consumer price of gasoline, fading at rate 0.80
IRFs_p_e_b = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e={"p_e_b": 10},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_p_e_b.png"
)
show_irfs(
    [IRFs_p_e_b],
    ["p_e_b"] + outputs,
    titles=["Energy price shock"] + names_outputs,
    labels=["Brown Energy Price Shock"],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_p_e_b.png"
)

# --- Shock 2: Carbon tax shock ---
# Interpretation: a 1% increase in the carbon tax on brown energy, fading at rate 0.80
IRFs_tau_b = plot_linear_irfs(
    shocks_list=["tau_b"],
    e={"tau_b": 10},
    rho={"tau_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_tau_b.png"
)

# --- Shock 3: Monetary policy shock (interest rate shock) ---
# Interpretation: a 1% unexpected increase in the nominal rate, fading at rate 0.80
IRFs_i = plot_linear_irfs(
    shocks_list=["ishock"],
    e={"ishock": 1},
    rho={"ishock": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_i.png"
)

# --- Compare IRFs across shocks ---
# Overlays the brown energy price shock and the carbon tax shock on the same plots


show_irfs(
    [IRFs_p_e_b, IRFs_tau_b],
    ["p_e_b", "tau_b"] + outputs,
    titles=["Energy price shock", "Carbon tax shock"] + names_outputs,
    labels=["Brown Energy Price Shock", "Carbon Tax Shock"],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_p_e_b_taub.png"
)

# %%
# =============================================================================
# 8. COUNTERFACTUAL: BROWN ENERGY PRICE SHOCK WITHOUT DURABLE ADOPTION
# =============================================================================

# --- Step 1: Compute IRFs under the brown energy tax response ---
unknowns_td_noadoption = ["Tax", "Y", "N", "piw", "psi_g"]
targets_td_noadoption  = ["asset_mkt", "GBC", "wnkpc", "labor_mkt", "D_B_target"]

IRFs_p_e_b_noadoption = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e  ={"p_e_b": 10},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td_noadoption,
    targets_td =targets_td_noadoption,
    ha     =hank,
    ss     =ss_hank,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# --- Step 2: Compare IRFs across fiscal rules ---
# Differences across the two IRFs reflect the distributional consequences
# of targeting brown energy users (via tau_b) vs. all households (via Tax).
show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_noadoption],
    ["p_e_b"] + ["psi_g"] + outputs,
    titles=["Energy price shock"] + ["Green Adoption Cost (Not to be shown)"] + names_outputs,
    labels=[
        "Baseline",
        "No Adoption",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_noAdoption.png"
)


# %%
# =============================================================================
# 8. COUNTERFACTUAL: BROWN ENERGY PRICE SHOCK WITHOUT DURABLE ADOPTION
# =============================================================================
# Goal: isolate the *direct* demand effect of an oil price shock from the
# *adoption* channel (households switching from Brown to Green durables).
#
# Strategy: compute an alternative steady state where psi_g is so large
# that no household ever switches durable type ("no adoption" equilibrium).
# We calibrate delta_b so that the share of Brown durables (D_B = ...)
# matches the baseline SS, ensuring a fair comparison.
#
# Unknowns (4): Tax, beta, N, delta_b
# Targets  (4): GBC, asset_mkt, labor_mkt, D_B = ...
# =============================================================================

# --- Step 1: Build starting point from baseline SS ---
# Start from the baseline HANK steady state and raise the switching cost
# to a prohibitively large value, effectively shutting down durable adoption.
ss_hank_no_adoption = deepcopy(ss_hank)
ss_hank_no_adoption["psi_g"] = 1e6   # Switching cost → ∞: no household ever switches durable

# --- Step 2: Solve for the counterfactual steady state ---
# We back out delta_b so that the Brown durable share matches the baseline (...).
# This ensures the two steady states are comparable in terms of durable composition.
unknowns_ss_no_adoption = {
    "Tax":     ss_hank["Tax"],
    "beta":    ss_hank["beta"],
    "N":       ss_hank["N"],
    "delta_b": ss_hank["delta_b"],   # Backed out to match D_B target
}
targets_ss_no_adoption = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B":       ss_hank['D_B'],   # Match baseline Brown durable share
}

ss_hank_no_adoption = hank.solve_steady_state(
    ss_hank_no_adoption,
    unknowns_ss_no_adoption,
    targets_ss_no_adoption,
    solver="hybr",
)

# --- Print key steady-state values ---
print("\nSteady State — No Adoption Counterfactual:")
print(f"  Government debt (B):      {ss_hank_no_adoption['B']:.4f}")
print(f"  Lump-sum tax (Tax):       {ss_hank_no_adoption['Tax']:.4f}")
print(f"  Real interest rate (r):   {ss_hank_no_adoption['r']:.4f}")
print(f"  Discount factor (β):      {ss_hank_no_adoption['beta']:.4f}")
print(f"  Brown depreciation (δ_b): {ss_hank_no_adoption['delta_b']:.4f}")
print(f"  Brown durables (D_B):     {np.round(ss_hank_no_adoption['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G):     {np.round(ss_hank_no_adoption['D_G'] * 100, 3)}%")


# %%
# --- Step 3: Compute IRFs for the brown energy price shock ---
# Same shock as before (1% rise in p_e_b, AR(1) with rho=0.80),
# but now evaluated at the no-adoption steady state.
IRFs_p_e_b_no_adoption = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e  ={"p_e_b": 10},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td =targets_td,
    ha     =hank,
    ss     =ss_hank_no_adoption,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# %%
# --- Step 4: Compare IRFs with and without the adoption channel ---
# Overlaying the two IRFs reveals the contribution of durable switching
# to the aggregate and distributional response to an oil price shock.
show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_no_adoption],
    ["p_e_b"] + outputs,
    titles=["Energy price shock"] + names_outputs,
    labels=[
        "Baseline (with adoption)",
        "Counterfactual (no adoption)",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_noAdoption_b.png"
)

# %%
# =============================================================================
# 9. COUNTERFACTUAL: BROWN ENERGY PRICE SHOCK WITH A BROWN ENERGY TAX RESPONSE
# =============================================================================
# Goal: compare two fiscal response rules to an oil price shock.

# =============================================================================

# --- Step 1: Compute IRFs under the brown energy tax response ---

IRFs_p_e_b_tau_response = plot_linear_irfs(
    shocks_list=["p_e_b","tau_b"],
    e  ={"p_e_b": 10,"tau_b": -10},
    rho={"p_e_b": 0.80,"tau_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td =targets_td,
    ha     =hank,
    ss     =ss_hank,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# --- Step 2: Compare IRFs across fiscal rules ---
# Differences across the two IRFs reflect the distributional consequences
# of targeting brown energy users (via tau_b) vs. all households (via Tax).
show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_tau_response],
    ["p_e_b"] + ["tau_b"] + outputs,
    titles=["Energy price shock"] + ["Brown Energy tax rate"]+ names_outputs,
    labels=[
        "Baseline",
        "Counterfactual (brown energy tax response)",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_brownSubsidy.png"
)

# %%
# =============================================================================
# 9. COUNTERFACTUAL: BROWN ENERGY PRICE SHOCK WITH A BROWN ENERGY TAX RESPONSE - DEFICIT FINANCED
# =============================================================================
# Goal: compare two fiscal response rules to an oil price shock.

# =============================================================================
unknowns_td_deficit = ["B", "Y", "N", "piw"]

# --- Step 1: Compute IRFs under the brown energy tax response ---

IRFs_p_e_b_tau_response_deficit = plot_linear_irfs(
    shocks_list=["p_e_b","tau_b"],
    e  ={"p_e_b": 10,"tau_b": -10},
    rho={"p_e_b": 0.80,"tau_b": 0.80},
    unknowns_td=unknowns_td_deficit,
    targets_td =targets_td,
    ha     =hank,
    ss     =ss_hank,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# --- Step 2: Compare IRFs across fiscal rules ---
# Differences across the two IRFs reflect the distributional consequences
# of targeting brown energy users (via tau_b) vs. all households (via Tax).
show_irfs(
    [IRFs_p_e_b_tau_response, IRFs_p_e_b_tau_response_deficit],
    ["p_e_b"] + ["tau_b"] + outputs + ["G"],
    titles=["Energy price shock"] + ["Brown Energy tax rate"]+ names_outputs + [r"Government Expenditures: $G$"],
    labels=[
        "Tax-financed",
        "Deficit-financed",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_brownSubsidy_deficit.png"
)

# %%
# =============================================================================
# 10. COUNTERFACTUAL A — CARBON TAX STEADY STATE (SAME DURABLE COMPOSITION)
# =============================================================================
# Goal: assess how the economy responds to an oil price shock when a carbon
# tax is already in place in steady state.
#
# We construct an alternative SS with tau_b = 0.05 (a 5% carbon tax on brown
# energy). To isolate the effect of the tax from any compositional change,
# we back out psi_g so that the Brown durable share remains at its baseline
# level (D_B = ss_hank["D_B"]). A higher psi_g makes switching to green
# harder, counteracting the adoption incentive created by the carbon tax.
#
# Unknowns (4): Tax, beta, N, psi_g
# Targets  (4): GBC, asset_mkt, labor_mkt, D_B = baseline
# =============================================================================

# --- Step 1: Start from baseline SS and impose the carbon tax ---
ss_hank_carbontax = deepcopy(ss_hank)
ss_hank_carbontax["tau_b"] = 0.05   # 5% carbon tax on brown energy

# --- Step 2: Solve for the counterfactual SS ---
# psi_g is backed out to keep D_B equal to the baseline share,
# so differences in IRFs reflect the tax environment, not composition.
unknowns_ss_carbontax = {
    "Tax":   ss_hank["Tax"],
    "beta":  ss_hank["beta"],
    "N":     ss_hank["N"],
    "psi_g": ss_hank["psi_g"],   # backed out to match D_B target
}
targets_ss_carbontax = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B":       ss_hank["D_B"],   # match baseline Brown durable share
}

ss_hank_carbontax = hank.solve_steady_state(
    ss_hank_carbontax,
    unknowns_ss_carbontax,
    targets_ss_carbontax,
    solver="hybr",
)

# --- Print key steady-state values ---
print("\nSteady State — Carbon Tax (same durable composition):")
print(f"  Government debt (B):          {ss_hank_carbontax['B']:.4f}")
print(f"  Lump-sum tax (Tax):           {ss_hank_carbontax['Tax']:.4f}")
print(f"  Real interest rate (r):       {ss_hank_carbontax['r']:.4f}")
print(f"  Discount factor (β):          {ss_hank_carbontax['beta']:.4f}")
print(f"  Green switching cost (ψ_g):   {ss_hank_carbontax['psi_g']:.4f}")
print(f"  Brown durables (D_B):         {np.round(ss_hank_carbontax['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G):         {np.round(ss_hank_carbontax['D_G'] * 100, 3)}%")

# --- Step 3: Compute IRFs for the brown energy price shock ---
IRFs_p_e_b_carbontax = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e  ={"p_e_b": 1},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td =targets_td,
    ha     =hank,
    ss     =ss_hank_carbontax,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# --- Step 4: Compare IRFs ---
# Differences reflect how a pre-existing carbon tax changes the transmission
# of an oil price shock, holding durable composition constant.
show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_carbontax],
    ["p_e_b"] + outputs,
    titles=["Energy price shock"] + names_outputs,
    labels=[
        "Baseline (τ_b = 0)",
        "Counterfactual (τ_b = 0.05, same D_B)",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_ETF.png"
)


# %%
# =============================================================================
# 11. COUNTERFACTUAL B — GREENER STEADY STATE (LOWER BROWN DURABLE SHARE)
# =============================================================================
# Goal: assess how the economy responds to an oil price shock when it starts
# from a "greener" steady state where fewer households own brown durables.
#
# Here we target D_B = 70% (vs. ~83% at baseline) and back out the carbon
# tax tau_b needed to decarbonize the durable stock to that level.
# Unlike Counterfactual A, the durable composition *does* differ from baseline.
# This allows us to study whether a greener fleet attenuates oil price shocks.
#
# Unknowns (4): Tax, beta, N, tau_b
# Targets  (4): GBC, asset_mkt, labor_mkt, D_B = 0.70
# =============================================================================

# --- Step 1: Solve for the greener SS ---
# tau_b is backed out so that the equilibrium Brown share equals 70%.
unknowns_ss_greener = {
    "Tax":   ss_hank["Tax"],
    "beta":  ss_hank["beta"],
    "N":     ss_hank["N"],
    "tau_b": 0,   # backed out to match D_B = 0.70 target
}
targets_ss_greener = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B":       0.90,   # 70% Brown share — greener than baseline
}

ss_hank_greener = hank.solve_steady_state(
    ss_hank,
    unknowns_ss_greener,
    targets_ss_greener,
    solver="hybr",
)

# --- Print key steady-state values ---
print("\nSteady State — Greener Economy (D_B = 90%):")
print(f"  Government debt (B):      {ss_hank_greener['B']:.4f}")
print(f"  Lump-sum tax (Tax):       {ss_hank_greener['Tax']:.4f}")
print(f"  Real interest rate (r):   {ss_hank_greener['r']:.4f}")
print(f"  Discount factor (β):      {ss_hank_greener['beta']:.4f}")
print(f"  Brown energy tax (τ_b):   {ss_hank_greener['tau_b']:.4f}")
print(f"  Brown durables (D_B):     {np.round(ss_hank_greener['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G):     {np.round(ss_hank_greener['D_G'] * 100, 3)}%")

# --- Step 2: Compute IRFs for the brown energy price shock ---
IRFs_p_e_b_greener = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e  ={"p_e_b": 10},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td =targets_td,
    ha     =hank,
    ss     =ss_hank_greener,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# --- Step 3: Compare IRFs ---
# Differences now reflect both the carbon tax environment *and* the
# compositional effect of having more green-durable households at the start.
show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_greener],
    ["p_e_b"] + outputs,
    titles=["Energy price shock"] + names_outputs,
    labels=[
        "Baseline (D_B = "+str(np.round(ss_hank['D_B'] * 100, 1))+r"%, $\tau_b$ = "+str(ss_hank['tau_b'])+")",
        "Counterfactual (D_B = "+str(np.round(ss_hank_greener['D_B'] * 100, 1))+r"%, $\tau_b$ = "+str(np.round(ss_hank_greener['tau_b'],2))+")",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_ETF2.png"
)
# %%



# %%
# %%
# =============================================================================
# 12. COUNTERFACTUAL: POLICY MIX. ACCOMODATIVE MONETARY POLICY
# =============================================================================
# Goal: Look at an accomodative monetary policy shock when oil shock.

# =============================================================================

# --- Step 1: Compute IRFs under the brown energy tax response ---

IRFs_p_e_b_i = plot_linear_irfs(
    shocks_list=["p_e_b","ishock"],
    e  ={"p_e_b": 10,"ishock": -1},
    rho={"p_e_b": 0.80,"ishock": 0.80},
    unknowns_td=unknowns_td,
    targets_td =targets_td,
    ha     =hank,
    ss     =ss_hank,
    outputs=outputs,
    titles =names_outputs,
    figsize=(12, 9),
)

# --- Step 2: Compare IRFs across fiscal rules ---
# Differences across the two IRFs reflect the distributional consequences
# of targeting brown energy users (via tau_b) vs. all households (via Tax).
show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_i],
    ["p_e_b"] + ["i"] + outputs,
    titles=["Energy price shock"] + ["Nominal Interest Rate"]+ names_outputs,
    labels=[
        "Baseline",
        "Counterfactual (Accomodative MP shock)",
    ],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_accomodativeMP.png"
)

# %% Taylor Rule vs Real rule to an oil shock
# =============================================================================
# 13. COUNTERFACTUAL: Real interest rate rule
IRFs_i_Taylor = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e={"p_e_b": 1},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs,
    titles=names_outputs,
)
IRFs_i_Real = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e={"p_e_b": 1},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank_real,
    ss=ss_hank_real,
    outputs=outputs,
    titles=names_outputs,
)

# --- Compare IRFs across shocks ---
# Overlays the brown energy price shock and the carbon tax shock on the same plots

show_irfs(
    [IRFs_i_Taylor, IRFs_i_Real],
    ["p_e_b", "i", "r"] + outputs,
    titles=["Energy price shock", "Nom. IR","Real IR"] + names_outputs,
    labels=["Taylor Rule", "Real Rule"],
    figsize=(12, 9),
    save_path="../../output/figures/IRFs/irfs_realIRRule.png"
)
# %%
# =============================================================================
# 14. DECOMPOSITION OF THE OUTPUT RESPONSE TO A BROWN ENERGY PRICE SHOCK
# =============================================================================
# Goal: decompose the Y response to the oil price shock into its components
# via the resource constraint identity (AD = AS = Y), which is satisfied
# automatically since rsrce_cstrt is not among the targets_td.
#
#   Y = C_CORE + (p_e_b * C_E_B) + (p_e_g * C_E_G) + psi_g * D_GB + G
#
# Energy spending is further split into brown vs. green to disentangle the
# direct price/volume effect on gasoline from the substitution effect
# towards electricity as households switch durables.
# =============================================================================

outputs_decomp = ["C_CORE", "C_E_B", "C_E_G", "D_GB", "G", "Y", "p_e_b"]

IRFs_decomp = plot_linear_irfs(
    shocks_list=["p_e_b"],
    e={"p_e_b": 10},
    rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs=outputs_decomp,
    titles=outputs_decomp,
)

# --- Steady-state levels used to linearize each contribution ---
p_e_b_ss = ss_hank["p_e_b"]
p_e_g_ss = ss_hank["p_e_g"]
C_E_B_ss = ss_hank["C_E_B"]
psi_g_ss = ss_hank["psi_g"]

# --- Contributions to dY (brown/green energy spending disentangled) ---
contrib_core          = IRFs_decomp["C_CORE"]
contrib_energy_brown  = p_e_b_ss * IRFs_decomp["C_E_B"] + C_E_B_ss * IRFs_decomp["p_e_b"]
contrib_energy_green  = p_e_g_ss * IRFs_decomp["C_E_G"]
contrib_durable        = psi_g_ss * IRFs_decomp["D_GB"]
contrib_G              = IRFs_decomp["G"]

contrib_total = (
    contrib_core + contrib_energy_brown + contrib_energy_green
    + contrib_durable + contrib_G
)

# --- Accounting check: sum of contributions should equal the Y IRF ---
T_plot = 40  # horizon to display
max_gap = np.max(np.abs(contrib_total[:T_plot] - IRFs_decomp["Y"][:T_plot]))
print(f"Max gap between sum of contributions and Y IRF: {max_gap:.6e}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(contrib_core[:T_plot], label="Core consumption", linewidth=2)
ax.plot(contrib_energy_brown[:T_plot], label="Brown energy spending", linewidth=2, color="saddlebrown")
ax.plot(contrib_energy_green[:T_plot], label="Green energy spending", linewidth=2, color="seagreen")
ax.plot(contrib_durable[:T_plot], label="Durable switching cost", linewidth=2)
ax.plot(contrib_G[:T_plot], label="Government spending", linewidth=2)
ax.plot(contrib_total[:T_plot], label="Sum of contributions", linewidth=2.5,
        linestyle="--", color="black")
ax.plot(IRFs_decomp["Y"][:T_plot], label="Y (direct IRF, check)", linewidth=2.5,
        linestyle=":", color="red", marker="o", markersize=3)

ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Quarters")
ax.set_ylabel("Deviation from steady state")
ax.set_title("Decomposition of the Output Response to a Brown Energy Price Shock")
ax.legend(frameon=False)
fig.tight_layout()

save_path = "../../output/figures/IRFs/irfs_Y_decomposition.png"
fig.savefig(save_path, dpi=200)
plt.show()



# %% Decomposition between direct and indirect response of Core Consumption
# Direct household response
J_hh = hh.jacobian(ss_hank, inputs=["p_e_b"], T=300)

shock_path = 10 * 0.80 ** np.arange(300)

C_CORE_direct = J_hh["C_CORE"]["p_e_b"] @ shock_path

# GE feedback
C_CORE_total = IRFs_decomp["C_CORE"]
C_CORE_GE = C_CORE_total - C_CORE_direct

# Plot
T_plot = 40
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(C_CORE_direct[:T_plot], label="Partial equilibrium", linewidth=2)
ax.plot(C_CORE_GE[:T_plot], label="GE feedback", linewidth=2)
ax.plot(C_CORE_total[:T_plot], label="Total", linestyle="--", linewidth=2.5)

ax.axhline(0, linewidth=0.8)
ax.set_xlabel("Quarters")
ax.set_ylabel("Deviation from steady state")
ax.set_title("Core Consumption Response to Energy Price Shock")
ax.legend(frameon=False)

plt.tight_layout()

plt.savefig(
    "../../output/figures/IRFs/C_CORE_decomposition.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# %%
