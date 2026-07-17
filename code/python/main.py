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
    2. Household block — isolated analysis
    3. Full model assembly (HA and HANK variants)
    4. Steady state — HA model
    5. Steady state — HANK model (core, headline, real-rule variants)
    6. Comparative statics
    7. Baseline IRFs (energy price, carbon tax, monetary shocks)
    8. Counterfactual: shutting down durable adoption
    9. Counterfactual: brown-energy tax response (tax- vs deficit-financed)
    10. Counterfactual: pre-existing carbon tax (same durable composition)
    11. Counterfactual: greener steady state (lower brown durable share)
    12. Counterfactual: accommodative monetary policy
    13. Taylor rule vs. real-rate rule
    14. Decomposition of the output / core-consumption response
    15. Headline vs. core inflation targeting
    16. Parameter sensitivity: rho_i, beta_spread

Most counterfactuals follow one of two patterns, factored into helpers in
Fun/my_funs.py:
    - irf_variant_comparison()      : same steady state, different shock/unknowns_td
    - ss_counterfactual_comparison(): alternative steady state, then IRFs there

Dependencies: sequence_jacobian, numpy, matplotlib
================================================================================
"""

# %%
# =============================================================================
# 1. IMPORTS
# =============================================================================

import warnings
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import sequence_jacobian as sj

from HH_Block import hh, hh_ss
from Model_Blocks import (
    fiscal, mkt_clearing, prod,
    rsrce_cstrt, nkpc, nkpc_ss, core_inflation, headline_inflation,
    taylor_rule_headline, real_rule, others,
    rsrce_cstrt_leak_E
)
from Fun.my_funs import *  # noqa: F401,F403 — plotting / IRF / steady-state helpers

warnings.filterwarnings("error", category=RuntimeWarning)

FIG_DIR = "../../output/figures"
IRF_DIR = f"{FIG_DIR}/IRFs"

# Only render the "exploratory" figures (plot_all=True) when actually iterating
# on the model; final headline figures are always plotted/saved regardless.
PLOT_ALL = False


# %%
# =============================================================================
# 2. CALIBRATION
# =============================================================================
# All parameters are quarterly unless otherwise noted.

baseline_calibration = {

    # -------------------------------------------------------------------------
    # Preferences
    # -------------------------------------------------------------------------
    "beta_bar":    0.965,      # Household discount factor (population mean)
    "beta_spread": 0.00,       # Dispersion of beta across the 3 discount-factor types
    "n_beta":      3,
    "gamma":       1 / 0.8,    # Coefficient of relative risk aversion (CRRA)
    "taste_shock": 1e-1,       # Idiosyncratic taste shock (smooths discrete durable choice)

    # Labor supply
    "frisch": 1,               # Frisch elasticity of labor supply
    "vphi":   1,               # Scale parameter on disutility of work; calibrated to match WNKPC at SS

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
    "delta_b": 0.0,            # Depreciation rate of brown durables (quarterly)
    "psi_g":   1.5,            # Switching cost from brown to green durable

    # -------------------------------------------------------------------------
    # Government
    # -------------------------------------------------------------------------
    "B":       2,               # Stock of government debt
    "G_ss":    0.1,             # Government spending (exogenous)
    "kappa_g": 0.10,            # Response of gov. exp. to debt deviation (Leeper, Plante & Traum, 2010)
    "Tax":     0,                # Lump-sum tax (endogenous at SS to satisfy GBC)
    "Tax_NFA": 0,                # Lump-sum tax to repay NFA (shock)
    "tau":     0,                # Labor income tax rate
    "leakage_E": 0,

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
    "nu":        0.40,         # Elasticity of substitution between core and energy
    "markup_ss": 1.2,          # Steady-state price markup in goods market

    # -------------------------------------------------------------------------
    # Financial Environment
    # -------------------------------------------------------------------------
    "r": 0.05 / 4,             # Real interest rate (5% annualized -> quarterly)

    # -------------------------------------------------------------------------
    # Monetary Policy (Taylor Rule)
    # i_t = rss + phi_pi * pi_t + ishock_t
    # -------------------------------------------------------------------------
    "rss":    0.05 / 4,        # Steady-state nominal interest rate
    "phi_pi": 1.2,             # Taylor rule coefficient on inflation
    "rho_i":  0.8,             # Persistence in Taylor Rule
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

print("Steady-state durable shares (partial equilibrium):")
print(f"  Brown durables (D_B): {np.round(hh_sol['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G): {np.round(hh_sol['D_G'] * 100, 3)}%")
print(f"  Consumption (C):      {hh_sol['C']}")


# %%
# =============================================================================
# 4. FULL MODEL ASSEMBLY
# =============================================================================
# Model variants:
#   - `ha`             : Real (flexible-price) HA model — no nominal rigidities
#   - `hank_ss`         : HANK model with the SS-only wage block (nkpc_ss), used to solve SS
#   - `hank`            : HANK model — wage rigidity, core-inflation Taylor rule
#   - `hank_headline`   : HANK model — Taylor rule responds to headline (energy-weighted) inflation
#   - `hank_real`       : HANK model — real-rate rule instead of a Taylor rule

ha = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod],
    name="Simple HA Model",
)

hank_ss = sj.create_model(
    [hh_ss, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc_ss, others],
    name="HANK Model S.S",
)

hank = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc, core_inflation, headline_inflation, taylor_rule_headline, others],
    name="HANK Model",
)

hank_headline = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc, core_inflation, headline_inflation,
     taylor_rule_headline, others],
    name="HANK Model Headline",
)

hank_real = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt, prod, nkpc, core_inflation, headline_inflation, real_rule, others],
    name="HANK Model Real Rule",
)

hank_leak = sj.create_model(
    [hh, fiscal, mkt_clearing, rsrce_cstrt_leak_E, prod, nkpc, core_inflation, headline_inflation, real_rule, others],
    name="HANK Model Leakage",
)


# %%
# =============================================================================
# 5. STEADY STATE — HA MODEL
# =============================================================================
# Unknowns (4): Tax, beta_bar, N, psi_g
# Targets  (4): GBC, asset_mkt, labor_mkt, D_B = 0.95

unknowns_ss = {
    "Tax":     baseline_calibration["Tax"],
    "beta_bar": 0.97,
    "N":       baseline_calibration["N"],
    "psi_g":   baseline_calibration["psi_g"],
}
targets_ss = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B":       0.95,
}

ss = ha.solve_steady_state(baseline_calibration, unknowns_ss, targets_ss, solver="hybr")


def print_ss_summary(ss_obj, label="HA Model"):
    print(f"\nSteady State — {label}:")
    print(f"  Government debt (B):   {ss_obj['B']:.4f}")
    print(f"  Lump-sum tax (Tax):    {ss_obj['Tax']:.4f}")
    print(f"  Real interest rate (r):{ss_obj['r']:.4f}")
    print(f"  Discount factor (β):   {ss_obj['beta_bar']:.4f}")
    print(f"  Brown durables (D_B):  {np.round(ss_obj['D_B'] * 100, 3)}%")
    print(f"  Green durables (D_G):  {np.round(ss_obj['D_G'] * 100, 3)}%")


print_ss_summary(ss, "HA Model")


# %%
# =============================================================================
# 5b. STEADY STATE — HANK MODEL (core, headline, real-rule variants)
# =============================================================================
# Adds wage stickiness: one extra unknown (vphi) and target (wnkpc).
# Unknowns (5): Tax, beta_bar, psi_g, vphi, N
# Targets  (5): GBC, asset_mkt, wnkpc, labor_mkt, D_B = 0.95

unknowns_ss_hank = {
    "Tax":     ss["Tax"],
    "beta_bar": ss["beta_bar"],
    "psi_g":   ss["psi_g"],
    "vphi":    baseline_calibration["vphi"],
    "N":       baseline_calibration["N"],
}
targets_ss_hank = {
    "GBC":       0.0,
    "asset_mkt": 0.0,
    "wnkpc":     0.0,
    "labor_mkt": 0.0,
    "D_B":       0.95,
}

calib = hank_ss.solve_steady_state(ss, unknowns_ss_hank, targets_ss_hank, solver="hybr")

# --- Fixed headline-inflation weights, computed once from the solved SS to
#     avoid a cyclic DAG dependency (household aggregate -> upstream block) ---
nom_C_ss = (
    calib["p_core"] * calib["C_CORE"]
    + calib["p_e_b"] * calib["C_E_B"]
    + calib["p_e_g"] * calib["C_E_G"]
)
calib["omega_core"] = calib["p_core"] * calib["C_CORE"] / nom_C_ss
calib["omega_eb"] = calib["p_e_b"] * calib["C_E_B"] / nom_C_ss
calib["omega_eg"] = calib["p_e_g"] * calib["C_E_G"] / nom_C_ss

ss_hank = hank.steady_state(calib)
ss_hank_headline = hank_headline.steady_state(calib)
ss_hank_real = hank_real.steady_state(calib)
ss_hank_leakage = hank_leak.steady_state(calib)


# --- Policy functions and stationary distribution at the GE steady state ---
ss_dict_ge = {"baseline": ss_hank}

# policy_function_disc(
#     ss_dict_ge, xmax=20, xmin=0, d_tilde_list=[0, 2], d_list=[0], ie_list=[2],
#     figsize=0.8, models=["baseline"],
#     title="Durable Adoption Decision: Brown Owners at Median Productivity",
#     save_path=f"{FIG_DIR}/policy_function_durable_brown_ie2.png",
# )
# policy_function_disc(
#     ss_dict_ge, xmax=20, xmin=0, d_tilde_list=[1, 3], d_list=[2], ie_list=[2],
#     figsize=0.8, models=["baseline"],
#     title="Durable Adoption Decision: Green Owners at Median Productivity",
#     save_path=f"{FIG_DIR}/policy_function_durable_green_ie2.png",
# )
# policy_function_switch_heatmap(ss_dict_ge, save_path=f"{FIG_DIR}/policy_function_heatmap.png")

# plot_distribution(ss_hank, truncate_at=11, normalize=False,
#                    save_path=f"{FIG_DIR}/stationary_distr.png")
# plot_distribution(ss_hank, lines_dim=1, truncate_at=11, normalize=False,
#                    labels=["Very Low", "Low", "Middle", "High", "Very High"],
#                    title="Wealth Distribution by Productivity Type (Mass in the Economy)",
#                    save_path=f"{FIG_DIR}/stationary_distr_prod.png")
# plot_distribution(ss_hank, lines_dim=0, truncate_at=11, normalize=False,
#                    labels=["BB", "BG", "GB", "GG"],
#                    title="Wealth Distribution by (collapsed) durable state",
#                    save_path=f"{FIG_DIR}/stationary_distr_durables.png")


# %%
# =============================================================================
# 6. COMPARATIVE STATICS
# =============================================================================
# Vary psi_g across a grid and re-solve the steady state to see how key
# aggregate variables respond.

unknowns_cs = {
    "Tax": baseline_calibration["Tax"],
    "beta_bar": 0.97,
    "N": baseline_calibration["N"],
}
targets_cs = {
    "GBC": 0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
}
param_grid = {"psi_g": ss["psi_g"] * np.linspace(0.1, 1.5, 5)}
cs_outputs = ["D_B", "D_G"]

comparative_statics_plot_shares(
    ha=hank, ss_base=ss_hank, param_grid=param_grid,
    unknowns_ss=unknowns_cs, targets_ss=targets_cs, outputs=cs_outputs,
    x_label="Green Adoption Cost",
    title="Steady-State Durable Composition Under Varying Adoption Costs",
    save_path=f"{FIG_DIR}/comp_stat/comp_stat_psig.png",
)


# %%
# =============================================================================
# 7. BASELINE IMPULSE RESPONSE FUNCTIONS (IRFs)
# =============================================================================
# Sequence-space Jacobian method. Each shock is an AR(1): x_t = rho * x_{t-1} + e * eps_t
#
# Unknowns (transition): Tax, Y, N, piw
# Targets  (transition): asset_mkt, GBC, wnkpc, labor_mkt

unknowns_td = ["Tax", "Y", "N", "piw"]
targets_td = ["asset_mkt", "GBC", "wnkpc", "labor_mkt"]

outputs = ["C", "Y", "piw", "D_B", "D_G", "Tax", "B"]
names_outputs = [
    r"Consumption: $C$",
    r"Output: $Y$",
    r"Wage Inflation: $\pi_w$",
    r"Brown Durable Stock: $D_B$",
    r"Green Durable Stock: $D_G$",
    r"Lump-Sum Tax: $Tax$",
    r"Public Debt: $B$",
]

# --- Shock 1: Brown energy price shock (e.g. oil price increase, 1%, rho=0.80) ---
IRFs_p_e_b = plot_linear_irfs(
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank_real, ss=ss_hank_real, outputs=outputs, titles=names_outputs,
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_p_e_b.png", plot=True,
)

# --- Shock 2: Carbon tax shock (1% increase, rho=0.80) ---
IRFs_tau_b = plot_linear_irfs(
    shocks_list=["tau_b"], e={"tau_b": 10}, rho={"tau_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank, ss=ss_hank, outputs=outputs, titles=names_outputs,
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_tau_b.png", plot=PLOT_ALL,
)

# --- Shock 3: Monetary policy shock (1% unexpected rate increase, rho=0.80) ---
IRFs_i = plot_linear_irfs(
    shocks_list=["ishock"], e={"ishock": 1}, rho={"ishock": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank, ss=ss_hank, outputs=outputs, titles=names_outputs,
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_i.png", plot=PLOT_ALL,
)

# --- Overlay: energy price shock vs. carbon tax shock ---
show_irfs(
    [IRFs_p_e_b, IRFs_tau_b], ["p_e_b", "tau_b"] + outputs,
    titles=["Energy price shock", "Carbon tax shock"] + names_outputs,
    labels=["Brown Energy Price Shock", "Carbon Tax Shock"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_p_e_b_taub.png",
)


# %%
# =============================================================================
# 8. COUNTERFACTUAL: SHUTTING DOWN DURABLE ADOPTION
# =============================================================================
# Two distinct ways to shut down the adoption channel:
#   8a. Along the transition only (psi_g adjusts each period to freeze D_B)
#   8b. In the steady state (psi_g -> infinity: no household ever switches)

# --- 8a: transition-level freeze ---
unknowns_td_noadoption = unknowns_td + ["psi_g"]
targets_td_noadoption = targets_td + ["D_B_target"]

IRFs_p_e_b_noadoption = irf_variant_comparison(
    ha=hank, ss=ss_hank, baseline_irfs=IRFs_p_e_b,
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td_noadoption, targets_td=targets_td_noadoption,
    outputs=outputs, titles=names_outputs,
    extra_vars=[("psi_g", "Green Adoption Cost (Not to be shown)")],
    labels=["Baseline", "No Adoption"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_noAdoption.png",
)

# --- 8b: steady-state freeze (psi_g -> 1e6), delta_b backed out to match D_B ---
ss_hank_no_adoption_start = deepcopy(ss_hank)
ss_hank_no_adoption_start["psi_g"] = 1e6

unknowns_ss_no_adoption = {
    "Tax": ss_hank["Tax"],
    "beta_bar": ss_hank["beta_bar"],
    "N": ss_hank["N"],
    "delta_b": ss_hank["delta_b"],
}
targets_ss_no_adoption = {
    "GBC": 0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B": ss_hank["D_B"],
}


def print_no_adoption_summary(ss_obj):
    print("\nSteady State — No Adoption Counterfactual:")
    print(f"  Government debt (B):      {ss_obj['B']:.4f}")
    print(f"  Lump-sum tax (Tax):       {ss_obj['Tax']:.4f}")
    print(f"  Real interest rate (r):   {ss_obj['r']:.4f}")
    print(f"  Discount factor (β):      {ss_obj['beta_bar']:.4f}")
    print(f"  Brown depreciation (δ_b): {ss_obj['delta_b']:.4f}")
    print(f"  Brown durables (D_B):     {np.round(ss_obj['D_B'] * 100, 3)}%")
    print(f"  Green durables (D_G):     {np.round(ss_obj['D_G'] * 100, 3)}%")


ss_hank_no_adoption, IRFs_p_e_b_no_adoption = ss_counterfactual_comparison(
    ha=hank, ss_start=ss_hank_no_adoption_start,
    unknowns_ss=unknowns_ss_no_adoption, targets_ss=targets_ss_no_adoption,
    baseline_irfs=IRFs_p_e_b,
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    outputs=outputs, titles=names_outputs,
    labels=["Baseline (with adoption)", "Counterfactual (no adoption)"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_noAdoption_b.png",
    print_summary=print_no_adoption_summary,
)


# %%
# =============================================================================
# 9. COUNTERFACTUAL: BROWN-ENERGY TAX RESPONSE TO THE OIL SHOCK
# =============================================================================
# Compare targeting brown-energy users (tau_b) vs. all households (Tax), and
# whether the tax response is financed by lump-sum taxes or by debt.

# --- 9a: tax-financed brown-energy tax response ---
IRFs_p_e_b_tau_response = irf_variant_comparison(
    ha=hank, ss=ss_hank, baseline_irfs=IRFs_p_e_b,
    shocks_list=["p_e_b", "tau_b"], e={"p_e_b": 10, "tau_b": -10}, rho={"p_e_b": 0.80, "tau_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    outputs=outputs, titles=names_outputs,
    extra_vars=[("tau_b", "Brown Energy tax rate")],
    labels=["Baseline", "Counterfactual (brown energy tax response)"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_brownSubsidy.png",
)

# --- 9b: deficit-financed variant (B adjusts instead of Tax) ---
unknowns_td_deficit = ["B", "Y", "N", "piw"]

IRFs_p_e_b_tau_response_deficit = irf_variant_comparison(
    ha=hank, ss=ss_hank, baseline_irfs=IRFs_p_e_b_tau_response,
    shocks_list=["p_e_b", "tau_b"], e={"p_e_b": 10, "tau_b": -10}, rho={"p_e_b": 0.80, "tau_b": 0.80},
    unknowns_td=unknowns_td_deficit, targets_td=targets_td,
    outputs=outputs + ["G"], titles=names_outputs + [r"Government Expenditures: $G$"],
    extra_vars=[("tau_b", "Brown Energy tax rate")],
    labels=["Tax-financed", "Deficit-financed"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_brownSubsidy_deficit.png",
)


# %%
# =============================================================================
# 10. COUNTERFACTUAL: PRE-EXISTING CARBON TAX (SAME DURABLE COMPOSITION)
# =============================================================================
# tau_b = 0.05 in steady state; psi_g backed out so D_B matches baseline,
# isolating the tax-environment effect from any compositional change.

ss_hank_carbontax_start = deepcopy(ss_hank)
ss_hank_carbontax_start["tau_b"] = 0.05

unknowns_ss_carbontax = {
    "Tax": ss_hank["Tax"],
    "beta_bar": ss_hank["beta_bar"],
    "N": ss_hank["N"],
    "psi_g": ss_hank["psi_g"],
}
targets_ss_carbontax = {
    "GBC": 0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B": ss_hank["D_B"],
}


def print_carbontax_summary(ss_obj):
    print("\nSteady State — Carbon Tax (same durable composition):")
    print(f"  Government debt (B):        {ss_obj['B']:.4f}")
    print(f"  Lump-sum tax (Tax):         {ss_obj['Tax']:.4f}")
    print(f"  Real interest rate (r):     {ss_obj['r']:.4f}")
    print(f"  Discount factor (β):        {ss_obj['beta_bar']:.4f}")
    print(f"  Green switching cost (ψ_g): {ss_obj['psi_g']:.4f}")
    print(f"  Brown durables (D_B):       {np.round(ss_obj['D_B'] * 100, 3)}%")
    print(f"  Green durables (D_G):       {np.round(ss_obj['D_G'] * 100, 3)}%")


ss_hank_carbontax, IRFs_p_e_b_carbontax = ss_counterfactual_comparison(
    ha=hank, ss_start=ss_hank_carbontax_start,
    unknowns_ss=unknowns_ss_carbontax, targets_ss=targets_ss_carbontax,
    baseline_irfs=IRFs_p_e_b,
    shocks_list=["p_e_b"], e={"p_e_b": 1}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    outputs=outputs, titles=names_outputs,
    labels=["Baseline (τ_b = 0)", "Counterfactual (τ_b = 0.05, same D_B)"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_ETF.png",
    print_summary=print_carbontax_summary,
)


# %%
# =============================================================================
# 11. COUNTERFACTUAL: GREENER STEADY STATE (LOWER BROWN DURABLE SHARE)
# =============================================================================
# Target D_B = 90% (vs. 95% at baseline); tau_b backed out to hit it.
# Unlike section 10, durable composition *does* differ from baseline here,
# so this captures both the tax environment and the compositional effect.

unknowns_ss_greener = {
    "Tax": ss_hank["Tax"],
    "beta_bar": ss_hank["beta_bar"],
    "N": ss_hank["N"],
    "tau_b": 0,
}
targets_ss_greener = {
    "GBC": 0.0,
    "asset_mkt": 0.0,
    "labor_mkt": 0.0,
    "D_B": 0.90,
}

ss_hank_greener = hank.solve_steady_state(ss_hank, unknowns_ss_greener, targets_ss_greener, solver="hybr")

print("\nSteady State — Greener Economy (D_B = 90%):")
print(f"  Government debt (B):     {ss_hank_greener['B']:.4f}")
print(f"  Lump-sum tax (Tax):      {ss_hank_greener['Tax']:.4f}")
print(f"  Real interest rate (r):  {ss_hank_greener['r']:.4f}")
print(f"  Discount factor (β):     {ss_hank_greener['beta_bar']:.4f}")
print(f"  Brown energy tax (τ_b):  {ss_hank_greener['tau_b']:.4f}")
print(f"  Brown durables (D_B):    {np.round(ss_hank_greener['D_B'] * 100, 3)}%")
print(f"  Green durables (D_G):    {np.round(ss_hank_greener['D_G'] * 100, 3)}%")

IRFs_p_e_b_greener = plot_linear_irfs(
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank, ss=ss_hank_greener, outputs=outputs, titles=names_outputs,
    figsize=(12, 9), plot=PLOT_ALL,
)

show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_greener], ["p_e_b"] + outputs,
    titles=["Energy price shock"] + names_outputs,
    labels=[
        f"Baseline (D_B = {np.round(ss_hank['D_B'] * 100, 1)}%, "
        rf"$\tau_b$ = {ss_hank['tau_b']})",
        f"Counterfactual (D_B = {np.round(ss_hank_greener['D_B'] * 100, 1)}%, "
        rf"$\tau_b$ = {np.round(ss_hank_greener['tau_b'], 2)})",
    ],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_ETF2.png",
)


# %%
# =============================================================================
# 12. COUNTERFACTUAL: ACCOMMODATIVE MONETARY POLICY DURING THE OIL SHOCK
# =============================================================================

IRFs_p_e_b_i = irf_variant_comparison(
    ha=hank, ss=ss_hank, baseline_irfs=IRFs_p_e_b,
    shocks_list=["p_e_b", "ishock"], e={"p_e_b": 10, "ishock": -1}, rho={"p_e_b": 0.80, "ishock": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    outputs=outputs, titles=names_outputs,
    extra_vars=[("i", "Nominal Interest Rate")],
    labels=["Baseline", "Counterfactual (Accomodative MP shock)"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_accomodativeMP.png",
)


# %%
# =============================================================================
# 13. TAYLOR RULE VS. REAL-RATE RULE
# =============================================================================

IRFs_i_Taylor = plot_linear_irfs(
    shocks_list=["p_e_b"], e={"p_e_b": 1}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank, ss=ss_hank, outputs=outputs, titles=names_outputs, plot=PLOT_ALL,
)
IRFs_i_Real = plot_linear_irfs(
    shocks_list=["p_e_b"], e={"p_e_b": 1}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank_real, ss=ss_hank_real, outputs=outputs, titles=names_outputs, plot=PLOT_ALL,
)

show_irfs(
    [IRFs_i_Taylor, IRFs_i_Real], ["p_e_b", "i", "r"] + outputs,
    titles=["Energy price shock", "Nom. IR", "Real IR"] + names_outputs,
    labels=["Taylor Rule", "Real Rule"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_realIRRule.png",
)


# %%
# =============================================================================
# 14. DECOMPOSITION OF THE OUTPUT RESPONSE TO A BROWN ENERGY PRICE SHOCK
# =============================================================================
# Y = C_CORE + p_e_b * C_E_B + p_e_g * C_E_G + psi_g * D_GB + G
# (resource-constraint identity; rsrce_cstrt is not a transition target, so
# this holds automatically — used here as an accounting check.)

outputs_decomp = ["C_CORE", "C_E_B", "C_E_G", "D_GB", "G", "Y", "p_e_b"]

IRFs_decomp = plot_linear_irfs(
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank, ss=ss_hank, outputs=outputs_decomp, titles=outputs_decomp, plot=PLOT_ALL,
)

p_e_b_ss = ss_hank["p_e_b"]
p_e_g_ss = ss_hank["p_e_g"]
C_E_B_ss = ss_hank["C_E_B"]
psi_g_ss = ss_hank["psi_g"]

contrib_core = IRFs_decomp["C_CORE"]
contrib_energy_brown = p_e_b_ss * IRFs_decomp["C_E_B"] + C_E_B_ss * IRFs_decomp["p_e_b"]
contrib_energy_green = p_e_g_ss * IRFs_decomp["C_E_G"]
contrib_durable = psi_g_ss * IRFs_decomp["D_GB"]
contrib_G = IRFs_decomp["G"]
contrib_total = contrib_core + contrib_energy_brown + contrib_energy_green + contrib_durable + contrib_G

T_plot = 40
max_gap = np.max(np.abs(contrib_total[:T_plot] - IRFs_decomp["Y"][:T_plot]))
print(f"Max gap between sum of contributions and Y IRF: {max_gap:.6e}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(contrib_core[:T_plot], label="Core consumption", linewidth=2)
ax.plot(contrib_energy_brown[:T_plot], label="Brown energy spending", linewidth=2, color="saddlebrown")
ax.plot(contrib_energy_green[:T_plot], label="Green energy spending", linewidth=2, color="seagreen")
ax.plot(contrib_durable[:T_plot], label="Durable switching cost", linewidth=2)
ax.plot(contrib_G[:T_plot], label="Government spending", linewidth=2)
ax.plot(contrib_total[:T_plot], label="Sum of contributions", linewidth=2.5, linestyle="--", color="black")
ax.plot(IRFs_decomp["Y"][:T_plot], label="Y (direct IRF, check)", linewidth=2.5,
        linestyle=":", color="red", marker="o", markersize=3)
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Quarters")
ax.set_ylabel("Deviation from steady state")
ax.set_title("Decomposition of the Output Response to a Brown Energy Price Shock")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{IRF_DIR}/irfs_Y_decomposition.png", dpi=200)
plt.show()

# --- Core consumption: direct (partial-eq.) vs. GE feedback ---
J_hh = hh.jacobian(ss_hank, inputs=["p_e_b"], T=300)
shock_path = 10 * 0.80 ** np.arange(300)
C_CORE_direct = J_hh["C_CORE"]["p_e_b"] @ shock_path
C_CORE_total = IRFs_decomp["C_CORE"]
C_CORE_GE = C_CORE_total - C_CORE_direct

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(C_CORE_direct[:T_plot], label="Partial equilibrium", linewidth=2)
ax.plot(C_CORE_GE[:T_plot], label="GE feedback", linewidth=2)
ax.plot(C_CORE_total[:T_plot], label="Total", linestyle="--", linewidth=2.5)
ax.axhline(0, linewidth=0.8)
ax.set_xlabel("Quarters")
ax.set_ylabel("Deviation from steady state")
ax.set_title("Core Consumption Response to Energy Price Shock")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{IRF_DIR}/C_CORE_decomposition.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
# =============================================================================
# 15. HEADLINE VS. CORE INFLATION TARGETING
# =============================================================================
# Reuses IRFs_p_e_b from Section 7 (identical shock/ss/unknowns_td — no need
# to recompute it).

IRFs_p_e_b_headline = plot_linear_irfs(
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank_headline, ss=ss_hank_headline, outputs=outputs, titles=names_outputs,
    figsize=(12, 9), plot=PLOT_ALL,
)

show_irfs(
    [IRFs_p_e_b, IRFs_p_e_b_headline], ["p_e_b", "pi_core", "pi_headline", "i", "r"] + outputs,
    titles=["Energy price shock", "pi_core", "pi_headline", "Nom. IR", "Real IR"] + names_outputs,
    labels=["Taylor Rule (core)", "Taylor Rule (headline)"],
    figsize=(12, 9), save_path=f"{IRF_DIR}/irfs_headline.png",
)


# %%
# =============================================================================
# 16. PARAMETER SENSITIVITY
# =============================================================================

# --- 16a: interest-rate smoothing (rho_i), dynamics-only -> no SS re-solve ---
irfs_rho_i = compare_irfs_by_parameter(
    param_name="rho_i", param_values=[0.0, 0.8],
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank_headline, ss=ss_hank_headline,
    outputs=["pi_core", "pi_headline", "i", "r"] + outputs,
    titles=["pi_core", "pi_headline", "Nom. IR", "Real IR"] + names_outputs,
    figsize=(12, 9), resolve_ss=False, plot=True,
    save_path=f"{IRF_DIR}/irfs_rho_i.png",
)

# --- 16b: beta heterogeneity spread, affects the SS -> full re-solve ---
results_beta_spread = compare_irfs_by_parameter(
    param_name="beta_spread", param_values=[0.0, 0.02, 0.03],
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td, targets_td=targets_td,
    ha=hank, ss=ss_hank,
    hank_ss=hank_ss, unknowns_ss=unknowns_ss, targets_ss=targets_ss,
    calibration=baseline_calibration,
    outputs=["pi_core", "i", "r"] + outputs,
    titles=["pi_core", "Nom. IR", "Real IR"] + names_outputs,
    figsize=(12, 9), resolve_ss=True, plot=True,
    save_path=f"{IRF_DIR}/irfs_beta_het.png",
)


#%%
unknowns_td_leak = ["Tax_NFA", "Tax", "Y", "N", "piw"]
targets_td_leak = ["asset_mkt", "rsrce_cstrt", "GBC", "wnkpc", "labor_mkt"]

outputs = ["C", "C_CORE", "AD","Y", "piw", "Tax", "Tax_NFA", "B", "rsrce_cstrt", "asset_mkt","pi_headline","C_E_B"]


ss_hank_real['leakage_E'] = 0

# %%
results_leakage = compare_irfs_by_parameter(
    param_name="leakage_E", param_values=[0, 0.5, 0.8, 1],
    shocks_list=["p_e_b"], e={"p_e_b": 10}, rho={"p_e_b": 0.80},
    unknowns_td=unknowns_td_leak, targets_td=targets_td_leak,
    ha=hank_leak, ss=ss_hank_real,
    hank_ss=hank_leak, unknowns_ss=unknowns_ss, targets_ss=targets_ss,
    calibration=baseline_calibration,
    outputs=outputs,
    figsize=(12, 9), resolve_ss=False, plot=True
)
# %%
