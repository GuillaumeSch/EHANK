#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, prod_durables, rsrce_cstrt_alt, nkpc, inflation, taylor_rule
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
#%matplotlib qt


# %%
# === Calibration dictionary ===
baseline_calibration = {
    # Preferences and taste shocks
    "taste_shock": 1e-1,       # Idiosyncratic taste shock
    "beta": 0.959,             # Discount factor
    "eis": 0.5,                # Elasticity of intertemporal substitution
    "gamma": 1/0.5,            # Relative risk aversion (curvature)
    "omega": 0.78,             # Relative weight of consumption c versus durable goods ˜d in the utility function
    "dbar": 1,                 # Subsistence level of durable goods
    "r": 0.06 / 4,             # Interest rate (quarterly)
    "N": 1,                    # Total labor supply
    "frisch": 1,               # Frisch elasticity
    "vphi": 1,                 # Disutility of work
    # Productivity process
    "rho_e": 0.95,             # Persistence of productivity shocks
    "sd_e": 0.5,               # Std. deviation of productivity shocks
    "n_e": 5,                  # Number of productivity grid points
    # Asset grid
    "min_a": 0.0,              # Minimum asset level
    "max_a": 100,              # Maximum asset level
    "n_a": 20,                 # Number of asset grid points
    # Labor market
    "N_core": 0.6,             # Labor demand for core goods
    "Y_core": 0.6,             # Labor demand for core goods
    "N_d": np.array([ 0.1, 0.1, 0.1, 0.1]),                # Labor demand for durable goods
    "tau":0,                   # Labor income tax
    # Durable goods
    "n_b": 2,                  # Number of brown vintages
    "n_g": 2,                  # Number of green vintages
    "chi": 0.5,                # Resale loss (fraction)
    "gamma_b": 1.0,            # Utility from brown durable
    "dep_util_frac_b": 0.7,    # Depreciation utility brown (Fraction of oldest vintage relative to newest)
    "gamma_g": 1.2,            # Utility from green durable
    "dep_util_frac_g": 0.7,    # Depreciation utility green (Fraction of oldest vintage relative to newest)
    "lifetime_b": 60,          # Average lifetime of brown durables (quarters)
    "lifetime_g": 60,          # Average lifetime of green durables (quarters)
    # Firms
    "alpha": 1,                # Share of labor in prod. function
    #"p_core": 1,               # Price of core, non-durable goods
    "Div": 0,                  # Dividends from firms
    "Z_core": 1,               # Core productivity
    "Z_d1": 15, "Z_d2": 60, "Z_d3": 10, "Z_d4": 1, # Durables productivities
    #Government
    "B" : 4,                   # Stock of  debt
    "G" : 0.3,                 # Government spendings
    "Tax": 0.358,              # Lump-sum tax
    #Energy
    "p_e_b" : 0.333,           # Price of brown energy (gas/petrol)
    "p_e_g" : 0.04,            # Price of green energy (electricity)
    "eps_b" : 0.40,            # Linear inefficiency of brown car vintages
    "eps_g" : 0.20,            # Linear inefficiency of green car vintages
    "tau_b" : 0.202,            # Carbon tax (no free. Dependend of T_E)
    "tau_g" : 0,               # Green energy subsidy
    #Prices
    "xi" : 0.97,               # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
    "nu" : 0.4,                # Elasticity of substitution between core and energy consumption
    "markup_ss" : 1,
    "w": 1,
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in baseline_calibration.items():
    globals()[k] = v

#%% === Create the model ===
#ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, carbon_tax], name="Simple HA Model")
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, rsrce_cstrt_alt], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))

#%% 

ss_baseline = ha.steady_state(baseline_calibration)
ss_baseline.toplevel

check_resource_constraint(ss_baseline)


#%%

# #%% Evaluate the model at the calibration with differences in a parameter.
evaluate_param_changes('r', [0.10/4], ha, baseline_calibration,
                      ss_vars = ['B', 'labor_mkt','asset_mkt', 'C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])

#%%
evaluate_two_param_changes('xi', [0.90, 0.95],'beta', [0.90, 0.95], ha, baseline_calibration,
                      ss_vars = ['B', 'labor_mkt','asset_mkt', 'C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])

#%%
t_start = time.perf_counter()

unknowns_ss = {'beta':0.959, 'N':1, 'Tax':0.358}
targets_ss = {'asset_mkt':0.0, 'labor_mkt':0.0, 'GBC': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')

display_ss_durables(ss)
display_calibrated_from_unknowns(ss, unknowns_ss)
check_resource_constraint(ss)


t_end = time.perf_counter()
print(f"total block runtime (solve + displays): {t_end - t_start:.3f} s")
# %%
evaluate_param_changes('N', [1.01], ha, ss,
                      ss_vars = ['B', 'labor_mkt','asset_mkt', 'GBC','C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])
# %%
evaluate_param_changes('w', [0.99, 1.01], ha, ss,
                      ss_vars = ['B', 'labor_mkt','asset_mkt', 'GBC','C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])
# %%
evaluate_param_changes('Y_core', [0.59, 0.61], ha, ss,
                      ss_vars = ['B', 'labor_mkt','asset_mkt', 'GBC','C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])
# %%
