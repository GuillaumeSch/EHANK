#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, rsrce_cstrt, nkpc, inflation, taylor_rule
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
#%matplotlib qt


# %%
# === Calibration dictionary ===
baseline_calibration = {

    # -------------------------
    # 1. Preferences
    # -------------------------
    "beta": 0.959,          # Discount factor
    "eis": 0.5,             # Elasticity of intertemporal substitution
    "gamma": 1/0.5,         # Risk aversion (CRRA parameter)
    "omega": 0.78,          # Weight of nondurable consumption vs durable services
    "taste_shock": 1e-1,    # Idiosyncratic taste shock
    "dbar": 1,              # Subsistence level of durable services

    # Labor disutility
    "frisch": 1,            # Frisch elasticity of labor supply
    "vphi": 1,              # Scale of disutility from work

    # -------------------------
    # 2. Productivity (idiosyncratic)
    # -------------------------
    "rho_e": 0.95,          # Persistence of productivity shocks
    "sd_e": 0.5,            # Std. dev. of productivity shocks
    "n_e": 5,               # Number of grid points

    # -------------------------
    # 3. Asset grid
    # -------------------------
    "min_a": 0.0,           # Minimum asset holdings
    "max_a": 100,           # Maximum asset holdings
    "n_a": 20,              # Grid size

    # -------------------------
    # 4. Labor and production
    # -------------------------
    "N": 1,                 # Total labor supply
    "Y": 1,                 # Total economy-wide output

    "alpha": 1,             # Labor share in production
    "Z": 1,                 # Productivity in core goods
    "p_core": 1,            # Price of nondurable consumption good
    "Div": 0,               # Dividends (if firms distribute profit)
    "w": 1,                 # Wage

    # -------------------------
    # 5. Durable goods (b = brown, g = green)
    # -------------------------
    "n_b": 2,               # Number of brown vintages
    "n_g": 2,               # Number of green vintages
    "chi": 0.5,             # Resale loss when selling a durable

    # Utility from durables
    "gamma_b": 1.0,         # Utility weight of brown durable
    "gamma_g": 1.2,         # Utility weight of green durable
    "dep_util_frac_b": 0.7, # Depreciation (utility) for oldest brown vintage
    "dep_util_frac_g": 0.7, # Same for green vintage

    # Physical lifetime
    "lifetime_b": 60,       # Brown car durability (quarters)
    "lifetime_g": 60,       # Green car durability (quarters)

    # Durable prices and quantities (per vintage)
    "p_d0": 1.0, "p_d1": 1.0, "p_d2": 1.0, "p_d3": 1.0, "p_d4": 1.0,
    # Durable quantities
    "kappa_d0": 0, "kappa_d1": 0.02, "kappa_d2": 0.02, "kappa_d3": 0.02, "kappa_d4": 0.02,

    # -------------------------
    # 6. Government
    # -------------------------
    "B": 4,                 # Government debt
    "G": 0.3,               # Government spending
    "Tax": 0.358,           # Lump-sum tax
    "tau": 0,               # Labor income tax rate

    # -------------------------
    # 7. Energy (brown vs green)
    # -------------------------
    "p_e_b": 0.333,         # Brown energy price (gasoline)
    "p_e_g": 0.04,          # Green energy price (electricity)

    "eps_b": 0.40,          # Inefficiency of brown vintage
    "eps_g": 0.20,          # Inefficiency of green vintage

    "tau_b": 0.202,         # Carbon tax applied to brown energy
    "tau_g": 0,             # Subsidy for green energy

    # -------------------------
    # 8. Prices and substitution structure
    # -------------------------
    "xi": 0.97,             # Share parameter for core goods in consumption
    "nu": 0.4,              # Elasticity of substitution core vs energy
    "markup_ss": 1,         # Steady-state markup

    # -------------------------
    # 9. Financial environment
    # -------------------------
    "r": 0.06 / 4,          # Quarterly interest rate
}


#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in baseline_calibration.items():
    globals()[k] = v

#%% === Create the model ===
#ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, carbon_tax], name="Simple HA Model")
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, rsrce_cstrt], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))


#%%
ss_0 = ha.steady_state(baseline_calibration)

#%%
unknowns_ss = {'beta':baseline_calibration['beta'],'Tax':baseline_calibration['Tax']}
targets_ss = {'asset_mkt':0.0,'GBC': 0.0}

ss_0 = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
check_resource_constraint(ss_0)

#%%
plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.01},
    rho = {"tau_b": 0.80},
    unknowns_td=['G','Tax'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha,
    ss=ss_0,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N", "D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "i", "r"],
    #titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b.png',
)


#%%
unknowns_ss = {'kappa_d3':0.02}
targets_ss = {'D_G':0.15}

ss_1 = ha.solve_steady_state(ss_0 , unknowns_ss, targets_ss, solver='hybr')
display_ss_durables(ss_1)
display_calibrated_from_unknowns(ss_1, unknowns_ss)


#%% Basic steady state solution
unknowns_ss = {'N': 0.6,'beta':0.94, 'Tax':0.258}
targets_ss = {'labor_mkt':0.0, 'asset_mkt':0.0, 'GBC': 0.0}

ss_0 = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')

#%%
check_resource_constraint(ss_0)


#%% C_CORE = Y_core
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, rsrce_cstrt, get_demand], name="Simple HA Model")

unknowns_ss = {'Y_core':0.6830348630484795, 'Y_d1':0.01, 'Y_d3':0.01}
targets_ss = {'diff_core':0.0, 'diff_BN':0.0, 'diff_GN':0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')

#%%
unknowns_ss = {'Z_d2':30, 'Z_d4':1}
targets_ss = {'diff_BO':0.0, 'diff_GO':0.0}

ss_2 = ha.solve_steady_state(ss , unknowns_ss, targets_ss, solver='hybr')

#%%
unknowns_ss = {'Y_core':0.6830348630484795, 'Y_d1':0.01, 'Y_d3':0.01}
targets_ss = {'diff_core':0.0, 'diff_BN':0.0, 'diff_GN':0.0}

ss_3 = ha.solve_steady_state(ss_2 , unknowns_ss, targets_ss, solver='hybr')

#%%
unknowns_ss = {'r':0.03594929476559624, 'Z_d2':21, 'Z_d4':13}
targets_ss = {'asset_mkt':0.0, 'diff_BO':0.0, 'diff_GO':0.0}

ss_4 = ha.solve_steady_state(ss_3 , unknowns_ss, targets_ss, solver='hybr')

#%%
unknowns_ss = {'r':ss_4['r'], 'Z_d2':ss_4['Z_d2'], 'Z_d4':ss_4['Z_d4'], 'Tax':0.358}
targets_ss = {'asset_mkt':0.0, 'diff_BO':0.0, 'diff_GO':0.0, 'GBC': 0.0 }

ss_5 = ha.solve_steady_state(ss_4 , unknowns_ss, targets_ss, solver='hybr')

#%%
evaluate_param_changes('N', [0.80], ha, ss_5,
                      ss_vars = ['asset_mkt', 'diff_core','diff_BN', 'diff_GN', 'diff_BO', 'diff_GO', 'labor_mkt'])

#%%
unknowns_ss = {'N': 0.8}
targets_ss = {'labor_mkt':0.0}

ss_6 = ha.solve_steady_state(ss_5 , unknowns_ss, targets_ss, solver='hybr')

#%%
unknowns_ss = {'N': ss_6['N'],'beta':0.94}
targets_ss = {'labor_mkt':0.0, 'asset_mkt':0.0}

ss_7 = ha.solve_steady_state(ss_6 , unknowns_ss, targets_ss, solver='hybr')

#%%
ss_7_mod = ss_7.copy()
ss_7_mod['Y_core'] = 0.65

unknowns_ss = {'N': ss_7['N'],'beta':ss_7['beta']}
targets_ss = {'labor_mkt':0.0, 'asset_mkt':0.0}

ss_8 = ha.solve_steady_state(ss_7_mod , unknowns_ss, targets_ss, solver='hybr')


#%%
evaluate_param_changes('beta', [0.94, 0.959, 0.960], ha, ss_6,
                      ss_vars = ['asset_mkt', 'diff_core','diff_BN', 'diff_GN', 'diff_BO', 'diff_GO', 'labor_mkt'])

#%%
# #%% Evaluate the model at the calibration with differences in a parameter.
evaluate_param_changes('r', [0.016, 0.02, 0.0], ha, ss_3,
                      ss_vars = ['asset_mkt', 'diff_core','diff_BN', 'diff_GN', 'diff_BO', 'diff_GO', 'labor_mkt'])

#%%
drawdag(ha)
#%% 

ss_baseline = ha.steady_state(baseline_calibration)
ss_baseline.toplevel

check_resource_constraint(ss_baseline)


#%%

# #%% Evaluate the model at the calibration with differences in a parameter.
evaluate_param_changes('r', [0.016], ha, baseline_calibration,
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
                      #ss_vars = ['B', 'labor_mkt','asset_mkt', 'GBC','C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])
                      ss_vars = ['B', 'labor_mkt','asset_mkt', 'rsrce_cstrt', 'GBC','C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])
# %%
evaluate_param_changes('w', [0.99, 1.01], ha, ss,
                      #ss_vars = ['B', 'labor_mkt','asset_mkt', 'rsrce_cstrt', 'GBC','C', 'A', 'D_N', 'D_BN','D_BO','D_GN','D_GO','AGG_TRANSF', 'C_CORE', 'C_E', 'T_E'])
                      ss_vars = ['AD', 'AD_CORE','AD_DURABLES', 'AS', 'AS_CORE','AS_DURABLES', 'rsrce_cstrt'])
# %%
evaluate_param_changes('Y_core', [0.59, 0.61, 0.7], ha, ss,
#                      ss_vars = ['AD', 'AD_CORE','AD_DURABLES', 'AS', 'AS_CORE','AS_DURABLES', 'rsrce_cstrt'])
                      ss_vars = ['AD', 'AS','rsrce_cstrt','labor_mkt','asset_mkt'])
# %%
evaluate_two_param_changes('Y_core', [0.59, 0.61, 0.7], 'w', [0.99, 1.01], ha, ss,
#                      ss_vars = ['AD', 'AD_CORE','AD_DURABLES', 'AS', 'AS_CORE','AS_DURABLES', 'rsrce_cstrt'])
                      ss_vars = ['AD', 'AS','rsrce_cstrt','labor_mkt','asset_mkt'])
# %%
