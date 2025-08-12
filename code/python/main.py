#%%
from HH_Durables_Block import hh
from Model_Blocks import fiscal, mkt_clearing, prod
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *

# %%
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
    "p_core": 1,               # Price of core, non-durable goods
    "Div": 0,                  # Dividends from firms
    "Tax": 0.5,                # Total tax
    #Government
    #"Y" : 1,                   # Output
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
    #Energy
    "p_e_b" : 0.3,               # Price of brown energy (gas/petrol)
    "p_e_g" : 0.3,               # Price of green energy (electricity)
    "tau_b" : 0,               # Carbon tax
    "tau_g" : 0,               # Green energy subsidy
    #Prices
    "xi" : 0.80,             # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
    "nu" : 0.4,#0.01,                   # Elasticity of substitution between core and energy consumption
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in cali["baseline"].items():
    globals()[k] = v
    
#Import the DD calibration (Some parameters are not present in the DD calibration.)
with open('calibration_ss_DD.json', 'r') as f:
    cali_DD = json.load(f)
    
cali['baseline_DD'] = deepcopy(cali['baseline'])
for key in cali_DD.keys():
    if key in cali['baseline_DD']:
        cali['baseline_DD'][key] = cali_DD[key]
    

#%% === Create the model ===
ha = sj.create_model([hh, fiscal, mkt_clearing, prod], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))


#%% === Solve the model Steady State ===
unknowns_ss = {'B':(0.80,0.999)}
targets_ss = {'asset_mkt'}

ss_DD = ha.solve_steady_state(cali['baseline_DD'] , unknowns_ss, targets_ss, solver='hybr')
display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

#%% === Solve the model transition dynmamics and get IRFs === §
plot_linear_irfs(
    shocks_list=['tau_b'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"goods_mkt"],
    ha=ha,
    ss=ss_DD,
    outputs=["tau_b","N", "G", "Tax","D_BO", "D_BN", "D_GO", "D_GN", "goods_mkt", "asset_mkt"]
)
# %%
