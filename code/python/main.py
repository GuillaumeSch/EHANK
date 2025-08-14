#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, prod_durables
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
    "N": 1,                    # Total labor supply
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
    "N_core": 0.5,             # Labor demand for core goods
    "N_d1_b": 1.0, "N_d2_b": 1.0, "N_d3_b": 1.0, "N_d4_b": 1.0,                # Labor demand for durable goods
    "mu_N_d": 0.05,            # Fraction that is applied to all labor demand for durables.
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
    "dep_util_frac_b": 1,      # Depreciation utility brown (Fraction of oldest vintage relative to newest)
    "gamma_g": 1.2,            # Utility from green durable
    "dep_util_frac_g": 1,      # Depreciation utility green (Fraction of oldest vintage relative to newest)
    "lifetime_b": 60,          # Average lifetime of brown durables (quarters)
    "lifetime_g": 60,          # Average lifetime of green durables (quarters)
    # Firms
    "alpha": 1,                # Share of labor in prod. function
    "p_core": 1,               # Price of core, non-durable goods
    "Div": 0,                  # Dividends from firms
    #"Tax": 0.5,                # Total tax
    "Z_core": 1,               # Core productivity
    "Z_d1": 1, "Z_d2": 1, "Z_d3": 1, "Z_d4": 1, # Durables productivities
    #Government
    #"Y" : 1,                  # Output
    "B" : 1,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
    #Energy
    "p_e_b" : 0.3,             # Price of brown energy (gas/petrol)
    "p_e_g" : 0.3,             # Price of green energy (electricity)
    "tau_b" : 0,               # Carbon tax
    "tau_g" : 0,               # Green energy subsidy
    #Prices
    "xi" : 0.8,                # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
    "nu" : 0.4,                # Elasticity of substitution between core and energy consumption
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
    
#Import the DD calibration (Some parameters are not present in the DD calibration.)
with open('calibration_ss_DD.json', 'r') as f:
    cali_DD = json.load(f)
    
cali['baseline_DD'] = deepcopy(cali['baseline'])
for key in cali_DD.keys():
    if key in cali['baseline_DD']:
        cali['baseline_DD'][key] = cali_DD[key]
    
for k, v in cali["baseline_DD"].items():
    globals()[k] = v

#%% === Create the model ===
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))



#%%
evaluate_param_changes('N', [0.60, 0.7, 0.8, 0.99], ha, cali['baseline'], 
                           ss_vars=['N_core', 'labor_mkt','asset_mkt', 'Tax',
                                    'A', 'B', 'N', 'C', 'C_CORE', 'C_E',])

#%% === Solve the model Steady State ===
unknowns_ss = {'N':(0.60,0.8),'mu_N_d':(0,1)}
targets_ss = {'asset_mkt':0,'labor_mkt':0}

#ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')
ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='broyden_custom')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

#%%
evaluate_param_changes('N', [0.60, 0.7, 0.8, 0.99], ha, ss_DD, 
                           ss_vars=['N_core', 'labor_mkt','asset_mkt', 'Tax',
                                    'A', 'B', 'N', 'C', 'C_CORE', 'C_E',])

#%%
cali_DD_alt = ss_DD.copy()
cali_DD_alt['Z_d1'] = 0.5

unknowns_ss = {'B':(0.80,0.999)}
targets_ss = {'asset_mkt'}

ss_DD_alt = ha.solve_steady_state(cali_DD_alt , unknowns_ss, targets_ss, solver='hybr')
display_ss_durables(ss_DD_alt)
display_calibrated_from_unknowns(ss_DD, ss_DD_alt)


#%%
#Check that the decomposition is correct.
ss_DD['C_P'] - (ss_DD['C_CORE_P_CORE'] + ss_DD['C_E_P_E'])


D = ss_DD.internals['hh']['consav']['D']

c = ss_DD.internals['hh']['consav']['c']
c_core = ss_DD.internals['hh']['consav']['c_core']
c_E = ss_DD.internals['hh']['consav']['c_E']

p_core = p_core
p_e = ss_DD.internals['hh']['p_e']
p_d = ss_DD.internals['hh']['p_d']
p_bundle = ss_DD.internals['hh']['p_bundle']





P = ss_DD.internals['hh']['durables']['law_of_motion'].P

P[:,:,:,:]

D[:,0,0,0]


xplus = P.copy()  # Create a copy to avoid modifying the original
for i in range(xplus.shape[2]):  # Loop over 3rd dimension
    for j in range(xplus.shape[3]):  # Loop over 4th dimension
        np.fill_diagonal(xplus[:,:,i,j], 0)
        
xminus = P.copy()  # Create a copy to avoid modifying the original
for i in range(xminus.shape[2]):  # Loop over 3rd dimension
    for j in range(xminus.shape[3]):  # Loop over 4th dimension
        np.fill_diagonal(xminus[:,:,i,j], 0)

Xplus = np.sum(xplus * D, axis=(1,2,3))

Xminus = np.sum(xminus * D, axis=(0,2,3))



C_core_p_core = np.sum(D * c_core * p_core)
C_E_p_e = np.sum(D * c_E * np.array(p_e)[..., np.newaxis, np.newaxis, np.newaxis])
np.sum(D * c * np.array(p_bundle)[..., np.newaxis, np.newaxis, np.newaxis])





ss_DD['C_CORE_P_CORE'] + C_E_p_e + np.sum(p_d * (Xplus - (1-chi)*Xminus))

ss_DD['w'] * N + ss_DD['r'] * ss_DD['A'] + -ss_DD['Tax']




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
